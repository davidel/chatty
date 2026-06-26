import json
import logging
import os
import re
import sys
import time
import uuid
import weakref
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

import openai
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.columns import Columns
from rich.markup import escape

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from chatty.utils import (
  count_tokens,
  truncate_output,
  get_ollama_models,
  load_system_prompt_from_file,
  parse_frontmatter,
  record_command_binaries,
  sanitize_tool_output
)
from chatty.tools import execute_tool, TOOLS_SCHEMA

logger = logging.getLogger("chatty")
console = Console()


@dataclass
class SessionConfig:
  provider: str
  model: str
  context_size: int = 8192
  sandbox: str = "./sandbox"
  api_key: Optional[str] = None
  url: Optional[str] = None
  max_loops: int = 20
  system_prompt_override: Optional[str] = None
  prompt_mode: str = "replace"
  skills_paths: List[str] = field(default_factory=list)
  max_read_chars: int = 40000
  max_grep_results: int = 100
  max_command_chars: int = 16000
  max_history_tool_chars: int = 1000
  history_keep_messages: int = 4
  max_url_chars: int = 24000
  max_dir_items: int = 200
  static_skills: Optional[bool] = None
  prompt_caching: bool = False


class LazyMarkdown:
  """A helper that wraps a Markdown string and only parses it when rendered.

  This prevents high CPU usage caused by parsing Markdown on every LLM token chunk.
  """

  def __init__(self, text: str):
    self.text = text

  def __rich_console__(self, console: Console, options: Any) -> Any:
    md = Markdown(self.text)
    return md.__rich_console__(console, options)

  def __rich_measure__(self, console: Console, options: Any) -> Any:
    md = Markdown(self.text)
    return md.__rich_measure__(console, options)


class ChatbotSession:
  _active_session = None

  def __init__(
    self,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    context_size: Optional[int] = None,
    sandbox: Optional[str] = None,
    api_key: Optional[str] = None,
    url: Optional[str] = None,
    max_loops: int = 20,
    system_prompt_override: Optional[str] = None,
    prompt_mode: str = "replace",
    skills_paths: Optional[List[str]] = None,
    max_read_chars: int = 40000,
    max_grep_results: int = 100,
    max_command_chars: int = 16000,
    max_history_tool_chars: int = 1000,
    history_keep_messages: int = 4,
    max_url_chars: int = 24000,
    max_dir_items: int = 200,
    static_skills: Optional[bool] = None,
    prompt_caching: bool = False,
    config: Optional[SessionConfig] = None
  ):
    ChatbotSession._active_session = self
    self.tool_calls_count: Dict[str, int] = {}
    self.external_binaries_count = 0
    self.external_binaries_breakdown: Dict[str, int] = {}
    
    if config is not None:
      self.config = config
    else:
      self.config = SessionConfig(
        provider=provider,
        model=model,
        context_size=context_size if context_size is not None else 8192,
        sandbox=sandbox if sandbox is not None else "./sandbox",
        api_key=api_key,
        url=url,
        max_loops=max_loops,
        system_prompt_override=system_prompt_override,
        prompt_mode=prompt_mode,
        skills_paths=skills_paths or [],
        max_read_chars=max_read_chars,
        max_grep_results=max_grep_results,
        max_command_chars=max_command_chars,
        max_history_tool_chars=max_history_tool_chars,
        history_keep_messages=history_keep_messages,
        max_url_chars=max_url_chars,
        max_dir_items=max_dir_items,
        static_skills=static_skills,
        prompt_caching=prompt_caching
      )

    # Ensure static_skills defaults correctly if not provided
    if self.config.static_skills is None:
      self.config.static_skills = (self.config.provider == "openrouter")

    # Ensure sandbox path is absolute
    self.config.sandbox = os.path.abspath(self.config.sandbox)

    self.background_commands = {}
    self.next_task_id = 1
    self.max_completed_tasks = 10
    
    # Register weakref finalizer for automatic cleanup on garbage collection or program exit
    self._finalizer = weakref.finalize(self, ChatbotSession._cleanup_resources, self.background_commands)
    
    # Internal state
    self.messages: List[Dict[str, Any]] = []
    self.current_loop = 0
    default_prompt = (
      "You are a helpful assistant with local sandboxed file access and shell execution capabilities.\n"
      "You have tools for: listing directories (list_dir), locating files (locate_files), checking file info (get_file_info), reading files (read_file), writing files (write_file), copying files/directories (copy_file), moving/renaming files/directories (move_file), deleting files (delete_file), deleting directories (delete_directory), creating directories (make_directory), formatting files (format_file), patching files (patch_file), applying multiple patches (multi_patch), editing line ranges (edit_lines), applying multiple line range edits (multi_edit_lines), searching regex patterns (search_grep), fetching web content (fetch_url), executing shell commands (run_command), checking background tasks (check_background_command), terminating background processes (kill_process), sleeping (sleep), and asking questions (ask_question).\n"
      "All paths provided to the tools will resolve relative to the sandbox directory.\n"
      "You are strictly prohibited from writing files outside the sandbox folder.\n"
      "CRITICAL: When you need to ask the user a question, clarify instructions, confirm decisions, or present a set of choices/options, you MUST use the dedicated 'ask_question' tool instead of asking questions in your conversational text response. This allows the CLI to prompt the user interactively and return their response to you in the tool execution loop.\n"
      "CRITICAL: You MUST use the dedicated, high-level filesystem tools (like list_dir, read_file, search_grep, locate_files, get_file_info, copy_file, move_file, delete_file, delete_directory, make_directory) instead of running command-line utilities (like grep, find, cat, head, tail, sed, awk, less, more, cp, mv, rm, rmdir, mkdir, ls) inside run_command. Shell execution using run_command is blocked for these actions and will return an error. You must use get_file_info instead of running 'wc' or 'wc -l' inside run_command.\n"
      "CRITICAL: For performing search-and-replace edits (similar to 'sed'), you MUST use 'multi_patch' (for multiple non-contiguous exact replacements), 'edit_lines' (for a single line number range), or 'multi_edit_lines' (for multiple non-contiguous line range edits) instead of using 'sed' or custom scripts in run_command.\n"
      "CRITICAL: When you need to reformat source code files or enforce layout/style guidelines (such as indentation, line-splitting, or spacing), you MUST use the dedicated 'format_file' tool instead of manually editing the files using 'edit_lines', 'patch_file', or 'multi_patch'.\n"
      "CRITICAL: You are strictly prohibited from using the shell 'sleep' command inside run_command to pause execution. You MUST use the dedicated 'sleep' tool instead.\n"
      "When running shell commands using run_command, if a command takes longer than 10 seconds, it will automatically transition to run in the background and return a 'Task ID'. You must NOT block. Instead, check its output or wait for its progress/completion by calling check_background_command with the Task ID and a timeout parameter. You are strictly prohibited from using the 'sleep' tool to wait for background commands; you MUST use check_background_command with a timeout parameter instead. Perform other file tasks (read, patch, edit) while waiting.\n"
      "To filter the output of run_command, use its optional 'output_filter' (regex), 'tail_lines', or 'head_lines' parameters rather than piping to grep or writing custom filtering scripts.\n"
      "When compilation, testing, verification, or running tools (like verilator, python scripts, compilers) is needed, you MUST execute them directly using the run_command tool instead of instructing the user to run them manually.\n"
      "Always use your tools proactively to solve tasks directly."
    )
    
    if self.config.system_prompt_override:
      if self.config.prompt_mode == "integrate":
        self.system_prompt = default_prompt + "\n\n" + self.config.system_prompt_override
      else:
        self.system_prompt = self.config.system_prompt_override
    else:
      self.system_prompt = default_prompt
    self.multiline_mode = False
    self.client = None
    
    # Ensure sandbox exists
    os.makedirs(self.sandbox, exist_ok=True)
    os.chdir(self.sandbox)
    
    # Initialize client
    self.init_client()
    
    # Initialize and register commands registry
    self._commands = {}
    self._register_commands()
    
    # Load active skills
    self.skills = {}
    self.load_skills()
    logger.info(f"ChatbotSession initialized. Provider: {self.provider}, Model: {self.model}, Sandbox: {self.sandbox}")

  @property
  def provider(self) -> str:
    return self.config.provider

  @provider.setter
  def provider(self, val: str):
    self.config.provider = val

  @property
  def model(self) -> str:
    return self.config.model

  @model.setter
  def model(self, val: str):
    self.config.model = val

  @property
  def context_size(self) -> int:
    return self.config.context_size

  @context_size.setter
  def context_size(self, val: int):
    self.config.context_size = val

  @property
  def sandbox(self) -> str:
    return self.config.sandbox

  @sandbox.setter
  def sandbox(self, val: str):
    self.config.sandbox = val

  @property
  def api_key(self) -> Optional[str]:
    return self.config.api_key

  @api_key.setter
  def api_key(self, val: Optional[str]):
    self.config.api_key = val

  @property
  def url(self) -> Optional[str]:
    return self.config.url

  @url.setter
  def url(self, val: Optional[str]):
    self.config.url = val

  @property
  def max_loops(self) -> int:
    return self.config.max_loops

  @max_loops.setter
  def max_loops(self, val: int):
    self.config.max_loops = val

  @property
  def skills_paths(self) -> List[str]:
    return self.config.skills_paths

  @skills_paths.setter
  def skills_paths(self, val: List[str]):
    self.config.skills_paths = val

  @property
  def max_read_chars(self) -> int:
    return self.config.max_read_chars

  @max_read_chars.setter
  def max_read_chars(self, val: int):
    self.config.max_read_chars = val

  @property
  def max_grep_results(self) -> int:
    return self.config.max_grep_results

  @max_grep_results.setter
  def max_grep_results(self, val: int):
    self.config.max_grep_results = val

  @property
  def max_command_chars(self) -> int:
    return self.config.max_command_chars

  @max_command_chars.setter
  def max_command_chars(self, val: int):
    self.config.max_command_chars = val

  @property
  def max_history_tool_chars(self) -> int:
    return self.config.max_history_tool_chars

  @max_history_tool_chars.setter
  def max_history_tool_chars(self, val: int):
    self.config.max_history_tool_chars = val

  @property
  def history_keep_messages(self) -> int:
    return self.config.history_keep_messages

  @history_keep_messages.setter
  def history_keep_messages(self, val: int):
    self.config.history_keep_messages = val

  @property
  def max_url_chars(self) -> int:
    return self.config.max_url_chars

  @max_url_chars.setter
  def max_url_chars(self, val: int):
    self.config.max_url_chars = val

  @property
  def max_dir_items(self) -> int:
    return self.config.max_dir_items

  @max_dir_items.setter
  def max_dir_items(self, val: int):
    self.config.max_dir_items = val

  @property
  def static_skills(self) -> bool:
    return self.config.static_skills

  @static_skills.setter
  def static_skills(self, val: bool):
    self.config.static_skills = val

  @property
  def prompt_caching(self) -> bool:
    return self.config.prompt_caching

  @prompt_caching.setter
  def prompt_caching(self, val: bool):
    self.config.prompt_caching = val

  def load_skills(self):
    """Scans all configured skills directories and loads/merges valid skill definitions."""
    self.skills = {}
    search_dirs = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_skills_dir = os.path.join(script_dir, "skills")
    if os.path.exists(default_skills_dir) and os.path.isdir(default_skills_dir):
      search_dirs.append(default_skills_dir)
      
    env_paths = os.environ.get("CHATBOT_SKILLS_PATH", "")
    if env_paths:
      for p in env_paths.split(os.pathsep):
        p = p.strip()
        if p and os.path.exists(p) and os.path.isdir(p):
          search_dirs.append(p)
          
    for p in self.skills_paths:
      p = p.strip()
      if p and os.path.exists(p) and os.path.isdir(p):
        search_dirs.append(p)
        
    sandbox_skills_dir = os.path.join(self.sandbox, "skills")
    if os.path.exists(sandbox_skills_dir) and os.path.isdir(sandbox_skills_dir):
      search_dirs.append(sandbox_skills_dir)
      
    unique_dirs = []
    for d in search_dirs:
      abs_d = os.path.abspath(d)
      if abs_d not in unique_dirs:
        unique_dirs.append(abs_d)
        
    for skills_dir in unique_dirs:
      try:
        for item in os.listdir(skills_dir):
          item_path = os.path.join(skills_dir, item)
          if os.path.isdir(item_path):
            skill_md_path = os.path.join(item_path, "SKILL.md")
            if os.path.exists(skill_md_path):
              try:
                with open(skill_md_path, 'r', encoding='utf-8', errors='ignore') as f:
                  content = f.read()
                meta, body = parse_frontmatter(content)
                if "name" not in meta:
                  meta["name"] = item
                self.skills[item] = {
                  "metadata": meta,
                  "body": body
                }
              except Exception:
                pass
      except Exception:
        pass
    logger.info(f"Loaded {len(self.skills)} skills: {list(self.skills.keys())}")

  def get_active_system_prompt(self) -> str:
    """Returns system prompt integrated with dynamically activated skills."""
    if self.static_skills:
      if not self.skills:
        return self.system_prompt
      active_skills = []
      for skill_name, skill in sorted(self.skills.items()):
        meta = skill["metadata"]
        active_skills.append(f"### Skill: {meta.get('name')}\n{skill['body']}")
      skills_text = "\n\n".join(active_skills)
      return f"{self.system_prompt}\n\n## Available Skills\n{skills_text}"

    active_skills = []
    active_names = []
    last_user_msg = ""
    for msg in reversed(self.messages):
      if msg.get("role") == "user":
        last_user_msg = msg.get("content") or ""
        break
        
    prompt_lower = last_user_msg.lower()
    
    for skill_name, skill in self.skills.items():
      meta = skill["metadata"]
      name = meta.get("name", "").lower()
      desc = meta.get("description", "").lower()
      tags = meta.get("tags", [])
      if not isinstance(tags, list):
        tags = [tags]
        
      match = False
      if name and name in prompt_lower:
        match = True
      elif any(str(tag).lower() in prompt_lower for tag in tags):
        match = True
        
      if match:
        active_skills.append(f"### Skill: {meta.get('name')}\n{skill['body']}")
        active_names.append(meta.get("name"))
        
    if active_skills:
      logger.info(f"System prompt built with activated skills: {active_names}")
      skills_text = "\n\n".join(active_skills)
      return f"{self.system_prompt}\n\n## Activated Skills\n{skills_text}"
    return self.system_prompt

  def init_client(self):
    """Initializes or updates the OpenAI client based on active settings."""
    if self.provider == "ollama":
      base = self.url or "http://localhost:11434/v1"
      self.client = openai.OpenAI(
        base_url=base,
        api_key="ollama"  # placeholder key
      )
    else:  # openrouter
      base = self.url or "https://openrouter.ai/api/v1"
      key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
      if not key:
        console.print(
          "[bold red]Warning:[/bold red] OpenRouter API key is not configured. "
          "Use [cyan]/api_key <key>[/cyan] or set the [cyan]OPENROUTER_API_KEY[/cyan] environment variable."
        )
        key = "missing_api_key"
      self.client = openai.OpenAI(
        base_url=base,
        api_key=key,
        default_headers={
          "HTTP-Referer": "https://github.com/davidel/chatty",
          "X-Title": "Chatty"
        }
      )

  def tool_run_tests(self, command: str = None) -> str:
    """Run tests in the sandbox, auto-detecting the testing framework if no command is provided."""
    import shutil

    detected_msg = ""
    if not command:
      if os.path.exists(os.path.join(self.sandbox, "pytest.ini")) or \
         os.path.exists(os.path.join(self.sandbox, "conftest.py")) or \
         os.path.isdir(os.path.join(self.sandbox, "tests")) or \
         os.path.isdir(os.path.join(self.sandbox, "test")):
        if shutil.which("pytest"):
          command = "pytest"
        else:
          command = "python -m unittest discover"
          
      elif os.path.exists(os.path.join(self.sandbox, "package.json")):
        if shutil.which("npm"):
          command = "npm test"
          
      elif os.path.exists(os.path.join(self.sandbox, "Cargo.toml")):
        if shutil.which("cargo"):
          command = "cargo test"
          
      elif os.path.exists(os.path.join(self.sandbox, "go.mod")):
        if shutil.which("go"):
          command = "go test ./..."
          
      elif os.path.exists(os.path.join(self.sandbox, "CMakeLists.txt")):
        if shutil.which("ctest"):
          command = "ctest"
        elif shutil.which("make"):
          command = "make test"
          
      elif os.path.exists(os.path.join(self.sandbox, "Makefile")) or \
           os.path.exists(os.path.join(self.sandbox, "makefile")):
        if shutil.which("make"):
          command = "make test"
          
      elif os.path.exists(os.path.join(self.sandbox, "meson.build")):
        if shutil.which("meson"):
          command = "meson test"
        elif shutil.which("ninja"):
          command = "ninja test"
          
      if not command:
        return "Error: Could not auto-detect a test suite. Please specify a custom test 'command' (e.g. 'pytest', 'npm test')."
        
      detected_msg = f"Auto-detected test command: '{command}'\n"

    result = self.tool_run_command(command)
    return detected_msg + result

  def validate_command_safety(self, command: str) -> Optional[str]:
    """Validates that the shell command does not bypass dedicated tools."""
    import shlex
    
    def check_cmd(cmd_str: str) -> Optional[str]:
      cmd_str = cmd_str.strip()
      if not cmd_str:
        return None
        
      try:
        tokens = shlex.split(cmd_str)
      except ValueError:
        tokens = cmd_str.split()
        
      sequencers = {'&&', '||', ';', '&', '(', '{', ')', '}'}
      is_cmd_token = True
      pipe_count = 0
      
      for token in tokens:
        if is_cmd_token:
          clean_token = os.path.basename(token.strip().lower())
          
          if clean_token in {'grep', 'egrep', 'fgrep', 'rgrep'}:
            if pipe_count == 0:
              return (
                f"Error: Using '{token}' directly in run_command is prohibited to search files. "
                "Please use the dedicated 'search_grep' tool instead."
              )
            else:
              return (
                f"Error: Using '{token}' directly in run_command is prohibited to filter output. "
                "Please use the 'output_filter' parameter of run_command instead (which is also supported by its wait-for-termination counterpart, check_background_command)."
              )
          elif clean_token == 'find':
            return (
              f"Error: Using 'find' in run_command is prohibited to locate files. "
              "Please use the dedicated 'locate_files' tool instead."
            )
          elif clean_token in {'kill', 'pkill', 'killall'}:
            return (
              f"Error: Using '{token}' in run_command is prohibited to terminate processes. "
              "Please use the dedicated 'kill_process' tool instead."
            )
          elif clean_token in {'cat', 'less', 'more'}:
            return (
              f"Error: Using '{token}' in run_command is prohibited to view files. "
              "Please use the dedicated 'read_file' tool instead."
            )
          elif clean_token in {'head', 'tail'}:
            if pipe_count == 0:
              return (
                f"Error: Using '{token}' in run_command is prohibited to inspect files. "
                "Please use the dedicated 'read_file' tool with start_line and end_line parameters."
              )
            else:
              return (
                f"Error: Using '{token}' directly in run_command is prohibited to filter output. "
                "Please use the 'head_lines' or 'tail_lines' parameter of run_command instead."
              )
          elif clean_token == 'awk':
            if pipe_count == 0:
              return (
                f"Error: Using 'awk' in run_command is prohibited to inspect files. "
                "Please use the dedicated 'read_file' tool with start_line and end_line parameters."
              )
            else:
              return (
                f"Error: Using 'awk' directly in run_command is prohibited to filter output. "
                "Please use the 'output_filter', 'head_lines', or 'tail_lines' parameters of run_command instead."
              )
          elif clean_token == 'sed':
            return (
              "Error: Using 'sed' in run_command is prohibited. "
              "Please use the dedicated 'multi_patch' tool (for multiple exact replacements), 'edit_lines' (for a single line number range), or 'multi_edit_lines' tool instead."
            )
          elif clean_token == 'sleep':
            return (
              "Error: Using 'sleep' in run_command is prohibited to pause execution. "
              "Please use the dedicated 'sleep' tool instead."
            )
          elif clean_token == 'wc':
            if pipe_count == 0:
              return (
                "Error: Using 'wc' in run_command is prohibited to count lines, words, or bytes in files. "
                "Please use the dedicated 'get_file_info' tool instead."
              )
            else:
              return (
                "Error: Using 'wc' directly in run_command is prohibited to count lines. "
                "Please use dedicated tools like 'get_file_info' or 'locate_files' to check file/directory information instead."
              )
          elif clean_token == 'cp':
            return (
              f"Error: Using '{token}' in run_command is prohibited to copy files or directories. "
              "Please use the dedicated 'copy_file' tool instead."
            )
          elif clean_token == 'mv':
            return (
              f"Error: Using '{token}' in run_command is prohibited to move or rename files or directories. "
              "Please use the dedicated 'move_file' tool instead."
            )
          elif clean_token == 'rm':
            return (
              f"Error: Using '{token}' in run_command is prohibited to delete files or directories. "
              "Please use the dedicated 'delete_file' (for files) or 'delete_directory' (for directories) tool instead."
            )
          elif clean_token == 'rmdir':
            return (
              f"Error: Using '{token}' in run_command is prohibited to delete directories. "
              "Please use the dedicated 'delete_directory' tool instead."
            )
          elif clean_token == 'mkdir':
            return (
              f"Error: Using '{token}' in run_command is prohibited to create directories. "
              "Please use the dedicated 'make_directory' tool instead."
            )
          elif clean_token in {'ls', 'dir'}:
            return (
              f"Error: Using '{token}' in run_command is prohibited to list directory contents. "
              "Please use the dedicated 'list_dir' tool instead."
            )
        
        if token.strip() in {'|', '|&'}:
          is_cmd_token = True
          pipe_count += 1
        elif token.strip() in sequencers:
          is_cmd_token = True
          pipe_count = 0
        else:
          is_cmd_token = False
          
      subshell_patterns = re.findall(r'\$\((.*?)\)|`(.*?)`', cmd_str)
      for match in subshell_patterns:
        for group in match:
          if group:
            err = check_cmd(group)
            if err:
              return err
      return None

    return check_cmd(command)

  def _apply_output_filters(
    self,
    text: str,
    output_filter: Optional[str] = None,
    head_lines: Optional[int] = None,
    tail_lines: Optional[int] = None
  ) -> str:
    """Applies output filter regex, head/tail limits, and joins lines."""
    if not text:
      return text
    lines = text.splitlines()
    if output_filter:
      try:
        pattern = re.compile(output_filter, re.IGNORECASE)
        lines = [line for line in lines if pattern.search(line)]
      except re.error as e:
        return f"Error applying output_filter: {e}"
    if head_lines is not None and head_lines > 0:
      lines = lines[:head_lines]
    if tail_lines is not None and tail_lines > 0:
      lines = lines[-tail_lines:]
    return "\n".join(lines)

  def tool_run_command(self, command: str, output_filter: Optional[str] = None, tail_lines: Optional[int] = None, head_lines: Optional[int] = None, combine_stderr: bool = False) -> str:
    """Execute a shell command, transitioning to background execution if it takes too long."""
    logger.info(f"Running shell command: '{command}' (filter={output_filter}, tail={tail_lines}, head={head_lines}, combine_stderr={combine_stderr})")
    validation_err = self.validate_command_safety(command)
    if validation_err:
      logger.warning(f"Rejected command '{command}': {validation_err}")
      return validation_err
    import subprocess
    import tempfile
    
    task_id = f"task_{self.next_task_id}"
    stdout_f = None
    stderr_f = None
    try:
      stdout_f = tempfile.NamedTemporaryFile(delete=False, mode='w+t', prefix=f"chatty_{task_id}_stdout_")
      if not combine_stderr:
        stderr_f = tempfile.NamedTemporaryFile(delete=False, mode='w+t', prefix=f"chatty_{task_id}_stderr_")
      record_command_binaries(command, self)
      proc = subprocess.Popen(
        command,
        shell=True,
        cwd=self.sandbox,
        stdout=stdout_f,
        stderr=subprocess.STDOUT if combine_stderr else stderr_f,
        start_new_session=True
      )
      try:
        proc.wait(timeout=10)
        stdout_f.close()
        if stderr_f:
          stderr_f.close()
        with open(stdout_f.name, 'r', errors='replace') as f:
          stdout = f.read()
        if stderr_f:
          with open(stderr_f.name, 'r', errors='replace') as f:
            stderr = f.read()
        else:
          stderr = ""
        try:
          os.unlink(stdout_f.name)
        except Exception:
          pass
        if stderr_f:
          try:
            os.unlink(stderr_f.name)
          except Exception:
            pass
        
        if output_filter or tail_lines is not None or head_lines is not None:
          stdout = self._apply_output_filters(stdout, output_filter, head_lines, tail_lines)
          stderr = self._apply_output_filters(stderr, output_filter, head_lines, tail_lines)
          
        output = []
        if stdout:
          output.append(f"Stdout:\n{truncate_output(stdout, max_chars=self.max_command_chars)}")
        if stderr:
          output.append(f"Stderr:\n{truncate_output(stderr, max_chars=self.max_command_chars)}")
        status = f"Command exited with code {proc.returncode}."
        logger.info(f"Command completed in foreground. Exit code: {proc.returncode}")
        return "\n".join(output) + f"\n{status}" if output else status
      except subprocess.TimeoutExpired:
        task_id = f"task_{self.next_task_id}"
        self.next_task_id += 1
        logger.info(f"Command timed out. Transitioned to background. Task ID: {task_id}")
        console.print(
          f"\n[bold yellow]⚙️  Command took > 10s and is now running in the background. "
          f"Task ID: {task_id}. Use '/status' or check_background_command to monitor progress.[/bold yellow]\n"
        )
        self.background_commands[task_id] = {
          "proc": proc,
          "command": command,
          "stdout_path": stdout_f.name,
          "stderr_path": stderr_f.name if stderr_f else None,
          "stdout_file": stdout_f,
          "stderr_file": stderr_f,
          "output_filter": output_filter,
          "tail_lines": tail_lines,
          "head_lines": head_lines
        }
        self._prune_background_commands()
        return (
          f"Info: The command is taking longer than 10 seconds. It is now running in the background.\n"
          f"Task ID: {task_id}\n"
          "You must NOT block. Instead, check its output later by calling the 'check_background_command' tool."
        )
    except Exception as e:
      if stdout_f:
        stdout_f.close()
        try:
          os.unlink(stdout_f.name)
        except Exception:
          pass
      if stderr_f:
        stderr_f.close()
        try:
          os.unlink(stderr_f.name)
        except Exception:
          pass
      return f"Error executing command: {str(e)}"

  def tool_check_background_command(
    self,
    task_id: str,
    timeout: Optional[float] = None,
    output_filter: Optional[str] = None,
    tail_lines: Optional[int] = None,
    head_lines: Optional[int] = None,
  ) -> str:
    """Check status of a background task and read its currently accumulated stdout and stderr."""
    logger.info(f"Checking status of background task: '{task_id}' (timeout={timeout})")
    task = self.background_commands.get(task_id)
    if not task:
      logger.warning(f"Check background task failed: Task ID '{task_id}' not found")
      return f"Error: Task ID '{task_id}' not found."
    proc = task["proc"]

    status = task.get("status")
    timed_out_while_waiting = False
    if status is None:
      # If timeout is specified, wait for the process to complete or timeout to expire
      if timeout is not None and timeout > 0:
        import time
        start_time = time.time()
        while proc.poll() is None:
          elapsed = time.time() - start_time
          if elapsed >= timeout:
            timed_out_while_waiting = True
            break
          time.sleep(min(0.2, timeout - elapsed))
      status = proc.poll()
      if status is not None:
        task["status"] = status
    try:
      with open(task["stdout_path"], 'r', errors='replace') as f:
        stdout_content = f.read()
      if task.get("stderr_path"):
        with open(task["stderr_path"], 'r', errors='replace') as f:
          stderr_content = f.read()
      else:
        stderr_content = ""
    except Exception as e:
      stdout_content = f"Error reading output: {e}"
      stderr_content = ""
      
    actual_filter = output_filter if output_filter is not None else task.get("output_filter")
    actual_tail = tail_lines if tail_lines is not None else task.get("tail_lines")
    actual_head = head_lines if head_lines is not None else task.get("head_lines")
    
    if actual_filter or actual_tail is not None or actual_head is not None:
      stdout_content = self._apply_output_filters(stdout_content, actual_filter, actual_head, actual_tail)
      stderr_content = self._apply_output_filters(stderr_content, actual_filter, actual_head, actual_tail)

    output = []
    if stdout_content:
      output.append(f"Stdout:\n{truncate_output(stdout_content, max_chars=self.max_command_chars)}")
    if stderr_content:
      output.append(f"Stderr:\n{truncate_output(stderr_content, max_chars=self.max_command_chars)}")
    if status is None:
      logger.info(f"Task '{task_id}' is STILL RUNNING.")
      status_msg = f"Status: Task '{task_id}' is STILL RUNNING"
      if timeout is not None and timeout > 0 and timed_out_while_waiting:
        status_msg += f" (the check timed out after {timeout} seconds)"
      status_msg += ".\n"
      return status_msg + ("\n".join(output) if output else "(No output generated yet)")
    else:
      logger.info(f"Task '{task_id}' FINISHED with exit code {status}.")
      try:
        if task.get("stdout_file"):
          task["stdout_file"].close()
        if task.get("stderr_file"):
          task["stderr_file"].close()
      except Exception:
        pass
      self._prune_background_commands()
      return (
        f"Status: Task '{task_id}' FINISHED with exit code {status}.\n"
        + ("\n".join(output) if output else "(No output generated)")
      )

  def tool_kill_process(self, task_id: str) -> str:
    """Terminate a background task/process by its Task ID."""
    import signal
    logger.info(f"Terminating background task: '{task_id}'")
    task = self.background_commands.get(task_id)
    if not task:
      logger.warning(f"Kill process failed: Task ID '{task_id}' not found")
      return f"Error: Task ID '{task_id}' not found."
    
    proc = task["proc"]
    status = task.get("status")
    if status is None:
      status = proc.poll()
      if status is not None:
        task["status"] = status
    if status is None:
      try:
        os.killpg(proc.pid, signal.SIGKILL)
        logger.info(f"Process group {proc.pid} terminated.")
      except Exception as e:
        logger.error(f"Failed to kill process group {proc.pid}: {e}")
        return f"Error terminating process: {e}"
      message = f"Successfully terminated background task '{task_id}'."
    else:
      message = f"Background task '{task_id}' had already exited with code {status}. Cleaned up resources."
    
    try:
      if task.get("stdout_file"):
        task["stdout_file"].close()
      if task.get("stderr_file"):
        task["stderr_file"].close()
    except Exception:
      pass
    try:
      os.unlink(task["stdout_path"])
      if task.get("stderr_path"):
        os.unlink(task["stderr_path"])
    except Exception:
      pass
        
    del self.background_commands[task_id]
    return message

  @staticmethod
  def _cleanup_resources(background_commands):
    """Kills all active background tasks and removes temporary files."""
    import signal
    import os
    global logger
    if background_commands:
      if logger is not None:
        try:
          logger.info(f"Cleaning up {len(background_commands)} background commands...")
        except Exception:
          pass
    for task_id, task in list(background_commands.items()):
      proc = task.get("proc")
      status = task.get("status")
      if proc and status is None and proc.poll() is None:
        try:
          os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
          pass
      try:
        if task.get("stdout_file"):
          task["stdout_file"].close()
        if task.get("stderr_file"):
          task["stderr_file"].close()
      except Exception:
        pass
      try:
        if os is not None and task.get("stdout_path"):
          os.unlink(task["stdout_path"])
      except Exception:
        pass
      try:
        if os is not None and task.get("stderr_path"):
          os.unlink(task["stderr_path"])
      except Exception:
        pass
    background_commands.clear()
    if logger is not None:
      try:
        logger.info("Background commands cleanup finished.")
      except Exception:
        pass

  def cleanup_background_commands(self):
    """Kills all active background tasks and removes temporary files."""
    ChatbotSession._cleanup_resources(self.background_commands)

  def _prune_background_commands(self):
    """Ensures we only keep the latest max_completed_tasks completed background task outputs, unlinking older ones."""
    completed_tasks = []
    for task_id, task in self.background_commands.items():
      status = task.get("status")
      if status is None:
        proc = task.get("proc")
        if proc:
          status = proc.poll()
          if status is not None:
            task["status"] = status
      if status is not None:
        completed_tasks.append(task_id)
    def get_task_num(tid):
      try:
        return int(tid.split("_")[1])
      except (IndexError, ValueError):
        return 0
    completed_tasks.sort(key=get_task_num)
    if len(completed_tasks) > self.max_completed_tasks:
      tasks_to_prune = completed_tasks[:-self.max_completed_tasks]
      for task_id in tasks_to_prune:
        task = self.background_commands[task_id]
        logger.info(f"Pruning old completed background task: '{task_id}'")
        try:
          if task.get("stdout_file"):
            task["stdout_file"].close()
          if task.get("stderr_file"):
            task["stderr_file"].close()
        except Exception:
          pass
        try:
          if task.get("stdout_path"):
            os.unlink(task["stdout_path"])
          if task.get("stderr_path"):
            os.unlink(task["stderr_path"])
        except Exception:
          pass
        del self.background_commands[task_id]

  def prune_history(self, log: bool = True) -> List[Dict[str, Any]]:
    """Prunes conversation history to respect the configured context size, compressing older tool outputs."""
    sys_prompt = self.get_active_system_prompt()
    system_msg = {"role": "system", "content": sys_prompt}
    if self.prompt_caching:
      system_msg["cache_control"] = {"type": "ephemeral"}
    sys_tokens = count_tokens(sys_prompt)
    
    if sys_tokens >= self.context_size:
      return [system_msg]
      
    processed_messages = []
    total_msgs = len(self.messages)
    
    for idx, msg in enumerate(self.messages):
      cloned_msg = dict(msg)
      # Compress tool outputs that are not part of the active window
      if cloned_msg.get("role") == "tool" and idx < total_msgs - self.history_keep_messages:
        content = cloned_msg.get("content") or ""
        if len(content) > self.max_history_tool_chars:
          half = self.max_history_tool_chars // 2
          truncated_len = len(content) - self.max_history_tool_chars
          cloned_msg["content"] = (
            f"{content[:half]}\n\n"
            f"... [TRUNCATED {truncated_len} CHARACTERS OF HISTORICAL TOOL OUTPUT] ...\n\n"
            f"{content[-half:]}"
          )
      if cloned_msg.get("role") == "assistant" and self.provider != "openrouter":
        for field in ["reasoning", "reasoning_content", "reasoning_details", "thought_signature"]:
          cloned_msg.pop(field, None)
      processed_messages.append(cloned_msg)
      
    pruned = []
    accumulated_tokens = sys_tokens
    
    # Process from newest to oldest
    for msg in reversed(processed_messages):
      content = msg.get("content") or ""
      # Estimate tool call tokens
      if msg.get("tool_calls"):
        content += json.dumps(msg["tool_calls"])
      if msg.get("tool_call_id"):
        content += msg["tool_call_id"]
        
      msg_tokens = count_tokens(content) + 12  # add safety overhead per message structure
      
      if accumulated_tokens + msg_tokens > self.context_size:
        break
        
      pruned.insert(0, msg)
      accumulated_tokens += msg_tokens
      
    if log:
      logger.info(f"Pruning history: kept {len(pruned)} out of {total_msgs} messages (accumulated tokens: {accumulated_tokens})")
      pruned_count = total_msgs - len(pruned)
      if pruned_count > 0:
        logger.warning(f"Context window limit reached. Pruned {pruned_count} messages from history.")
        console.print(
          f"\n[bold yellow]⚠️  Context Warning: {pruned_count} older message(s) were pruned from history "
          f"to fit the context size limit ({self.context_size} tokens).[/bold yellow]"
        )
        # If the very first user prompt is no longer in the pruned message history
        if self.messages and self.messages[0] not in pruned:
          console.print(
            "[bold red]⚠️  Critical: Your initial prompt/instructions have been pruned from context! "
            "The AI may lose track of the overall goal. Consider running '/compress' to reload a summary recap.[/bold red]\n"
          )
    # Filter orphaned tool messages
    defined_ids = set()
    for msg in pruned:
      if msg.get("role") == "assistant" and msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
          defined_ids.add(tc.get("id"))
          
    final_pruned = []
    for msg in pruned:
      if msg.get("role") == "tool":
        t_id = msg.get("tool_call_id")
        if t_id not in defined_ids:
          continue
      final_pruned.append(msg)
      
    if self.prompt_caching and final_pruned:
      final_pruned = [dict(msg) for msg in final_pruned]
      final_pruned[-1]["cache_control"] = {"type": "ephemeral"}
      if len(final_pruned) >= 2:
        final_pruned[-2]["cache_control"] = {"type": "ephemeral"}
        
    return [system_msg] + final_pruned

  def extract_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
    """Attempts to parse JSON tool calls from plain text content when the LLM returns

    JSON tool calls as a plain text string instead of using structured
    tool_calls fields.
    """
    text = text.strip()

    def parse_single_dict_tool_call(data: Any) -> Optional[Dict[str, Any]]:
      if not isinstance(data, dict):
        return None
      if "name" in data and "arguments" in data:
        args = data["arguments"]
        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
        return {
          "id": "call_text_parsed",
          "type": "function",
          "function": {
            "name": data["name"],
            "arguments": args_str
          }
        }
      elif data.get("type") == "function" and "function" in data:
        func = data["function"]
        if isinstance(func, dict) and "name" in func and "arguments" in func:
          args = func["arguments"]
          args_str = json.dumps(args) if isinstance(args, dict) else str(args)
          return {
            "id": "call_text_parsed",
            "type": "function",
            "function": {
              "name": func["name"],
              "arguments": args_str
            }
          }
      return None

    # 1. Check if the entire text is a JSON object
    try:
      data = json.loads(text, strict=False)
      tool_call = parse_single_dict_tool_call(data)
      if tool_call:
        return [tool_call]
    except Exception:
      pass

    # 2. Try parsing code blocks
    import re
    code_blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    for block in code_blocks:
      try:
        data = json.loads(block.strip(), strict=False)
        tool_call = parse_single_dict_tool_call(data)
        if tool_call:
          return [tool_call]
      except Exception:
        pass

    # 3. Try parsing nested braces using brace tracking
    n = len(text)
    for i in range(n):
      if text[i] == '{':
        brace_count = 0
        in_string = False
        escaped = False
        for j in range(i, n):
          char = text[j]
          if in_string:
            if escaped:
              escaped = False
            elif char == '\\':
              escaped = True
            elif char == '"':
              in_string = False
          else:
            if char == '"':
              in_string = True
            elif char == '{':
              brace_count += 1
            elif char == '}':
              brace_count -= 1
              if brace_count == 0:
                potential = text[i:j+1]
                try:
                  data = json.loads(potential, strict=False)
                  tool_call = parse_single_dict_tool_call(data)
                  if tool_call:
                    return [tool_call]
                except Exception:
                  pass
                break

    return []


  def get_tools(self) -> Optional[List[Dict[str, Any]]]:
    """Returns list of tools, optionally annotated with cache_control for OpenRouter."""
    if not TOOLS_SCHEMA:
      return None

    from chatty.tools import get_available_formatters
    available = get_available_formatters()
    available_str = ", ".join(available) if available else "none detected"

    tools = []
    for t in TOOLS_SCHEMA:
      t_copy = dict(t)
      if t_copy["function"]["name"] == "format_file":
        t_copy["function"] = dict(t_copy["function"])
        t_copy["function"]["description"] = (
          "Format a source code file using the appropriate formatter. Shows a diff of changes. "
          f"Detected available formatters on this system: {available_str}. "
          "You can optionally specify a custom formatter and the path to a tool-specific rules/configuration file."
        )
      tools.append(t_copy)

    if self.prompt_caching:
      if tools:
        tools[-1] = dict(tools[-1])
        tools[-1]["cache_control"] = {"type": "ephemeral"}
      return tools
    return tools

  def _log_llm_request(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> None:
    """Logs detailed information about the LLM request in DEBUG mode."""
    if not logger.isEnabledFor(logging.DEBUG):
      return

    # Mask headers
    headers = getattr(self.client, "default_headers", {})
    masked_headers = {}
    for k, v in headers.items():
      if k.lower() in ("authorization", "api-key", "x-api-key"):
        if isinstance(v, str):
          if len(v) > 12:
            masked_headers[k] = v[:8] + "..." + v[-4:]
          else:
            masked_headers[k] = "..."
        else:
          masked_headers[k] = "..."
      else:
        masked_headers[k] = v

    # Mask API key
    api_key = getattr(self.client, "api_key", None)
    masked_key = "None"
    if api_key:
      if len(api_key) > 12:
        masked_key = api_key[:8] + "..." + api_key[-4:]
      else:
        masked_key = "..."

    logger.debug("=== LLM Request Details ===")
    logger.debug(f"Provider: {self.provider}")
    logger.debug(f"Model: {self.model}")
    logger.debug(f"Base URL: {getattr(self.client, 'base_url', 'Unknown')}")
    logger.debug(f"API Key: {masked_key}")
    logger.debug(f"Default Headers: {masked_headers}")
    logger.debug(f"Timeout: {getattr(self.client, 'timeout', 'Default')}")
    logger.debug(f"Max Retries: {getattr(self.client, 'max_retries', 'Default')}")
    logger.debug(f"Request Messages ({len(messages)}):")
    try:
      logger.debug(json.dumps(messages, indent=2, default=str))
    except Exception as e:
      logger.debug(f"Error serializing request messages: {e}")
      logger.debug(str(messages))

    if tools:
      logger.debug(f"Request Tools ({len(tools)}):")
      try:
        logger.debug(json.dumps(tools, indent=2, default=str))
      except Exception as e:
        logger.debug(f"Error serializing request tools: {e}")
        logger.debug(str(tools))
    else:
      logger.debug("Request Tools: None")
    logger.debug("===========================")

  def _log_llm_response_summary(
    self,
    content_accumulated: str,
    tool_calls_accumulated: List[Dict[str, Any]],
    extra_fields_accumulated: Dict[str, Any],
    finish_reason: Optional[str] = None,
    usage: Optional[Any] = None,
    response_model: Optional[str] = None,
    system_fingerprint: Optional[str] = None,
    chunk_id: Optional[str] = None
  ) -> None:
    """Logs detailed summary of the LLM response in DEBUG mode."""
    if not logger.isEnabledFor(logging.DEBUG):
      return

    logger.debug("=== LLM Response Details ===")
    logger.debug(f"Chunk ID: {chunk_id}")
    logger.debug(f"Response Model: {response_model}")
    logger.debug(f"System Fingerprint: {system_fingerprint}")
    logger.debug(f"Finish Reason: {finish_reason}")

    if usage:
      if hasattr(usage, "prompt_tokens"):
        logger.debug(f"Usage: prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}, total_tokens={usage.total_tokens}")
      elif isinstance(usage, dict):
        logger.debug(f"Usage: prompt_tokens={usage.get('prompt_tokens')}, completion_tokens={usage.get('completion_tokens')}, total_tokens={usage.get('total_tokens')}")
      else:
        logger.debug(f"Usage: {usage}")
    else:
      logger.debug("Usage: Not provided/available")

    for field, val in extra_fields_accumulated.items():
      if val is not None:
        logger.debug(f"Extra Field ({field}): {val}")

    if content_accumulated:
      logger.debug(f"Assistant response content ({len(content_accumulated)} chars):")
      logger.debug(content_accumulated)
    else:
      logger.debug("Assistant response content: None")

    if tool_calls_accumulated:
      logger.debug(f"Assistant response tool calls ({len(tool_calls_accumulated)}):")
      try:
        logger.debug(json.dumps(tool_calls_accumulated, indent=2, default=str))
      except Exception as e:
        logger.debug(f"Error serializing tool calls: {e}")
        logger.debug(str(tool_calls_accumulated))
    else:
      logger.debug("Assistant response tool calls: None")
    logger.debug("============================")

  def run_llm_cycle(self):
    """Executes a full inference cycle, resolving tool calls recursively."""
    self.load_skills()
    max_tool_loops = self.max_loops
    loop_count = 0
    logger.info(f"Starting LLM cycle. Max sequential tool loops: {max_tool_loops}")
    
    while loop_count < max_tool_loops:
      self.current_loop = loop_count + 1
      max_retries = 3
      api_succeeded = False
      finish_reason = None
      
      for attempt in range(1, max_retries + 1):
        # Prepare message payloads based on limit settings
        active_messages = self.prune_history()
        
        # If this is a retry and the last message in active_messages is a tool message,
        # append a temporary user message nudge to bypass trailing tool message chat template issues
        if attempt > 1 and active_messages and active_messages[-1].get("role") == "tool":
          active_messages.append({
            "role": "user",
            "content": "Please continue the task using the tool output above."
          })
        
        # Start LLM stream call
        tool_calls_accumulated = []
        content_accumulated = ""
        extra_fields_accumulated = {
          "reasoning": None,
          "reasoning_content": None,
          "reasoning_details": None,
          "thought_signature": None
        }
        
        logger.info(f"Loop {loop_count + 1}/{max_tool_loops} (Attempt {attempt}/{max_retries}): Sending request to LLM (model={self.model}) with {len(active_messages)} messages")
        try:
          # Live rendering console helper
          panel = Panel("Connecting to LLM...", title="Assistant", border_style="green")
          with Live(Group(panel, self.get_rich_status_bar()), 
                    refresh_per_second=12, console=console, transient=True) as live:
            
            # Log request details in DEBUG mode
            self._log_llm_request(active_messages, self.get_tools())
            
            # Try calling with stream_options={"include_usage": True}
            try:
              stream = self.client.chat.completions.create(
                model=self.model,
                messages=active_messages,
                tools=self.get_tools(),
                stream=True,
                stream_options={"include_usage": True}
              )
            except Exception as e:
              logger.debug(f"Failed to call API with stream_options: {e}. Retrying without stream_options.")
              stream = self.client.chat.completions.create(
                model=self.model,
                messages=active_messages,
                tools=self.get_tools(),
                stream=True
              )
            
            first_metadata_chunk = True
            first_chunk = True
            finish_reason = None
            usage_metadata = None
            chunk_id = None
            resp_model = None
            sys_fp = None
            for chunk in stream:
              # Capture usage if present
              if hasattr(chunk, "usage") and chunk.usage:
                usage_metadata = chunk.usage
              elif hasattr(chunk, "model_extra") and chunk.model_extra and "usage" in chunk.model_extra:
                usage_metadata = chunk.model_extra["usage"]

              # Log metadata on first chunk
              if first_metadata_chunk:
                chunk_id = getattr(chunk, "id", None)
                resp_model = getattr(chunk, "model", None)
                sys_fp = getattr(chunk, "system_fingerprint", None)
                logger.debug(
                  f"LLM response started. Chunk ID: {chunk_id}, Model: {resp_model}, System Fingerprint: {sys_fp}"
                )
                first_metadata_chunk = False

              if not chunk.choices:
                continue
              choice = chunk.choices[0]
              delta = choice.delta
              if hasattr(choice, "finish_reason") and choice.finish_reason:
                finish_reason = choice.finish_reason
              
              # Extract any OpenRouter extra fields for reasoning/thought
              extra_fields = ["reasoning", "reasoning_content", "reasoning_details", "thought_signature"]
              has_new_reasoning = False
              for field in extra_fields:
                val = getattr(delta, field, None)
                if val is None and hasattr(delta, "model_extra") and delta.model_extra:
                  val = delta.model_extra.get(field)
                if val is None and isinstance(delta, dict):
                  val = delta.get(field)
                  
                if val is not None:
                  has_new_reasoning = True
                  if extra_fields_accumulated[field] is None:
                    extra_fields_accumulated[field] = val
                  elif isinstance(val, str) and isinstance(extra_fields_accumulated[field], str):
                    extra_fields_accumulated[field] += val
                  else:
                    extra_fields_accumulated[field] = val
              
              # Process streaming content
              if delta.content:
                content_accumulated += delta.content

              # Process streaming reasoning/thought
              reasoning_accumulated = (extra_fields_accumulated.get("reasoning_content") or 
                                       extra_fields_accumulated.get("reasoning") or "")
              
              if delta.content or has_new_reasoning:
                first_chunk = False
                renderables = []
                if reasoning_accumulated.strip():
                  renderables.append(Panel(LazyMarkdown(reasoning_accumulated), title="Thinking", border_style="yellow"))
                if content_accumulated:
                  renderables.append(Panel(LazyMarkdown(content_accumulated), title="Assistant", border_style="green"))
                
                if renderables:
                  panel = Group(*renderables)
                  live.update(Group(panel, self.get_rich_status_bar()))
                  
              # Process streaming tool calls
              if delta.tool_calls:
                first_chunk = False
                for tc in delta.tool_calls:
                  idx = tc.index
                  while len(tool_calls_accumulated) <= idx:
                    tool_calls_accumulated.append({
                      "id": None,
                      "type": "function",
                      "function": {"name": "", "arguments": ""}
                    })
                  
                  item = tool_calls_accumulated[idx]
                  if tc.id:
                    item["id"] = tc.id
                  if tc.function:
                    if tc.function.name:
                      item["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                      item["function"]["arguments"] += tc.function.arguments
                        
                  # Render loading indicator
                  panel = Panel(f"Accumulating tool arguments... ({len(tool_calls_accumulated)} call(s))", 
                                 title="System", border_style="yellow")
                  live.update(Group(panel, self.get_rich_status_bar()))
            # Remove status bar before exiting Live context
            live.update(panel)
          
          # Reconstruct and print the final panels permanently to console
          final_panels = []
          if reasoning_accumulated.strip():
            final_panels.append(Panel(Markdown(reasoning_accumulated), title="Thinking", border_style="yellow"))
          if content_accumulated:
            final_panels.append(Panel(Markdown(content_accumulated), title="Assistant", border_style="green"))
          
          if final_panels:
            console.print(Group(*final_panels))
          
          if finish_reason == "length":
            logger.warning("LLM response was truncated due to output token limit (finish_reason='length').")
            console.print("\n[bold yellow]⚠️  Warning: The AI's response was truncated because it reached the maximum output token limit.[/bold yellow]\n")
          
          logger.info(f"LLM call succeeded. Content size: {len(content_accumulated)} chars, Tool calls count: {len(tool_calls_accumulated)}")
          self._log_llm_response_summary(
            content_accumulated=content_accumulated,
            tool_calls_accumulated=tool_calls_accumulated,
            extra_fields_accumulated=extra_fields_accumulated,
            finish_reason=finish_reason,
            usage=usage_metadata,
            response_model=resp_model,
            system_fingerprint=sys_fp,
            chunk_id=chunk_id
          )
        except Exception as e:
          logger.exception("Error calling LLM API")
          console.print(f"[bold red]Error calling API:[/bold red] {str(e)}")
          break
            
        # If we didn't receive structured tool calls, try to extract them from text content
        if not tool_calls_accumulated and content_accumulated:
          parsed_calls = self.extract_tool_calls_from_text(content_accumulated)
          if parsed_calls:
            tool_calls_accumulated = parsed_calls
            content_accumulated = ""
            
        # Check if response was empty (no content, no tool calls)
        is_empty_response = (not tool_calls_accumulated) and (not content_accumulated or not content_accumulated.strip())
        
        if not is_empty_response:
          api_succeeded = True
          break
          
        if attempt < max_retries:
          logger.info(f"LLM returned an empty response on loop {self.current_loop} (attempt {attempt}/{max_retries}). Retrying in 2s...")
          console.print(f"[bold yellow]⚠️  LLM returned an empty response. Retrying in 2s (attempt {attempt}/{max_retries})...[/bold yellow]")
          time.sleep(2)
        else:
          logger.info(f"LLM returned multiple empty responses on loop {self.current_loop}. Breaking cycle.")
          console.print("[bold red]❌  LLM returned multiple empty responses. Breaking cycle.[/bold red]")
          
      if not api_succeeded:
        break
              
      # Ensure every accumulated tool call has a unique ID
      for tc in tool_calls_accumulated:
        if not tc.get("id") or tc.get("id") == "call_text_parsed":
          tc["id"] = f"call_{uuid.uuid4().hex[:12]}"
              
      # Construct assistant message record
      assistant_msg = {"role": "assistant"}
      if content_accumulated:
        assistant_msg["content"] = content_accumulated
      else:
        assistant_msg["content"] = None
          
      if tool_calls_accumulated:
        assistant_msg["tool_calls"] = []
        for tc in tool_calls_accumulated:
          assistant_msg["tool_calls"].append({
            "id": tc["id"],
            "type": "function",
            "function": {
              "name": tc["function"]["name"],
              "arguments": tc["function"]["arguments"]
            }
          })
              
      for field, val in extra_fields_accumulated.items():
        if val is not None:
          assistant_msg[field] = val
            
      self.messages.append(assistant_msg)
      
      # If the response was truncated due to output token limit, automatically continue
      if finish_reason == "length":
        logger.warning("LLM response was truncated (finish_reason='length'). Automatically continuing...")
        console.print("[bold yellow]🔄  AI response was truncated because it reached the maximum output token limit. Automatically continuing...[/bold yellow]")
        loop_count += 1
        continue
      
      # If no tools called, we're finished with this turn
      if not tool_calls_accumulated:
        break
          
      # Otherwise, execute requested tools sequentially
      for tc in tool_calls_accumulated:
        t_id = tc["id"]
        t_name = tc["function"]["name"]
        t_args_raw = tc["function"]["arguments"]
        
        try:
          args_parsed = json.loads(t_args_raw) if t_args_raw else {}
        except Exception as e:
          args_parsed = {}
          t_result = f"Error: Arguments failed JSON parsing: {str(e)}"
        else:
          # Execute tool
          if t_name == "ask_question":
            logger.info(f"Executing tool {t_name} (id={t_id}) with arguments: {args_parsed}")
            t_result = execute_tool(t_name, args_parsed, self)
          else:
            exec_panel = Panel(
              f"Name: [cyan]{t_name}[/cyan]\nArguments: [yellow]{escape(json.dumps(args_parsed, indent=2))}[/yellow]",
              title="🔧 Executing Tool",
              border_style="yellow"
            )
            with Live(Group(exec_panel, self.get_rich_status_bar()), refresh_per_second=12, console=console) as live:
              logger.info(f"Executing tool {t_name} (id={t_id}) with arguments: {args_parsed}")
              t_result = execute_tool(t_name, args_parsed, self)
              # Remove status bar before exiting Live context
              live.update(exec_panel)
              
        # Print result summary nicely
        console.print(Panel(
          Text(t_result),
          title="🔧 Tool Result",
          border_style="dim yellow"
        ))
        
        truncated = "TRUNCATED" in t_result or "truncated" in t_result.lower() or "WARNING" in t_result
        logger.info(f"Tool {t_name} (id={t_id}) completed. Result size: {len(t_result)} characters (truncated: {truncated})")
        logger.debug(f"Tool {t_name} (id={t_id}) result content: {t_result}")
        
        # Record result for context
        # We wrap tool output in a JSON object to prevent issues with gateways/models
        # trying to parse raw code/braces as invalid JSON.
        t_result_sanitized = sanitize_tool_output(t_result)
        wrapped_content = json.dumps({"output": t_result_sanitized})

        self.messages.append({
          "role": "tool",
          "tool_call_id": t_id,
          "name": t_name,
          "content": wrapped_content
        })
          
      loop_count += 1
      
    if loop_count >= max_tool_loops:
      console.print("[bold red]Reached maximum sequential tool loop executions. Breaking cycle.[/bold red]")
    self.current_loop = 0

  def _register_commands(self):
    self._commands["/exit"] = self._cmd_exit
    self._commands["/quit"] = self._cmd_exit
    self._commands["/clear"] = self._cmd_clear
    self._commands["/reset"] = self._cmd_clear
    self._commands["/compress"] = self._cmd_compress
    self._commands["/help"] = self._cmd_help
    self._commands["/status"] = self._cmd_status
    self._commands["/tool_stats"] = self._cmd_tool_stats
    self._commands["/provider"] = self._cmd_provider
    self._commands["/model"] = self._cmd_model
    self._commands["/sandbox"] = self._cmd_sandbox
    self._commands["/context"] = self._cmd_context
    self._commands["/loops"] = self._cmd_loops
    self._commands["/api_key"] = self._cmd_api_key
    self._commands["/multiline"] = self._cmd_multiline
    self._commands["/system"] = self._cmd_system
    self._commands["/load"] = self._cmd_load
    self._commands["/save"] = self._cmd_save
    self._commands["/save_session"] = self._cmd_save
    self._commands["/load_session"] = self._cmd_load_session
    self._commands["/tools"] = self._cmd_tools
    self._commands["/history"] = self._cmd_history

  def _cmd_exit(self, arg: str) -> bool:
    self.cleanup_background_commands()
    console.print("[bold green]Goodbye![/bold green]")
    return False

  def _cmd_clear(self, arg: str) -> bool:
    self.messages.clear()
    console.print("[bold green]Conversation history cleared.[/bold green]")
    return True

  def _cmd_compress(self, arg: str) -> bool:
    self.compress_context()
    return True

  def _cmd_help(self, arg: str) -> bool:
    self.show_help()
    return True

  def _cmd_status(self, arg: str) -> bool:
    self.show_status()
    return True

  def _cmd_tool_stats(self, arg: str) -> bool:
    self.show_tool_stats()
    return True

  def _cmd_provider(self, arg: str) -> bool:
    if not arg:
      console.print(f"Current provider: [bold cyan]{self.provider}[/bold cyan]")
    elif arg in ("ollama", "openrouter"):
      self.provider = arg
      self.init_client()
      console.print(f"Switched provider to: [bold green]{self.provider}[/bold green]")
    else:
      console.print("[bold red]Error: Provider must be 'ollama' or 'openrouter'.[/bold red]")
    return True

  def _cmd_model(self, arg: str) -> bool:
    if not arg:
      console.print(f"Current model: [bold cyan]{self.model}[/bold cyan]")
    else:
      self.model = arg
      console.print(f"Model updated to: [bold green]{self.model}[/bold green]")
    return True

  def _cmd_sandbox(self, arg: str) -> bool:
    if not arg:
      console.print(f"Current sandbox path: [bold cyan]{self.sandbox}[/bold cyan]")
    else:
      abs_p = os.path.abspath(arg)
      os.makedirs(abs_p, exist_ok=True)
      self.sandbox = abs_p
      self.load_skills()
      console.print(f"Sandbox updated to: [bold green]{self.sandbox}[/bold green]")
    return True

  def _cmd_context(self, arg: str) -> bool:
    if not arg:
      console.print(f"Current context size: [bold cyan]{self.context_size}[/bold cyan] tokens")
    else:
      try:
        self.context_size = int(arg)
        console.print(f"Context size updated to: [bold green]{self.context_size}[/bold green] tokens")
      except ValueError:
        console.print("[bold red]Error: Context size must be an integer.[/bold red]")
    return True

  def _cmd_loops(self, arg: str) -> bool:
    if not arg:
      console.print(f"Current max loop limit: [bold cyan]{self.max_loops}[/bold cyan]")
    else:
      try:
        self.max_loops = int(arg)
        console.print(f"Max loop limit updated to: [bold green]{self.max_loops}[/bold green]")
      except ValueError:
        console.print("[bold red]Error: Max loops must be an integer.[/bold red]")
    return True

  def _cmd_api_key(self, arg: str) -> bool:
    if not arg:
      console.print("API Key: [dim](hidden)[/dim]")
    else:
      self.api_key = arg
      self.init_client()
      console.print("[bold green]API key updated successfully.[/bold green]")
    return True

  def _cmd_multiline(self, arg: str) -> bool:
    self.multiline_mode = not self.multiline_mode
    status = "enabled" if self.multiline_mode else "disabled"
    console.print(f"Multiline mode [bold cyan]{status}[/bold cyan].")
    if self.multiline_mode:
      console.print("[dim]Use Alt+Enter or Esc+Enter to submit message.[/dim]")
    return True

  def _cmd_system(self, arg: str) -> bool:
    if not arg:
      console.print(Panel(self.system_prompt, title="Current System Prompt", border_style="cyan"))
    else:
      self.system_prompt = arg
      console.print("[bold green]System prompt updated.[/bold green]")
    return True

  def _cmd_load(self, arg: str) -> bool:
    if not arg:
      console.print("[bold red]Error: Usage: /load <file_path> [append|replace][/bold red]")
    else:
      parts = arg.strip().rsplit(maxsplit=1)
      opt = "append"
      file_path = arg.strip()
      if len(parts) == 2 and parts[1].lower() in ("append", "replace"):
        file_path = parts[0].strip()
        opt = parts[1].lower()
      file_path = os.path.expanduser(file_path)
      try:
        loaded_prompt = load_system_prompt_from_file(file_path)
        if opt == "replace":
          self.system_prompt = loaded_prompt
          console.print(f"[bold green]System prompt replaced with content from {file_path}[/bold green]")
        else:
          self.system_prompt += f"\n\n{loaded_prompt}"
          console.print(f"[bold green]Appended prompt content from {file_path} to system prompt.[/bold green]")
      except Exception as e:
        console.print(f"[bold red]Error loading prompt file: {str(e)}[/bold red]")
    return True

  def _cmd_save(self, arg: str) -> bool:
    if not arg:
      console.print("[bold red]Error: Usage: /save_session <file_path>[/bold red]")
    else:
      file_path = os.path.expanduser(arg.strip())
      if not os.path.isabs(file_path):
        file_path = os.path.join(self.sandbox, file_path)
      dir_name = os.path.dirname(file_path)
      if dir_name:
        os.makedirs(dir_name, exist_ok=True)
      session_data = {
        "provider": self.provider,
        "model": self.model,
        "context_size": self.context_size,
        "sandbox": self.sandbox,
        "max_loops": self.max_loops,
        "system_prompt": self.system_prompt,
        "messages": self.messages,
        "tool_calls_count": self.tool_calls_count,
        "external_binaries_count": self.external_binaries_count,
        "external_binaries_breakdown": self.external_binaries_breakdown,
      }
      if self.api_key:
        session_data["api_key"] = self.api_key
      if self.url:
        session_data["url"] = self.url
      try:
        with open(file_path, "w", encoding="utf-8") as f:
          json.dump(session_data, f, indent=2, default=str)
        console.print(f"[bold green]Session saved successfully to {file_path}[/bold green]")
      except Exception as e:
        console.print(f"[bold red]Error saving session: {str(e)}[/bold red]")
    return True

  def _cmd_load_session(self, arg: str) -> bool:
    if not arg:
      console.print("[bold red]Error: Usage: /load_session <file_path>[/bold red]")
    else:
      file_path = os.path.expanduser(arg.strip())
      if not os.path.isabs(file_path):
        file_path = os.path.join(self.sandbox, file_path)
      try:
        with open(file_path, "r", encoding="utf-8") as f:
          session_data = json.load(f)
        if "provider" in session_data:
          self.provider = session_data["provider"]
        if "model" in session_data:
          self.model = session_data["model"]
        if "context_size" in session_data:
          self.context_size = session_data["context_size"]
        if "sandbox" in session_data:
          sandbox_path = os.path.abspath(session_data["sandbox"])
          if os.path.exists(sandbox_path):
            self.sandbox = sandbox_path
        if "max_loops" in session_data:
          self.max_loops = session_data["max_loops"]
        if "system_prompt" in session_data:
          self.system_prompt = session_data["system_prompt"]
        if "messages" in session_data:
          self.messages = session_data["messages"]
        if "tool_calls_count" in session_data:
          self.tool_calls_count = session_data["tool_calls_count"]
        if "external_binaries_count" in session_data:
          self.external_binaries_count = session_data["external_binaries_count"]
        if "external_binaries_breakdown" in session_data:
          self.external_binaries_breakdown = session_data["external_binaries_breakdown"]
        if "api_key" in session_data:
          self.api_key = session_data["api_key"]
        if "url" in session_data:
          self.url = session_data["url"]
        self.init_client()
        self.load_skills()
        console.print(f"[bold green]Session loaded successfully from {file_path}[/bold green]")
      except Exception as e:
        console.print(f"[bold red]Error loading session: {str(e)}[/bold red]")
    return True

  def _cmd_tools(self, arg: str) -> bool:
    self.show_tools()
    return True

  def _cmd_history(self, arg: str) -> bool:
    console.print("[bold cyan]Conversation History (estimated tokens):[/bold cyan]")
    for idx, msg in enumerate(self.messages):
      role = msg["role"]
      content = msg.get("content") or ""
      reasoning = msg.get("reasoning_content") or msg.get("reasoning")
      display_text = ""
      if reasoning:
        display_text += f"[Thinking: {reasoning[:60]}...]\n"
      display_text += content
      if "tool_calls" in msg:
        display_text += f"\n[Calls tools: {[tc['function']['name'] for tc in msg['tool_calls']]}]"
      tok = count_tokens(content)
      if reasoning:
        tok += count_tokens(reasoning)
      console.print(f" {idx + 1}. [bold]{role}[/bold]: {display_text[:80].replace('\n', ' ')}... ({tok} tokens)")
    return True

  def handle_command(self, cmd_line: str) -> bool:
    """
    Parses and handles slash commands.
    Returns True if program should continue, False to exit.
    """
    parts = cmd_line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    
    handler = self._commands.get(cmd)
    if handler:
      return handler(arg)
    
    console.print(f"[bold red]Unknown command:[/bold red] {cmd}. Type [cyan]/help[/cyan] for options.")
    return True

  def compress_context(self):
    """Summarizes the history, clears the context, and reloads the summary."""
    if not self.messages:
      console.print("[bold yellow]History is empty. Nothing to compress.[/bold yellow]")
      return

    # Prepare messages for summarization
    active_messages = self.prune_history()
    summary_instruction = (
      "Summarize our progress, the current task we are focusing on, "
      "any code modifications made so far, and the immediate next steps. "
      "Keep the summary concise but preserve all technical details, filenames, "
      "function names, paths, and key design decisions."
    )
    active_messages.append({"role": "user", "content": summary_instruction})

    # Log request details in DEBUG mode
    self._log_llm_request(active_messages, None)

    content_accumulated = ""
    logger.info("Generating summary for /compress command")
    try:
      with Live(Panel("Connecting to LLM for summary...", title="Context Compression", border_style="yellow"),
                refresh_per_second=12, console=console) as live:
        
        try:
          stream = self.client.chat.completions.create(
            model=self.model,
            messages=active_messages,
            stream=True,
            stream_options={"include_usage": True}
          )
        except Exception as e:
          logger.debug(f"Failed to call API with stream_options: {e}. Retrying without stream_options.")
          stream = self.client.chat.completions.create(
            model=self.model,
            messages=active_messages,
            stream=True
          )
        
        first_metadata_chunk = True
        first_content_chunk = True
        usage_metadata = None
        chunk_id = None
        resp_model = None
        sys_fp = None
        finish_reason = None
        for chunk in stream:
          # Capture usage if present
          if hasattr(chunk, "usage") and chunk.usage:
            usage_metadata = chunk.usage
          elif hasattr(chunk, "model_extra") and chunk.model_extra and "usage" in chunk.model_extra:
            usage_metadata = chunk.model_extra["usage"]

          # Log metadata on first chunk
          if first_metadata_chunk:
            chunk_id = getattr(chunk, "id", None)
            resp_model = getattr(chunk, "model", None)
            sys_fp = getattr(chunk, "system_fingerprint", None)
            logger.debug(
              f"LLM context compression response started. Chunk ID: {chunk_id}, Model: {resp_model}, System Fingerprint: {sys_fp}"
            )
            first_metadata_chunk = False

          if not chunk.choices:
            continue
          choice = chunk.choices[0]
          delta = choice.delta
          if hasattr(choice, "finish_reason") and choice.finish_reason:
            finish_reason = choice.finish_reason
          
          if delta.content:
            if first_content_chunk:
              live.update(Panel("", title="Context Summary", border_style="yellow"))
              first_content_chunk = False
            content_accumulated += delta.content
            live.update(Panel(Markdown(content_accumulated), title="Context Summary", border_style="yellow"))
            
      logger.info("Summary generation succeeded")
      self._log_llm_response_summary(
        content_accumulated=content_accumulated,
        tool_calls_accumulated=[],
        extra_fields_accumulated={},
        finish_reason=finish_reason,
        usage=usage_metadata,
        response_model=resp_model,
        system_fingerprint=sys_fp,
        chunk_id=chunk_id
      )
    except Exception as e:
      logger.exception("Error calling LLM API for context summary")
      console.print(f"[bold red]Error calling API for summary:[/bold red] {str(e)}")
      return

    if not content_accumulated.strip():
      console.print("[bold red]Failed to generate summary. Context was not cleared.[/bold red]")
      return

    # Clear and reload context
    self.messages.clear()
    self.messages.append({
      "role": "user",
      "content": "Summarize our progress and task context so far to optimize the context window."
    })
    self.messages.append({
      "role": "assistant",
      "content": content_accumulated
    })
    
    console.print("[bold green]Conversation history cleared and recap reloaded.[/bold green]")

  def show_help(self):
    """Displays formatted CLI usage guide."""
    table = Table(title="Slash Commands", show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    table.add_row("/help", "Show this help screen")
    table.add_row("/status", "Display current session configuration")
    table.add_row("/tool_stats", "Show statistics on tool and external binary calls")
    table.add_row("/provider [ollama|openrouter]", "View or switch the LLM backend provider")
    table.add_row("/model [name]", "View or switch the current LLM model")
    table.add_row("/sandbox [path]", "View or change the sandbox directory path")
    table.add_row("/context [tokens]", "View or modify the history token limit")
    table.add_row("/loops [iterations]", "View or modify the max sequential tool loops limit")
    table.add_row("/api_key [key]", "Configure the OpenRouter API Key")
    table.add_row("/system [text]", "View or edit the system instructions")
    table.add_row("/load <path> [append|replace]", "Load system prompt guidelines from a file")
    table.add_row("/save_session <path>", "Save the entire session status to a JSON file")
    table.add_row("/load_session <path>", "Load a saved session status from a JSON file")
    table.add_row("/multiline", "Toggle multiline prompt input (Alt+Enter to send)")
    table.add_row("/history", "View message records and sizing details")
    table.add_row("/tools", "List available sandbox tools and schemas")
    table.add_row("/clear / /reset", "Clear conversation memory")
    table.add_row("/compress", "Summarize history, clear context, and reload summary")
    table.add_row("/exit / /quit", "Exit the application")
    console.print(table)

  def show_status(self):
    """Displays configured status parameters."""
    table = Table(title="Active Session Status", show_header=False)
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="green")
    table.add_row("Provider", self.provider)
    table.add_row("Model", self.model)
    table.add_row("Sandbox Path", self.sandbox)
    table.add_row("Context Limit", f"{self.context_size} tokens")
    table.add_row("Max Loop Iterations", f"{self.max_loops} loops")
    table.add_row("Total Messages", str(len(self.messages)))
    table.add_row("Multiline Input", "Enabled" if self.multiline_mode else "Disabled")
    console.print(table)

  def show_tool_stats(self):
    """Displays statistics on tool and external binary calls."""
    # Tool calls table
    tool_table = Table(title="Tool Execution Stats", show_header=True, header_style="bold yellow")
    tool_table.add_column("Tool Name", style="cyan")
    tool_table.add_column("Call Count", style="green", justify="right")
    
    sorted_tools = sorted(self.tool_calls_count.items(), key=lambda x: (-x[1], x[0]))
    total_tool_calls = sum(self.tool_calls_count.values())
    
    for name, count in sorted_tools:
      tool_table.add_row(name, str(count))
    
    if not sorted_tools:
      tool_table.add_row("[dim]No tools called yet[/dim]", "0")
    else:
      tool_table.add_section()
      tool_table.add_row("[bold]Total Tool Calls[/bold]", f"[bold]{total_tool_calls}[/bold]")
        
    # External binary table
    bin_table = Table(title="External Binary Execution Stats", show_header=True, header_style="bold magenta")
    bin_table.add_column("Binary Name", style="cyan")
    bin_table.add_column("Call Count", style="green", justify="right")
    
    sorted_bins = sorted(self.external_binaries_breakdown.items(), key=lambda x: (-x[1], x[0]))
    
    for name, count in sorted_bins:
      bin_table.add_row(name, str(count))
        
    if not sorted_bins:
      bin_table.add_row("[dim]No external binaries executed yet[/dim]", "0")
    else:
      bin_table.add_section()
      bin_table.add_row("[bold]Total Binary Calls[/bold]", f"[bold]{self.external_binaries_count}[/bold]")
        
    console.print(Columns([tool_table, bin_table], equal=False, expand=True))

  def show_tools(self):
    """Lists available filesystem functions."""
    table = Table(title="Available Sandboxed Tools", show_header=True, header_style="bold yellow")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description", style="white")
    for tool in TOOLS_SCHEMA:
      func = tool["function"]
      table.add_row(func["name"], func["description"])
    console.print(table)

  def get_rich_status_bar(self):
    """Returns a Rich Table rendering the status bar."""
    total_tokens = 0
    active_messages = self.prune_history(log=False)
    if active_messages:
      sys_msg = active_messages[0]
      total_tokens += count_tokens(sys_msg.get("content") or "")
      for msg in active_messages[1:]:
        content = msg.get("content") or ""
        if msg.get("tool_calls"):
          content += json.dumps(msg["tool_calls"])
        if msg.get("tool_call_id"):
          content += msg["tool_call_id"]
        total_tokens += count_tokens(content) + 12

    table = Table(
      show_header=False,
      show_edge=False,
      show_lines=False,
      box=None,
      padding=0,
      expand=True,
      style="#e0e0e0 on #222222"
    )
    table.add_column()
    table.add_row(Text.from_markup(
      f" [bold]Chatty CLI[/bold] |"
      f" [bold]Provider:[/bold] [green]{self.provider}[/green] |"
      f" [bold]Model:[/bold] [yellow]{self.model}[/yellow] |"
      f" [bold]Tokens:[/bold] {total_tokens}/{self.context_size} |"
      f" [bold]Loops:[/bold] [cyan]{self.current_loop}/{self.max_loops}[/cyan] |"
      f" [bold]Sandbox:[/bold] {self.sandbox} "
    ))
    return table

  def start_loop(self):
    """Runs the interactive input/output CLI loop."""
    # Create keybindings for multiline submissions
    kb = KeyBindings()
    
    @kb.add('escape', 'enter')
    def _(event):
      event.current_buffer.validate_and_handle()
        
    # File history tracking
    history_file = os.path.expanduser("~/.agent_chat_history")
    toolbar_style = Style.from_dict({
      'bottom-toolbar': 'bg:#222222 fg:#e0e0e0 noreverse',
    })
    session = PromptSession(
      history=FileHistory(history_file),
      key_bindings=kb,
      style=toolbar_style
    )
    
    # Display starting banner
    console.print(Panel(
      "[bold green]Welcome to the Sandboxed AI Chatbot CLI![/bold green]\n"
      "This script interfaces with Ollama and OpenRouter and restricts file write operations to the sandbox.\n"
      "Type [cyan]/help[/cyan] to display slash commands.\n"
      "Press [cyan]Ctrl+D[/cyan] or type [cyan]/exit[/cyan] to exit.",
      title="Chatty Sandboxed Chatbot",
      border_style="magenta"
    ))
    
    def get_bottom_toolbar():
      total_tokens = 0
      active_messages = self.prune_history(log=False)
      if active_messages:
        sys_msg = active_messages[0]
        total_tokens += count_tokens(sys_msg.get("content") or "")
        for msg in active_messages[1:]:
          content = msg.get("content") or ""
          if msg.get("tool_calls"):
            content += json.dumps(msg["tool_calls"])
          if msg.get("tool_call_id"):
            content += msg["tool_call_id"]
          total_tokens += count_tokens(content) + 12
      
      return HTML(
        f" <b>Chatty CLI</b> |"
        f" <b>Provider:</b> <ansigreen>{self.provider}</ansigreen> |"
        f" <b>Model:</b> <ansiyellow>{self.model}</ansiyellow> |"
        f" <b>Tokens:</b> {total_tokens}/{self.context_size} |"
        f" <b>Loops:</b> <ansicyan>{self.current_loop}/{self.max_loops}</ansicyan> |"
        f" <b>Sandbox:</b> {self.sandbox} "
      )

    self.show_status()
    
    while True:
      # Format interactive prompt dynamically
      multiline_indicator = " [ML]" if self.multiline_mode else ""
      prompt_html = (
        f"<ansicyan><b>AI-Sandbox</b></ansicyan> "
        f"(<ansigreen>{self.provider}</ansigreen>:<ansiyellow>{self.model}</ansiyellow>)"
        f"{multiline_indicator} &gt; "
      )
      
      try:
        # Read user input
        user_input = session.prompt(
          HTML(prompt_html),
          multiline=self.multiline_mode,
          bottom_toolbar=get_bottom_toolbar
        )
        
        # Check for empty input
        if not user_input.strip():
          continue
            
        # Check for slash commands
        if user_input.strip().startswith("/"):
          logger.info(f"Slash Command: {user_input.strip()}")
          should_continue = self.handle_command(user_input)
          if not should_continue:
            break
          continue
            
        # Append user query to history and execute loop
        logger.info(f"User Input: {user_input}")
        self.messages.append({"role": "user", "content": user_input})
        self.run_llm_cycle()
        
      except KeyboardInterrupt:
        # Handle Ctrl+C (clear current input or confirm exit)
        console.print("\n[yellow]KeyboardInterrupt (Ctrl+C). Type /exit to quit.[/yellow]")
      except EOFError:
        # Handle Ctrl+D
        self.cleanup_background_commands()
        console.print("\n[bold green]Goodbye![/bold green]")
        break
      except Exception as e:
        logger.exception("Unexpected error in CLI loop")
        console.print(f"[bold red]Unexpected error in CLI loop:[/bold red] {str(e)}")
