import json
import logging
import os
import re
import sys
import uuid
from typing import List, Dict, Any, Tuple, Optional

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
  record_command_binaries
)
from chatty.tools import execute_tool, TOOLS_SCHEMA

logger = logging.getLogger("chatty")
console = Console()


class ChatbotSession:
  _active_session = None

  def __init__(self, provider: str, model: str, context_size: int, sandbox: str, api_key: str = None, url: str = None, max_loops: int = 20, system_prompt_override: str = None, prompt_mode: str = "replace", skills_paths: List[str] = None, max_read_chars: int = 40000, max_grep_results: int = 100, max_command_chars: int = 16000, max_history_tool_chars: int = 1000, history_keep_messages: int = 4, max_url_chars: int = 24000, max_dir_items: int = 200, static_skills: bool = None, prompt_caching: bool = False):
    ChatbotSession._active_session = self
    self.tool_calls_count: Dict[str, int] = {}
    self.external_binaries_count = 0
    self.external_binaries_breakdown: Dict[str, int] = {}
    
    self.provider = provider
    self.prompt_caching = prompt_caching
    if static_skills is None:
      self.static_skills = (provider == "openrouter")
    else:
      self.static_skills = static_skills
    self.model = model
    self.context_size = context_size
    self.sandbox = os.path.abspath(sandbox)
    self.api_key = api_key
    self.url = url
    self.max_loops = max_loops
    self.background_commands = {}
    self.next_task_id = 1
    self.skills_paths = skills_paths or []
    
    # Cutoff limits configurations
    self.max_read_chars = max_read_chars
    self.max_grep_results = max_grep_results
    self.max_command_chars = max_command_chars
    self.max_history_tool_chars = max_history_tool_chars
    self.history_keep_messages = history_keep_messages
    self.max_url_chars = max_url_chars
    self.max_dir_items = max_dir_items
    
    # Internal state
    self.messages: List[Dict[str, Any]] = []
    self.current_loop = 0
    default_prompt = (
      "You are a helpful assistant with local sandboxed file access and shell execution capabilities.\n"
      "You have tools for: listing directories (list_dir), locating files (locate_files), checking file info (get_file_info), reading files (read_file), writing files (write_file), patching files (patch_file), editing line ranges (edit_lines), searching regex patterns (search_grep), fetching web content (fetch_url), executing shell commands (run_command), checking background tasks (check_background_command), terminating background processes (kill_process), and sleeping/waiting (sleep).\n"
      "All paths provided to the tools will resolve relative to the sandbox directory.\n"
      "You are strictly prohibited from writing files outside the sandbox folder.\n"
      "CRITICAL: You MUST use the dedicated, high-level filesystem tools (like read_file, search_grep, locate_files, get_file_info) instead of running command-line utilities (like grep, find, cat, head, tail, sed, awk, less, more) inside run_command. Shell execution using run_command is blocked for these actions and will return an error. You must use get_file_info instead of running 'wc' or 'wc -l' inside run_command.\n"
      "CRITICAL: You are strictly prohibited from using the shell 'sleep' command inside run_command to pause execution. You MUST use the dedicated 'sleep' tool instead.\n"
      "When running shell commands using run_command, if a command takes longer than 10 seconds, it will automatically transition to run in the background and return a 'Task ID'. You must NOT block. Instead, check its output later by calling check_background_command with the Task ID to get progress or final output, or terminate it by calling kill_process with the Task ID. Perform other file tasks (read, patch, edit) or use the 'sleep' tool while waiting.\n"
      "To filter the output of run_command, use its optional 'output_filter' (regex), 'tail_lines', or 'head_lines' parameters rather than piping to grep or writing custom filtering scripts.\n"
      "When compilation, testing, verification, or running tools (like verilator, python scripts, compilers) is needed, you MUST execute them directly using the run_command tool instead of instructing the user to run them manually.\n"
      "Always use your tools proactively to solve tasks directly."
    )
    
    if system_prompt_override:
      if prompt_mode == "integrate":
        self.system_prompt = default_prompt + "\n\n" + system_prompt_override
      else:
        self.system_prompt = system_prompt_override
    else:
      self.system_prompt = default_prompt
    self.multiline_mode = False
    self.client = None
    
    # Ensure sandbox exists
    os.makedirs(self.sandbox, exist_ok=True)
    os.chdir(self.sandbox)
    
    # Initialize client
    self.init_client()
    
    # Load active skills
    self.skills = {}
    self.load_skills()
    logger.info(f"ChatbotSession initialized. Provider: {self.provider}, Model: {self.model}, Sandbox: {self.sandbox}")

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
      
      for token in tokens:
        if is_cmd_token:
          clean_token = os.path.basename(token.strip().lower())
          
          if clean_token in {'grep', 'egrep', 'fgrep', 'rgrep'}:
            return (
              f"Error: Using '{token}' directly in run_command is prohibited to search files. "
              "Please use the dedicated 'search_grep' tool instead."
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
          elif clean_token in {'head', 'tail', 'sed', 'awk'}:
            return (
              f"Error: Using '{token}' in run_command is prohibited to inspect files. "
              "Please use the dedicated 'read_file' tool with start_line and end_line parameters."
            )
          elif clean_token == 'sleep':
            return (
              "Error: Using 'sleep' in run_command is prohibited to pause execution. "
              "Please use the dedicated 'sleep' tool instead."
            )
        
        if token.strip() in sequencers:
          is_cmd_token = True
        elif token.strip() in {'|', '|&'}:
          is_cmd_token = False
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

  def tool_run_command(self, command: str, output_filter: Optional[str] = None, tail_lines: Optional[int] = None, head_lines: Optional[int] = None) -> str:
    """Execute a shell command, transitioning to background execution if it takes too long."""
    logger.info(f"Running shell command: '{command}' (filter={output_filter}, tail={tail_lines}, head={head_lines})")
    validation_err = self.validate_command_safety(command)
    if validation_err:
      logger.warning(f"Rejected command '{command}': {validation_err}")
      return validation_err
    import subprocess
    import tempfile
    
    def apply_filters(text: str) -> str:
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

    stdout_f = None
    stderr_f = None
    try:
      stdout_f = tempfile.NamedTemporaryFile(delete=False, mode='w+t', dir=self.sandbox)
      stderr_f = tempfile.NamedTemporaryFile(delete=False, mode='w+t', dir=self.sandbox)
      record_command_binaries(command, self)
      proc = subprocess.Popen(
        command,
        shell=True,
        cwd=self.sandbox,
        stdout=stdout_f,
        stderr=stderr_f,
        start_new_session=True
      )
      try:
        proc.wait(timeout=10)
        stdout_f.close()
        stderr_f.close()
        with open(stdout_f.name, 'r', errors='replace') as f:
          stdout = f.read()
        with open(stderr_f.name, 'r', errors='replace') as f:
          stderr = f.read()
        try:
          os.unlink(stdout_f.name)
        except Exception:
          pass
        try:
          os.unlink(stderr_f.name)
        except Exception:
          pass
        
        if output_filter or tail_lines is not None or head_lines is not None:
          stdout = apply_filters(stdout)
          stderr = apply_filters(stderr)
          
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
          "stderr_path": stderr_f.name,
          "stdout_file": stdout_f,
          "stderr_file": stderr_f,
          "output_filter": output_filter,
          "tail_lines": tail_lines,
          "head_lines": head_lines
        }
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

  def tool_check_background_command(self, task_id: str) -> str:
    """Check status of a background task and read its currently accumulated stdout and stderr."""
    logger.info(f"Checking status of background task: '{task_id}'")
    task = self.background_commands.get(task_id)
    if not task:
      logger.warning(f"Check background task failed: Task ID '{task_id}' not found")
      return f"Error: Task ID '{task_id}' not found."
    proc = task["proc"]
    status = proc.poll()
    try:
      with open(task["stdout_path"], 'r', errors='replace') as f:
        stdout_content = f.read()
      with open(task["stderr_path"], 'r', errors='replace') as f:
        stderr_content = f.read()
    except Exception as e:
      stdout_content = f"Error reading output: {e}"
      stderr_content = ""
      
    output_filter = task.get("output_filter")
    tail_lines = task.get("tail_lines")
    head_lines = task.get("head_lines")
    
    def apply_filters(text: str) -> str:
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

    if output_filter or tail_lines is not None or head_lines is not None:
      stdout_content = apply_filters(stdout_content)
      stderr_content = apply_filters(stderr_content)

    output = []
    if stdout_content:
      output.append(f"Stdout:\n{truncate_output(stdout_content, max_chars=self.max_command_chars)}")
    if stderr_content:
      output.append(f"Stderr:\n{truncate_output(stderr_content, max_chars=self.max_command_chars)}")
    if status is None:
      logger.info(f"Task '{task_id}' is STILL RUNNING.")
      return (
        f"Status: Task '{task_id}' is STILL RUNNING.\n"
        + ("\n".join(output) if output else "(No output generated yet)")
      )
    else:
      logger.info(f"Task '{task_id}' FINISHED with exit code {status}.")
      try:
        task["stdout_file"].close()
        task["stderr_file"].close()
        os.unlink(task["stdout_path"])
        os.unlink(task["stderr_path"])
      except Exception:
        pass
      del self.background_commands[task_id]
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
    status = proc.poll()
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
      task["stdout_file"].close()
      task["stderr_file"].close()
      os.unlink(task["stdout_path"])
      os.unlink(task["stderr_path"])
    except Exception:
      pass
        
    del self.background_commands[task_id]
    return message

  def cleanup_background_commands(self):
    """Kills all active background tasks and removes temporary files."""
    import signal
    if self.background_commands:
      logger.info(f"Cleaning up {len(self.background_commands)} background commands...")
    for task_id, task in list(self.background_commands.items()):
      proc = task["proc"]
      if proc.poll() is None:
        try:
          os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
          pass
      try:
        task["stdout_file"].close()
        task["stderr_file"].close()
        os.unlink(task["stdout_path"])
        os.unlink(task["stderr_path"])
      except Exception:
        pass
    self.background_commands.clear()
    logger.info("Background commands cleanup finished.")

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
    """
    Attempts to parse JSON tool calls from plain text content when the LLM returns
    JSON tool calls as a plain text string instead of using structured tool_calls fields.
    """
    text = text.strip()
    
    # Check if the entire text is a JSON object
    try:
      data = json.loads(text)
      if isinstance(data, dict):
        if "name" in data and "arguments" in data:
          args = data["arguments"]
          args_str = json.dumps(args) if isinstance(args, dict) else str(args)
          return [{
            "id": "call_text_parsed",
            "type": "function",
            "function": {
              "name": data["name"],
              "arguments": args_str
            }
          }]
        elif data.get("type") == "function" and "function" in data:
          func = data["function"]
          if "name" in func and "arguments" in func:
            args = func["arguments"]
            args_str = json.dumps(args) if isinstance(args, dict) else str(args)
            return [{
              "id": "call_text_parsed",
              "type": "function",
              "function": {
                "name": func["name"],
                "arguments": args_str
              }
            }]
    except Exception:
      pass
        
    # Search for a JSON object block using first '{' and last '}'
    first_idx = text.find('{')
    last_idx = text.rfind('}')
    if first_idx != -1 and last_idx != -1 and last_idx > first_idx:
      potential_json = text[first_idx:last_idx+1]
      try:
        data = json.loads(potential_json)
        if isinstance(data, dict) and "name" in data and "arguments" in data:
          args = data["arguments"]
          args_str = json.dumps(args) if isinstance(args, dict) else str(args)
          return [{
            "id": "call_text_parsed",
            "type": "function",
            "function": {
              "name": data["name"],
              "arguments": args_str
            }
          }]
      except Exception:
        pass
            
    return []

  def get_tools(self) -> Optional[List[Dict[str, Any]]]:
    """Returns list of tools, optionally annotated with cache_control for OpenRouter."""
    if not TOOLS_SCHEMA:
      return None
    if self.prompt_caching:
      tools = [dict(t) for t in TOOLS_SCHEMA]
      if tools:
        tools[-1] = dict(tools[-1])
        tools[-1]["cache_control"] = {"type": "ephemeral"}
      return tools
    return TOOLS_SCHEMA

  def run_llm_cycle(self):
    """Executes a full inference cycle, resolving tool calls recursively."""
    self.load_skills()
    max_tool_loops = self.max_loops
    loop_count = 0
    logger.info(f"Starting LLM cycle. Max sequential tool loops: {max_tool_loops}")
    
    while loop_count < max_tool_loops:
      self.current_loop = loop_count + 1
      # Prepare message payloads based on limit settings
      active_messages = self.prune_history()
      logger.debug(f"LLM request payload messages: {json.dumps(active_messages, default=str)}")
      
      # Start LLM stream call
      tool_calls_accumulated = []
      content_accumulated = ""
      extra_fields_accumulated = {
        "reasoning": None,
        "reasoning_content": None,
        "reasoning_details": None,
        "thought_signature": None
      }
      
      logger.info(f"Loop {loop_count + 1}/{max_tool_loops}: Sending request to LLM (model={self.model}) with {len(active_messages)} messages")
      try:
        # Live rendering console helper
        panel = Panel("Connecting to LLM...", title="Assistant", border_style="green")
        with Live(Group(panel, self.get_rich_status_bar()), 
                  refresh_per_second=12, console=console) as live:
          
          stream = self.client.chat.completions.create(
            model=self.model,
            messages=active_messages,
            tools=self.get_tools(),
            stream=True
          )
          
          first_chunk = True
          finish_reason = None
          for chunk in stream:
            if not chunk.choices:
              continue
            choice = chunk.choices[0]
            delta = choice.delta
            if hasattr(choice, "finish_reason") and choice.finish_reason:
              finish_reason = choice.finish_reason
            
            # Extract any OpenRouter extra fields for reasoning/thought
            extra_fields = ["reasoning", "reasoning_content", "reasoning_details", "thought_signature"]
            for field in extra_fields:
              val = getattr(delta, field, None)
              if val is None and hasattr(delta, "model_extra") and delta.model_extra:
                val = delta.model_extra.get(field)
              if val is None and isinstance(delta, dict):
                val = delta.get(field)
                
              if val is not None:
                if extra_fields_accumulated[field] is None:
                  extra_fields_accumulated[field] = val
                elif isinstance(val, str) and isinstance(extra_fields_accumulated[field], str):
                  extra_fields_accumulated[field] += val
                else:
                  extra_fields_accumulated[field] = val
            
            # Process streaming content
            if delta.content:
              if first_chunk:
                panel = Panel("", title="Assistant", border_style="green")
                first_chunk = False
              content_accumulated += delta.content
              panel = Panel(Markdown(content_accumulated), title="Assistant", border_style="green")
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
        
        if finish_reason == "length":
          logger.warning("LLM response was truncated due to output token limit (finish_reason='length').")
          console.print("\n[bold yellow]⚠️  Warning: The AI's response was truncated because it reached the maximum output token limit.[/bold yellow]\n")
        
        logger.info(f"LLM call succeeded. Content size: {len(content_accumulated)} chars, Tool calls count: {len(tool_calls_accumulated)}")
        if content_accumulated:
          logger.debug(f"Assistant response content: {content_accumulated}")
        if tool_calls_accumulated:
          logger.debug(f"Assistant response tool calls: {tool_calls_accumulated}")
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
              
      if self.provider == "openrouter":
        for field, val in extra_fields_accumulated.items():
          if val is not None:
            assistant_msg[field] = val
            
      self.messages.append(assistant_msg)
      
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
        self.messages.append({
          "role": "tool",
          "tool_call_id": t_id,
          "name": t_name,
          "content": t_result
        })
          
      loop_count += 1
      
    if loop_count >= max_tool_loops:
      console.print("[bold red]Reached maximum sequential tool loop executions. Breaking cycle.[/bold red]")
    self.current_loop = 0

  def handle_command(self, cmd_line: str) -> bool:
    """
    Parses and handles slash commands.
    Returns True if program should continue, False to exit.
    """
    parts = cmd_line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    
    if cmd in ("/exit", "/quit"):
      self.cleanup_background_commands()
      console.print("[bold green]Goodbye![/bold green]")
      return False
      
    elif cmd in ("/clear", "/reset"):
      self.messages.clear()
      console.print("[bold green]Conversation history cleared.[/bold green]")
      
    elif cmd == "/compress":
      self.compress_context()
      
    elif cmd == "/help":
      self.show_help()
      
    elif cmd == "/status":
      self.show_status()
      
    elif cmd == "/tool_stats":
      self.show_tool_stats()
      
    elif cmd == "/provider":
      if not arg:
        console.print(f"Current provider: [bold cyan]{self.provider}[/bold cyan]")
      elif arg in ("ollama", "openrouter"):
        self.provider = arg
        self.init_client()
        console.print(f"Switched provider to: [bold green]{self.provider}[/bold green]")
      else:
        console.print("[bold red]Error: Provider must be 'ollama' or 'openrouter'.[/bold red]")
          
    elif cmd == "/model":
      if not arg:
        console.print(f"Current model: [bold cyan]{self.model}[/bold cyan]")
      else:
        self.model = arg
        console.print(f"Model updated to: [bold green]{self.model}[/bold green]")
          
    elif cmd == "/sandbox":
      if not arg:
        console.print(f"Current sandbox path: [bold cyan]{self.sandbox}[/bold cyan]")
      else:
        abs_p = os.path.abspath(arg)
        os.makedirs(abs_p, exist_ok=True)
        self.sandbox = abs_p
        self.load_skills()
        console.print(f"Sandbox updated to: [bold green]{self.sandbox}[/bold green]")
          
    elif cmd == "/context":
      if not arg:
        console.print(f"Current context size: [bold cyan]{self.context_size}[/bold cyan] tokens")
      else:
        try:
          self.context_size = int(arg)
          console.print(f"Context size updated to: [bold green]{self.context_size}[/bold green] tokens")
        except ValueError:
          console.print("[bold red]Error: Context size must be an integer.[/bold red]")
              
    elif cmd == "/loops":
      if not arg:
        console.print(f"Current max loop limit: [bold cyan]{self.max_loops}[/bold cyan]")
      else:
        try:
          self.max_loops = int(arg)
          console.print(f"Max loop limit updated to: [bold green]{self.max_loops}[/bold green]")
        except ValueError:
          console.print("[bold red]Error: Max loops must be an integer.[/bold red]")
              
    elif cmd == "/api_key":
      if not arg:
        console.print("API Key: [dim](hidden)[/dim]")
      else:
        self.api_key = arg
        self.init_client()
        console.print("[bold green]API key updated successfully.[/bold green]")
          
    elif cmd == "/multiline":
      self.multiline_mode = not self.multiline_mode
      status = "enabled" if self.multiline_mode else "disabled"
      console.print(f"Multiline mode [bold cyan]{status}[/bold cyan].")
      if self.multiline_mode:
        console.print("[dim]Use Alt+Enter or Esc+Enter to submit message.[/dim]")
          
    elif cmd == "/system":
      if not arg:
        console.print(Panel(self.system_prompt, title="Current System Prompt", border_style="cyan"))
      else:
        self.system_prompt = arg
        console.print("[bold green]System prompt updated.[/bold green]")
          
    elif cmd == "/load":
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
              
    elif cmd == "/tools":
      self.show_tools()
      
    elif cmd == "/history":
      console.print("[bold cyan]Conversation History (estimated tokens):[/bold cyan]")
      for idx, msg in enumerate(self.messages):
        role = msg["role"]
        content = msg.get("content") or ""
        if "tool_calls" in msg:
          content += f"\n[Calls tools: {[tc['function']['name'] for tc in msg['tool_calls']]}]"
        tok = count_tokens(content)
        console.print(f" {idx + 1}. [bold]{role}[/bold]: {content[:80]}... ({tok} tokens)")
          
    else:
      console.print(f"[bold red]Unknown command:[/bold red] {cmd}. Type [cyan]/help[/cyan] for options.")
      
    return True

  def compress_context(self):
    """Summarizes the history, clears the context, and reloads the summary."""
    if not self.messages:
      console.print("[bold yellow]History is empty. Nothing to compress.[/bold yellow]")
      return

    # Prepare messages for summarization
    active_messages = self.prune_history()
    logger.debug(f"Context compression request payload messages: {json.dumps(active_messages, default=str)}")
    summary_instruction = (
      "Summarize our progress, the current task we are focusing on, "
      "any code modifications made so far, and the immediate next steps. "
      "Keep the summary concise but preserve all technical details, filenames, "
      "function names, paths, and key design decisions."
    )
    active_messages.append({"role": "user", "content": summary_instruction})

    content_accumulated = ""
    logger.info("Generating summary for /compress command")
    try:
      with Live(Panel("Connecting to LLM for summary...", title="Context Compression", border_style="yellow"),
                refresh_per_second=12, console=console) as live:
        
        stream = self.client.chat.completions.create(
          model=self.model,
          messages=active_messages,
          stream=True
        )
        
        first_chunk = True
        for chunk in stream:
          if not chunk.choices:
            continue
          delta = chunk.choices[0].delta
          
          if delta.content:
            if first_chunk:
              live.update(Panel("", title="Context Summary", border_style="yellow"))
              first_chunk = False
            content_accumulated += delta.content
            live.update(Panel(Markdown(content_accumulated), title="Context Summary", border_style="yellow"))
      logger.info("Summary generation succeeded")
      if content_accumulated:
        logger.debug(f"Generated summary: {content_accumulated}")
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
      title="Antigravity Sandboxed Chatbot",
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
