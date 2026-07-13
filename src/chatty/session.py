import contextlib
import json
import logging
import os
import re
import sys
import time
import uuid
import weakref
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field

from chatty.runner import SubprocessRunner, cleanup_resources
from chatty.llm import (
  _invalidate_token_cache,
  _calculate_tokens_for_messages,
  init_client,
  _throttle_request,
  _format_api_error,
  _is_retryable_exception,
  _resolve_model_and_provider,
  get_oracle_model,
  consult_oracle,
  prune_history,
  _log_llm_request,
  _log_llm_response_summary,
  run_llm_cycle
)

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
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document

from chatty.utils import (
  count_tokens,
  truncate_output,
  get_ollama_models,
  load_system_prompt_from_file,
  parse_frontmatter,
  record_command_binaries,
  sanitize_tool_output,
  repair_json
)
from chatty.tools import execute_tool, TOOLS_SCHEMA
from chatty.safety import validate_command_safety, active_session_var
from chatty.commands import COMMANDS

logger = logging.getLogger("chatty")
console = Console()


class CachedList(list):
  """A list subclass that triggers a callback when mutated in place."""

  def __init__(self, *args, on_change=None, **kwargs):
    if len(args) == 1 and isinstance(args[0], list) and not kwargs:
      super().__init__(args[0])
    else:
      super().__init__(*args, **kwargs)
    self.on_change = on_change

  def _trigger(self):
    if self.on_change:
      self.on_change()

  def append(self, item):
    super().append(item)
    self._trigger()

  def extend(self, iterable):
    super().extend(iterable)
    self._trigger()

  def insert(self, index, item):
    super().insert(index, item)
    self._trigger()

  def remove(self, item):
    super().remove(item)
    self._trigger()

  def pop(self, index=-1):
    val = super().pop(index)
    self._trigger()
    return val

  def clear(self):
    super().clear()
    self._trigger()

  def sort(self, *args, **kwargs):
    super().sort(*args, **kwargs)
    self._trigger()

  def reverse(self):
    super().reverse()
    self._trigger()

  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    self._trigger()

  def __delitem__(self, key):
    super().__delitem__(key)
    self._trigger()

  def __iadd__(self, other):
    res = super().__iadd__(other)
    self._trigger()
    return res

  def __imul__(self, other):
    res = super().__imul__(other)
    self._trigger()
    return res


@dataclass
class SessionConfig:
  provider: str
  model: str
  oracle_model: Optional[str] = None
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
  headless: bool = False
  whitelist: List[str] = field(default_factory=list)
  models: List[str] = field(default_factory=list)
  max_thinking_chars: int = 12000
  max_thinking_leeway_chars: int = 2000
  api_delay: float = 2.5


from chatty.ui import LazyMarkdown, optional_live, ChattyCompleter, LiveScreenLayout


class ChatbotSession:
  _active_session = None

  def __init__(
    self,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    oracle_model: Optional[str] = None,
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
    headless: bool = False,
    whitelist: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    max_thinking_chars: int = 12000,
    max_thinking_leeway_chars: int = 2000,
    api_delay: float = 2.5,
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
        oracle_model=oracle_model,
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
        prompt_caching=prompt_caching,
        headless=headless,
        whitelist=whitelist or [],
        models=models or ([model] if model else []),
        max_thinking_chars=max_thinking_chars,
        max_thinking_leeway_chars=max_thinking_leeway_chars,
        api_delay=api_delay
      )

    # Ensure static_skills defaults correctly if not provided
    if self.config.static_skills is None:
      self.config.static_skills = (self.config.provider == "openrouter")

    # Ensure sandbox path is absolute
    self.config.sandbox = os.path.abspath(self.config.sandbox)

    # Ensure models list is never empty if we have an active model
    if not self.config.models and self.config.model:
      self.config.models = [self.config.model]

    self.background_commands = {}
    self.whitelisted_script_signatures = set()
    self.next_task_id = 1
    self.max_completed_tasks = 10
    self._active_live = None
    
    self.runner = SubprocessRunner(session=self)
    # Register weakref finalizer for automatic cleanup on garbage collection or program exit
    self._finalizer = weakref.finalize(self, cleanup_resources, self.background_commands, self.sandbox)
    
    # Internal state
    self._cached_history_tokens = None
    self._messages = CachedList([], on_change=self._invalidate_token_cache)
    self.current_loop = 0
    default_prompt = (
      "You are a helpful assistant with local sandboxed file access and shell execution capabilities.\n"
      "You have tools for: listing directories (list_dir), locating files (locate_files), checking file info (get_file_info), reading files (read_file), writing files (write_file), copying files/directories (copy_file), moving/renaming files/directories (move_file), deleting files (delete_file), deleting directories (delete_directory), creating directories (make_directory), formatting files (format_file), patching files (patch_file), applying multiple patches (multi_patch), editing line ranges (edit_lines), applying multiple line range edits (multi_edit_lines), searching regex patterns (search_grep), fetching web content (fetch_url), executing shell commands (run_command), checking background tasks (check_background_command), peeking at background task output (peek_task_output), terminating background processes (kill_process), sleeping (sleep), and asking questions (ask_question).\n"
      "All paths provided to the tools will resolve relative to the sandbox directory.\n"
      "You are strictly prohibited from writing files outside the sandbox folder.\n"
      "CRITICAL: When you need to ask the user a question, clarify instructions, confirm decisions, or present a set of choices/options, you MUST use the dedicated 'ask_question' tool instead of asking questions in your conversational text response. This allows the CLI to prompt the user interactively and return their response to you in the tool execution loop.\n"
      "CRITICAL: You MUST use the dedicated, high-level filesystem tools (like list_dir, read_file, search_grep, locate_files, get_file_info, copy_file, move_file, delete_file, delete_directory, make_directory) instead of running command-line utilities (like grep, find, cat, head, tail, sed, awk, less, more, cp, mv, rm, rmdir, mkdir, ls) inside run_command. Shell execution using run_command is blocked for these actions and will return an error. You must use get_file_info instead of running 'wc' or 'wc -l' inside run_command.\n"
      "CRITICAL: For performing search-and-replace edits (similar to 'sed'), you MUST use 'multi_patch' (for multiple non-contiguous exact replacements), 'edit_lines' (for a single line number range), or 'multi_edit_lines' (for multiple non-contiguous line range edits) instead of using 'sed' or custom scripts in run_command.\n"
      "CRITICAL: When you need to reformat source code files or enforce layout/style guidelines (such as indentation, line-splitting, or spacing), you MUST use the dedicated 'format_file' tool instead of manually editing the files using 'edit_lines', 'patch_file', or 'multi_patch'.\n"
      "CRITICAL: You are strictly prohibited from using the shell 'sleep' command inside run_command to pause execution. You MUST use the dedicated 'sleep' tool instead.\n"
      "When running shell commands using run_command, if a command takes longer than 10 seconds, it will automatically transition to run in the background and return a 'Task ID'. You must NOT block. Instead, check its output or wait for its progress/completion by calling check_background_command with the Task ID and a timeout parameter. You can also peek at the accumulated output of a background task without blocking or waiting by calling peek_task_output. You are strictly prohibited from using the 'sleep' tool to wait for background commands; you MUST use check_background_command with a timeout parameter instead. Perform other file tasks (read, patch, edit) while waiting.\n"
      "To filter the output of run_command, use its optional 'output_filter' (regex), 'tail_lines', or 'head_lines' parameters rather than piping to grep or writing custom filtering scripts.\n"
      "When compilation, testing, verification, or running tools (like verilator, python scripts, compilers) is needed, you MUST execute them directly using the run_command tool instead of instructing the user to run them manually.\n"
      "CRITICAL: Keep your internal thinking/reasoning process concise, structured, and goal-oriented. Do not repeat the same thoughts or arguments. Transition to final content or tool calls as quickly as possible. If you are stuck or require more information, immediately stop thinking and state the problem or ask a question.\n"
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
    self.last_api_call_time = 0.0
    
    # Ensure sandbox exists
    os.makedirs(self.sandbox, exist_ok=True)
    os.chdir(self.sandbox)
    
    # Initialize client
    self.init_client()
    
    # Landlock sandboxing support
    self.landlock_bin = None
    if sys.platform == "linux" and self.sandbox:
      try:
        from chatty.landlock import compile_landlock_binary
        self.landlock_bin = compile_landlock_binary()
      except Exception as e:
        logger.warning(f"Could not initialize Landlock support: {e}")
    
    # Initialize and register commands registry
    self._commands = COMMANDS
    
    # Whitelist and Interactive Permission Sets
    self.allowed_ro_paths: Set[str] = set()
    self.allowed_rw_paths: Set[str] = set()
    self.temp_allowed_ro_paths: Set[str] = set()
    self.temp_allowed_rw_paths: Set[str] = set()
    
    # Process initial whitelist from arguments or config
    whitelist_paths = whitelist if whitelist is not None else (self.config.whitelist if hasattr(self.config, "whitelist") else [])
    for val in whitelist_paths:
      mode = "ro"
      path = val
      if ":" in val:
        if val.endswith(":ro"):
          mode = "ro"
          path = val[:-3]
        elif val.endswith(":rw"):
          mode = "rw"
          path = val[:-3]
      abs_path = os.path.realpath(path)
      if mode == "ro":
        self.allowed_ro_paths.add(abs_path)
        if not self.config.headless:
          console.print(f"[bold green]Added Whitelisted Read-Only path:[/bold green] {abs_path}")
      else:
        self.allowed_rw_paths.add(abs_path)
        if not self.config.headless:
          console.print(f"[bold green]Added Whitelisted Read-Write path:[/bold green] {abs_path}")
          
    self.load_skills()
    logger.info(f"ChatbotSession initialized. Provider: {self.provider}, Model: {self.model}, Sandbox: {self.sandbox}")

  @property
  def messages(self) -> List[Dict[str, Any]]:
    return self._messages

  @messages.setter
  def messages(self, value: List[Dict[str, Any]]):
    self._messages = CachedList(value, on_change=self._invalidate_token_cache)
    self._invalidate_token_cache()

  def _invalidate_token_cache(self):
    return _invalidate_token_cache(self)

  def _calculate_tokens_for_messages(self, messages: List[Dict[str, Any]]) -> int:
    return _calculate_tokens_for_messages(self, messages)

  @contextlib.contextmanager
  def _pause_live(self):
    live = getattr(self, "_active_live", None)
    if live is not None:
      live.stop()
      try:
        yield
      finally:
        live.start()
    else:
      yield

  def _print(self, *args, **kwargs):
    if not self.headless:
      console.print(*args, **kwargs)

  def __getattr__(self, name: str) -> Any:
    if name == "config":
      raise AttributeError("config not initialized yet")
    config = self.__dict__.get("config")
    if config is not None and hasattr(config, name):
      return getattr(config, name)
    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

  def __setattr__(self, name: str, value: Any):
    config = self.__dict__.get("config")
    if name != "config" and config is not None and hasattr(config, name):
      setattr(config, name, value)
    else:
      super().__setattr__(name, value)


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
    return init_client(self)

  def _throttle_request(self):
    return _throttle_request(self)

  def _format_api_error(self, e: Exception) -> str:
    return _format_api_error(self, e)

  def _is_retryable_exception(self, e: Exception) -> bool:
    return _is_retryable_exception(self, e)

  def _resolve_model_and_provider(self, model_name: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    return _resolve_model_and_provider(self, model_name)

  def get_oracle_model(self) -> str:
    """Get model ID of the oracle fallback based on active provider."""
    return get_oracle_model(self)

  def consult_oracle(self, query: str) -> str:
    """Consults the oracle model for suggestions/reasoning assistance."""
    return consult_oracle(self, query)

  def tool_run_tests(self, command: str = None) -> str:
    """Run tests in the sandbox, auto-detecting the testing framework if no command is provided."""
    return self.runner.tool_run_tests(command)

  def validate_command_safety(self, command: str) -> Optional[str]:
    """Validates that the shell command does not bypass dedicated tools."""
    return validate_command_safety(command, session=self)

  def tool_run_command(self, command: str, output_filter: Optional[str] = None, tail_lines: Optional[int] = None, head_lines: Optional[int] = None, combine_stderr: bool = False) -> str:
    """Execute a shell command, transitioning to background execution if it takes too long."""
    return self.runner.tool_run_command(command, output_filter, tail_lines, head_lines, combine_stderr)

  def tool_check_background_command(
    self,
    task_id: str,
    timeout: Optional[float] = None,
    output_filter: Optional[str] = None,
    tail_lines: Optional[int] = None,
    head_lines: Optional[int] = None,
  ) -> str:
    """Check status of a background task and read its currently accumulated stdout and stderr."""
    return self.runner.tool_check_background_command(task_id, timeout, output_filter, tail_lines, head_lines)

  def tool_peek_task_output(
    self,
    task_id: str,
    tail_lines: int = 20,
    output_filter: Optional[str] = None
  ) -> str:
    """Peek at the currently accumulated output of a background task without blocking."""
    return self.runner.tool_peek_task_output(task_id, tail_lines, output_filter)

  def tool_kill_process(self, task_id: str) -> str:
    """Terminate a background task/process by its Task ID."""
    return self.runner.tool_kill_process(task_id)

  def cleanup_background_commands(self):
    """Kills all active background tasks and removes temporary files."""
    self.runner.cleanup_background_commands()

  def _prune_background_commands(self):
    """Prunes old completed background commands."""
    return self.runner._prune_background_commands()

  def _get_pgroup_resources(self, pid: Any) -> Optional[Dict[str, Any]]:
    """Get process group resource usage."""
    return self.runner._get_pgroup_resources(pid)

  def _format_pgroup_resources(self, pid: Any) -> str:
    """Format the process group resources as a user-friendly string."""
    return self.runner._format_pgroup_resources(pid)

  def _apply_output_filters(self, text: str, output_filter: Optional[str] = None, head_lines: Optional[int] = None, tail_lines: Optional[int] = None) -> str:
    """Applies output filter regex, head/tail limits, and joins lines."""
    return self.runner._apply_output_filters(text, output_filter, head_lines, tail_lines)

  def prune_history(self, log: bool = True) -> List[Dict[str, Any]]:
    """Prunes conversation history to respect the configured context size, compressing older tool outputs."""
    return prune_history(self, log)

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
        if isinstance(args, str):
          try:
            json.loads(args)
            args_str = args
          except Exception:
            args_str = json.dumps(args)
        else:
          args_str = json.dumps(args)
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
          if isinstance(args, str):
            try:
              json.loads(args)
              args_str = args
            except Exception:
              args_str = json.dumps(args)
          else:
            args_str = json.dumps(args)
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
      try:
        data = json.loads(repair_json(text), strict=False)
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
        try:
          data = json.loads(repair_json(block.strip()), strict=False)
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
                  try:
                    data = json.loads(repair_json(potential), strict=False)
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
    return _log_llm_request(self, messages, tools)

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
    return _log_llm_response_summary(
      self,
      content_accumulated,
      tool_calls_accumulated,
      extra_fields_accumulated,
      finish_reason,
      usage,
      response_model,
      system_fingerprint,
      chunk_id
    )

  def run_llm_cycle(self):
    """Executes a full inference cycle, resolving tool calls recursively."""
    return run_llm_cycle(self)

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
      return handler(self, arg)
    
    console_print = self._print if hasattr(self, "_print") else console.print
    console_print(f"[bold red]Unknown command:[/bold red] {cmd}. Type [cyan]/help[/cyan] for options.")
    return True

  def compress_context(self, keep_messages: Optional[int] = None):
    """Summarizes the history, clears the context, and reloads the summary."""
    if not self.messages:
      self._print("[bold yellow]History is empty. Nothing to compress.[/bold yellow]")
      return

    if keep_messages is None:
      keep_messages = self.history_keep_messages

    keep_messages = min(keep_messages, len(self.messages))
    keep_msgs = self.messages[-keep_messages:] if keep_messages > 0 else []

    # Prepare messages for summarization
    active_messages = self.prune_history()
    summary_instruction = (
      "Summarize our progress and active context. You MUST structure your response exactly as follows:\n\n"
      "### 📋 Goal & Task Context\n"
      "<Describe the current high-level goal and active subtasks>\n\n"
      "### 🛠️ Codebase Modifications & Environment State\n"
      "<List files created/modified, active paths/permissions, and critical variables>\n\n"
      "### ⚠️ Immediate Issues or Failures\n"
      "<List recent failing tests, compilation errors, or blockages>\n\n"
      "### ⏭️ Next Steps\n"
      "<List 1-3 immediate action items>\n\n"
      "Keep the summary concise but preserve all technical details, filenames, "
      "function names, paths, and key design decisions."
    )

    # Check for previous summaries in active messages
    has_prev_summary = any(
      "Summarize our progress and task context" in (msg.get("content") or "")
      for msg in active_messages
    )
    if has_prev_summary:
      summary_instruction += (
        "\n\nIMPORTANT: A previous summary is present in the history. You must carry "
        "forward all unresolved goals, code modifications, background context, and system "
        "state from it into the new summary."
      )

    active_messages.append({"role": "user", "content": summary_instruction})

    # Log request details in DEBUG mode
    self._log_llm_request(active_messages, None)

    content_accumulated = ""
    logger.info("Generating summary for /compress command")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
      try:
        panels = [{"title": "Context Compression", "content": "Connecting to LLM for summary...", "border_style": "yellow"}]
        with optional_live(LiveScreenLayout(panels, self.get_rich_status_bar()),
                           console=console, enabled=not self.headless, refresh_per_second=12) as live:
          
          # Resolve model and provider
          actual_model, extra_body = self._resolve_model_and_provider(self.model)
          
          self._throttle_request()
          try:
            kwargs = {
              "model": actual_model,
              "messages": active_messages,
              "stream": True,
              "stream_options": {"include_usage": True}
            }
            if extra_body:
              kwargs["extra_body"] = extra_body
            stream = self.client.chat.completions.create(**kwargs)
          except Exception as e:
            if self._is_retryable_exception(e):
              raise
            logger.debug(f"Failed to call API with stream_options: {e}. Retrying without stream_options.")
            self._throttle_request()
            kwargs = {
              "model": actual_model,
              "messages": active_messages,
              "stream": True
            }
            if extra_body:
              kwargs["extra_body"] = extra_body
            stream = self.client.chat.completions.create(**kwargs)
          
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
                panels[0] = {"title": "Context Summary", "content": "", "border_style": "yellow"}
                first_content_chunk = False
              content_accumulated += delta.content
              panels[0]["content"] = content_accumulated
              live.update(LiveScreenLayout(panels, self.get_rich_status_bar()))
          
          # Remove status bar before exiting Live context
          live.update(LiveScreenLayout(panels, None))
              
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
        self.last_api_call_time = time.time()
        break
      except Exception as e:
        self.last_api_call_time = time.time()
        logger.exception("Error calling LLM API for context summary")
        formatted_err = self._format_api_error(e)
        if attempt < max_retries and self._is_retryable_exception(e):
          err_msg = str(e).lower()
          is_rate_limit = any(ind in err_msg for ind in [
            "rate limit", "rate-limit", "rate_limited", "rate-limited",
            "too many requests", "high-frequency", "risk_control", "risk control"
          ])
          backoff_time = 5 * (2 ** attempt) if is_rate_limit else 2 ** attempt
          self._print(f"[bold yellow]⚠️  Error calling API for summary: {formatted_err}. Retrying in {backoff_time}s (attempt {attempt}/{max_retries})...[/bold yellow]")
          time.sleep(backoff_time)
          content_accumulated = ""
          continue
        else:
          self._print(f"[bold red]Error calling API for summary:[/bold red] {formatted_err}")
          return

    if not content_accumulated.strip():
      self._print("[bold red]Failed to generate summary. Context was not cleared.[/bold red]")
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
    if keep_msgs:
      self.messages.extend(keep_msgs)
    
    self._print("[bold green]Conversation history cleared and recap reloaded.[/bold green]")

  def save_session(self, file_path: str):
    """Saves session state to a JSON file."""
    file_path = os.path.expanduser(file_path)
    if not os.path.isabs(file_path):
      file_path = os.path.join(self.sandbox, file_path)
    dir_name = os.path.dirname(file_path)
    if dir_name:
      os.makedirs(dir_name, exist_ok=True)
    session_data = {
      "provider": self.provider,
      "model": self.model,
      "models": self.models,
      "oracle_model": getattr(self, "oracle_model", None),
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
    with open(file_path, "w", encoding="utf-8") as f:
      json.dump(session_data, f, indent=2, default=str)

  def load_session(self, file_path: str):
    """Loads session state from a JSON file."""
    file_path = os.path.expanduser(file_path)
    if not os.path.isabs(file_path):
      file_path = os.path.join(self.sandbox, file_path)
    with open(file_path, "r", encoding="utf-8") as f:
      session_data = json.load(f)
    if "provider" in session_data:
      self.provider = session_data["provider"]
    if "model" in session_data:
      self.model = session_data["model"]
    if "models" in session_data:
      self.models = session_data["models"]
    if "oracle_model" in session_data:
      self.oracle_model = session_data["oracle_model"]
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

  def add_whitelist_path(self, path: str, mode: str = "rw") -> str:
    """Adds a path to the whitelist for read-only or read-write access."""
    abs_path = os.path.realpath(path)
    if mode == "ro":
      self.allowed_ro_paths.add(abs_path)
    else:
      self.allowed_rw_paths.add(abs_path)
    return abs_path

  def remove_whitelist_path(self, path: str) -> Tuple[str, bool]:
    """Removes a path from the whitelist."""
    abs_path = os.path.realpath(path)
    removed = False
    if abs_path in self.allowed_ro_paths:
      self.allowed_ro_paths.remove(abs_path)
      removed = True
    if abs_path in self.allowed_rw_paths:
      self.allowed_rw_paths.remove(abs_path)
      removed = True
    return abs_path, removed

  def clear_whitelist_paths(self):
    """Clears all whitelisted paths."""
    self.allowed_ro_paths.clear()
    self.allowed_rw_paths.clear()

  def has_path_permission(self, path: str, write: bool = False) -> bool:
    """Checks if an absolute path is whitelisted for the given access mode."""
    abs_path = os.path.realpath(path)
    
    # A path whitelisted for RW implicitly has RO access as well
    for allowed in self.allowed_rw_paths:
      if os.path.commonpath([allowed, abs_path]) == allowed:
        return True
    for allowed in self.temp_allowed_rw_paths:
      if os.path.commonpath([allowed, abs_path]) == allowed:
        return True
        
    # Check RO whitelist only if we don't need write access
    if not write:
      for allowed in self.allowed_ro_paths:
        if os.path.commonpath([allowed, abs_path]) == allowed:
          return True
      for allowed in self.temp_allowed_ro_paths:
        if os.path.commonpath([allowed, abs_path]) == allowed:
          return True
          
    return False

  def prompt_for_path_permission(self, target_path: str, write: bool = False) -> bool:
    """Prompts the user to allow/deny access to a path, specifying RO/RW mode."""
    if self.headless:
      return False

    abs_path = os.path.realpath(target_path)
    mode_str = "READ-WRITE" if write else "READ-ONLY"
    
    self._print(f"\n[bold yellow]⚠️  Warning: AI is attempting {mode_str} access to a path outside the sandbox:[/bold yellow]")
    self._print(f"   Target path: [cyan]{abs_path}[/cyan]\n")
    
    try:
      with self._pause_live():
        self._print(
          f"Allow {mode_str} access? "
          f"[bold green]\\[y][/bold green]es / "
          f"[bold red]\\[n][/bold red]o / "
          f"[bold cyan]\\[a][/bold cyan]lways (whitelist file) / "
          f"[bold magenta]\\[p][/bold magenta]arents (whitelist upper folder): ",
          end=""
        )
        response = input().strip().lower()
        
        if response == 'a':
          if write:
            self.allowed_rw_paths.add(abs_path)
          else:
            self.allowed_ro_paths.add(abs_path)
          logger.info(f"User whitelisted {mode_str} access to: {abs_path}")
          return True
          
        elif response == 'p':
          # Compile parent directories up to the root
          parents = []
          curr = os.path.dirname(abs_path)
          while True:
            parents.append(curr)
            next_curr = os.path.dirname(curr)
            if next_curr == curr:
              break
            curr = next_curr
            
          # Limit display to the top 5 closest parent directories to keep terminal clean
          parents = parents[:5]
          
          self._print("\n[bold cyan]Select an upper directory to whitelist recursively:[/bold cyan]")
          for idx, parent in enumerate(parents, 1):
            self._print(f"  {idx}: {parent}")
          self._print("  c: Cancel\n")
          
          self._print("Enter choice (1-N or c): ", end="")
          choice = input().strip().lower()
          if choice == 'c':
            return False
          try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(parents):
              chosen_parent = parents[choice_idx]
              if write:
                self.allowed_rw_paths.add(chosen_parent)
              else:
                self.allowed_ro_paths.add(chosen_parent)
              logger.info(f"User whitelisted parent path {mode_str} access to: {chosen_parent}")
              return True
            else:
              self._print("[bold red]Invalid selection.[/bold red]")
              return False
          except ValueError:
            self._print("[bold red]Invalid selection.[/bold red]")
            return False
            
        elif response in ('y', 'yes'):
          if write:
            self.temp_allowed_rw_paths.add(abs_path)
          else:
            self.temp_allowed_ro_paths.add(abs_path)
          logger.info(f"User allowed {mode_str} access once to: {abs_path}")
          return True
        else:
          logger.info(f"User denied access to: {abs_path}")
          return False
    except (KeyboardInterrupt, EOFError):
      return False

  def prompt_for_script_permission(self, code_str: str, signature: str) -> bool:
    """Prompts the user to allow/deny execution of a Python script with an option to whitelist its signature."""
    if self.headless:
      return False

    self._print("\n[bold yellow]================================================================================[/bold yellow]")
    self._print("[bold yellow]⚠️  The LLM is trying to execute a Python script that accesses the filesystem.[/bold yellow]")
    self._print("[bold yellow]================================================================================[/bold yellow]\n")

    # Render syntax highlighted script
    from rich.syntax import Syntax
    syntax = Syntax(code_str, "python", theme="monokai", line_numbers=True)
    self._print(Panel(syntax, title="[cyan]Python Code[/cyan]"))

    self._print("\n[bold cyan]Signature (Operations):[/bold cyan]")
    if signature:
      for op in signature.split(','):
        self._print(f"  - [yellow]{op}[/yellow]")
    else:
      self._print("  - [yellow](No forbidden filesystem operations detected in AST)[/yellow]")
    self._print("")

    try:
      with self._pause_live():
        self._print(
          "Allow execution? "
          "[bold green]\\[y][/bold green]es / "
          "[bold red]\\[n][/bold red]o / "
          "[bold cyan]\\[w][/bold cyan]hitelist (allow this class of scripts in this session): ",
          end=""
        )
        response = input().strip().lower()
        if response == 'w':
          self.whitelisted_script_signatures.add(signature)
          logger.info(f"User whitelisted script signature: {signature}")
          return True
        elif response in ('y', 'yes'):
          logger.info("User allowed script execution once")
          return True
        else:
          logger.info("User denied script execution")
          return False
    except (KeyboardInterrupt, EOFError):
      return False

  def show_whitelist(self):
    """Displays whitelisted paths and their permissions."""
    from chatty.ui import show_whitelist
    show_whitelist(self)

  def show_help(self):
    """Displays formatted CLI usage guide."""
    from chatty.ui import show_help
    show_help(self)

  def show_status(self):
    """Displays configured status parameters."""
    from chatty.ui import show_status
    show_status(self)

  def show_tool_stats(self):
    """Displays statistics on tool and external binary calls."""
    from chatty.ui import show_tool_stats
    show_tool_stats(self)

  def show_tools(self):
    """Lists available filesystem functions."""
    from chatty.ui import show_tools
    show_tools(self)

  def get_rich_status_bar(self):
    """Returns a Rich Table rendering the status bar."""
    from chatty.ui import get_rich_status_bar
    return get_rich_status_bar(self)

  def start_loop(self):
    """Runs the interactive input/output CLI loop."""
    from chatty.ui import start_interactive_loop
    start_interactive_loop(self)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.cleanup_background_commands()
