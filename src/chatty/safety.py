import ast
import os
import re
import shlex
from typing import List, Optional, Set, Any
import contextvars

active_session_var = contextvars.ContextVar("active_session", default=None)


def get_safe_path(sandbox_dir: str, target_path: str, write: bool = False) -> str:
  """
  Resolves the absolute path of target_path and ensures it lies strictly inside sandbox_dir.
  Raises PermissionError if a directory traversal attempt is made.
  """
  abs_sandbox = os.path.realpath(sandbox_dir)
  
  # If path is absolute, check it. Otherwise, combine it with sandbox.
  if os.path.isabs(target_path):
    resolved_path = os.path.realpath(target_path)
  else:
    resolved_path = os.path.realpath(os.path.join(abs_sandbox, target_path))
    
  # Check if target is inside the sandbox folder by finding their common prefix
  common_prefix = os.path.commonpath([abs_sandbox, resolved_path])
  if common_prefix != abs_sandbox:
    session = active_session_var.get()
    if session:
      if session.has_path_permission(resolved_path, write=write):
        return resolved_path
      if session.prompt_for_path_permission(resolved_path, write=write):
        return resolved_path

    mode_err = "Write/Modify" if write else "Read"
    raise PermissionError(
      f"Access Denied: {mode_err} Access to path '{target_path}' resolves to '{resolved_path}' "
      f"which is outside the sandbox directory '{abs_sandbox}'."
    )
  return resolved_path



def load_ignore_patterns(sandbox_dir: str) -> List[str]:
  """Loads default ignore patterns and appends patterns from .gitignore if present."""
  patterns = [
    ".git",
    ".chatty",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".DS_Store",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".db",
    ".sqlite",
    ".bin"
  ]
  gitignore_path = os.path.join(sandbox_dir, ".gitignore")
  if os.path.exists(gitignore_path):
    try:
      with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            if line.endswith('/'):
              line = line[:-1]
            patterns.append(line)
    except Exception:
      pass
  return list(set(patterns))


def is_path_ignored(rel_path: str, patterns: List[str]) -> bool:
  """Checks if a relative path matches any of the ignore patterns."""
  import fnmatch
  rel_path_norm = rel_path.replace(os.sep, '/')
  segments = rel_path_norm.split('/')
  prefixes = []
  current = []
  for segment in segments:
    current.append(segment)
    prefixes.append('/'.join(current))
  for pattern in patterns:
    pattern_norm = pattern.replace(os.sep, '/')
    for segment in segments:
      if fnmatch.fnmatch(segment, pattern_norm):
        return True
    for prefix in prefixes:
      if fnmatch.fnmatch(prefix, pattern_norm):
        return True
      if prefix.startswith(pattern_norm + '/'):
        return True
  return False




def is_text_file(file_path: str) -> bool:
  """Check if a file is a text file by scanning for null bytes."""
  try:
    with open(file_path, "rb") as f:
      chunk = f.read(8192)
      return b"\0" not in chunk
  except Exception:
    return False


def count_lines(file_path: str) -> int:
  """Count the number of lines in a text file."""
  count = 0
  try:
    with open(file_path, "rb") as f:
      last_char = None
      for chunk in iter(lambda: f.read(65536), b""):
        if chunk:
          count += chunk.count(b"\n")
          last_char = chunk[-1]
      if last_char is not None and last_char != 10:
        count += 1
  except Exception:
    pass
  return count


def get_structural_signature(code_str: str) -> str:
  try:
    tree = ast.parse(code_str)
  except SyntaxError:
    return "invalid_syntax"

  ops = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for name in node.names:
        ops.add(f"import:{name.name.split('.')[0]}")
    elif isinstance(node, ast.ImportFrom):
      if node.module:
        ops.add(f"import:{node.module.split('.')[0]}")
    elif isinstance(node, ast.Call):
      if isinstance(node.func, ast.Name):
        ops.add(f"call:{node.func.id}")
      elif isinstance(node.func, ast.Attribute):
        parts = []
        curr = node.func
        while isinstance(curr, ast.Attribute):
          parts.append(curr.attr)
          curr = curr.value
        if isinstance(curr, ast.Name):
          parts.append(curr.id)
        parts.reverse()
        ops.add(f"call:{'.'.join(parts)}")
  return ",".join(sorted(ops))


def is_signature_whitelisted(sig: str, whitelisted_sigs: Set[str]) -> bool:
  if sig in whitelisted_sigs:
    return True
  sig_set = set(sig.split(',')) if sig else set()
  for w_sig in whitelisted_sigs:
    w_set = set(w_sig.split(',')) if w_sig else set()
    if sig_set.issubset(w_set):
      return True
  return False


def has_forbidden_filesystem_ops(sig: str) -> bool:
  if not sig or sig == "invalid_syntax":
    return False
  for op in sig.split(','):
    if op.startswith("import:") and op.split(":")[1] in {"os", "shutil", "pathlib", "glob", "fnmatch"}:
      return True
    if op == "call:open":
      return True
    if op.startswith("call:") and op.split(":")[1].split(".")[0] in {"os", "shutil", "pathlib", "glob", "fnmatch"}:
      return True
  return False


def validate_command_safety(command: str, session: Optional[Any] = None) -> Optional[str]:
  """Validates that the shell command does not bypass dedicated tools."""
  
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
        
        token_name = clean_token.split('=')[0]
        if token_name in {'tail_lines', 'head_lines', 'output_filter'}:
          return (
            f"Error: Using '{token_name}' inside the command string is prohibited. "
            f"Please pass '{token_name}' as a separate tool argument (JSON parameter) "
            "to run_command instead of piping or appending it inside the command string."
          )
          
        if clean_token in {'python', 'python3', 'py', 'python2'}:
          active_session = session if session is not None else active_session_var.get()
          script_code = None
          script_name = "inline script"
          
          # Check for inline script (-c "...")
          for idx, t in enumerate(tokens):
            if t == "-c" and idx + 1 < len(tokens):
              script_code = tokens[idx + 1]
              break
              
          # Check for file script
          if not script_code:
            for t in tokens:
              if t.endswith(".py"):
                target_path = t
                if active_session and not os.path.isabs(target_path):
                  target_path = os.path.join(active_session.sandbox, target_path)
                if os.path.exists(target_path):
                  try:
                    with open(target_path, "r", encoding="utf-8") as f:
                      script_code = f.read()
                    script_name = t
                  except Exception:
                    pass
                  break

          if script_code:
            sig = get_structural_signature(script_code)
            if has_forbidden_filesystem_ops(sig):
              if active_session:
                if is_signature_whitelisted(sig, active_session.whitelisted_script_signatures):
                  # Whitelisted, allow!
                  pass
                else:
                  # Prompt user
                  allowed = active_session.prompt_for_script_permission(script_code, sig)
                  if not allowed:
                    return (
                      f"Error: Executing Python script ({script_name}) with direct filesystem operations is prohibited. "
                      "Please use the dedicated workspace tools (like read_file, write_file, search_grep, list_dir) instead."
                    )
              else:
                return (
                  f"Error: Executing Python script ({script_name}) with direct filesystem operations is prohibited. "
                  "Please use the dedicated workspace tools (like read_file, write_file, search_grep, list_dir) instead."
                )
        
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
            "Please use the dedicated 'patch_file' tool (supporting Aider-style SEARCH/REPLACE blocks) instead."
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
