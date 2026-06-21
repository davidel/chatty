#!/usr/bin/env python3
import os
import sys
import argparse
import json
import re
import urllib.parse
import requests
import openai
import tiktoken
from typing import List, Dict, Any, Tuple

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.columns import Columns

# Prompt toolkit for advanced input capabilities
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

# Initialize global Rich console
console = Console()

# --- Sandboxed File System Tools ---

def get_safe_path(sandbox_dir: str, target_path: str) -> str:
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
    raise PermissionError(
      f"Access Denied: Path '{target_path}' resolves to '{resolved_path}' "
      f"which is outside the sandbox directory '{abs_sandbox}'."
    )
  return resolved_path

def tool_list_dir(sandbox_dir: str, path: str = ".") -> str:
    """List the contents of a directory path inside the sandbox."""
    try:
        safe_p = get_safe_path(sandbox_dir, path)
        if not os.path.exists(safe_p):
            return f"Error: Path '{path}' does not exist."
        if not os.path.isdir(safe_p):
            return f"Error: Path '{path}' is not a directory."
            
        items = os.listdir(safe_p)
        result = []
        for item in sorted(items):
            full_path = os.path.join(safe_p, item)
            rel_path = os.path.relpath(full_path, sandbox_dir)
            if os.path.isdir(full_path):
                result.append(f"[DIR]  {rel_path}/")
            else:
                size = os.path.getsize(full_path)
                result.append(f"[FILE] {rel_path} ({size} bytes)")
        return "\n".join(result) if result else "(Empty directory)"
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def tool_read_file(sandbox_dir: str, path: str, start_line: int = None, end_line: int = None) -> str:
  """Read the contents of a file inside the sandbox, optionally specifying a 1-indexed line range."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: File '{path}' does not exist."
    if not os.path.isfile(safe_p):
      return f"Error: Path '{path}' is not a file."
      
    with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
      if start_line is None and end_line is None:
        return f.read()
        
      lines = f.readlines()
      total_lines = len(lines)
      
      s = 1 if start_line is None else start_line
      e = total_lines if end_line is None else end_line
      
      if s < 1 or s > total_lines:
        return f"Error: start_line {start_line} is out of range. The file '{path}' has {total_lines} lines."
      if e < s or e > total_lines:
        return f"Error: end_line {end_line} is invalid (must be between start_line {s} and total file lines {total_lines})."
        
      return "".join(lines[s-1:e])
  except Exception as e:
    return f"Error reading file: {str(e)}"


def load_ignore_patterns(sandbox_dir: str) -> List[str]:
  """Loads default ignore patterns and appends patterns from .gitignore if present."""
  patterns = [
    ".git",
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
  segments = rel_path.split(os.sep)
  for pattern in patterns:
    for segment in segments:
      if fnmatch.fnmatch(segment, pattern):
        return True
    if fnmatch.fnmatch(rel_path, pattern):
      return True
  return False


def tool_locate_files(sandbox_dir: str, pattern: str, path: str = ".") -> str:
  """Locate files recursively inside the sandbox directory matching a glob pattern, ignoring files in .gitignore and common cache directories."""
  import fnmatch
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: Path '{path}' does not exist."
    if not os.path.isdir(safe_p):
      return f"Error: Path '{path}' is not a directory."
      
    ignore_patterns = load_ignore_patterns(sandbox_dir)
    results = []
    
    for root, dirs, files in os.walk(safe_p):
      for d in list(dirs):
        dir_path = os.path.join(root, d)
        try:
          safe_dir = get_safe_path(sandbox_dir, dir_path)
          rel_dir = os.path.relpath(safe_dir, sandbox_dir)
          if is_path_ignored(rel_dir, ignore_patterns):
            dirs.remove(d)
        except PermissionError:
          dirs.remove(d)
          
      for file in files:
        full_path = os.path.join(root, file)
        try:
          safe_file_path = get_safe_path(sandbox_dir, full_path)
          rel_path = os.path.relpath(safe_file_path, sandbox_dir)
          if is_path_ignored(rel_path, ignore_patterns):
            continue
          if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(rel_path, pattern):
            results.append(rel_path)
        except Exception:
          continue
          
    return "\n".join(results) if results else "No matching files found."
  except Exception as e:
    return f"Error locating files: {str(e)}"


def tool_get_file_info(sandbox_dir: str, path: str) -> str:
  """Get metadata info (size, last modified, type) about a path inside the sandbox."""
  import time
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: Path '{path}' does not exist."
      
    stat = os.stat(safe_p)
    is_dir = os.path.isdir(safe_p)
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
    
    info = [
      f"Path: {path}",
      f"Type: {'Directory' if is_dir else 'File'}",
      f"Size: {stat.st_size} bytes",
      f"Last Modified: {mtime}"
    ]
    return "\n".join(info)
  except Exception as e:
    return f"Error getting file info: {str(e)}"


def tool_fetch_url(url: str) -> str:
  """Fetch the text content of a public URL and convert it to clean text (removes HTML tags)."""
  try:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    html = response.text
    
    html_clean = re.sub(r'<(script|style|head|header|footer|nav).*?>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html_clean = re.sub(r'<br\s*/?>', '\n', html_clean, flags=re.IGNORECASE)
    html_clean = re.sub(r'</?(p|div|li|h[1-6]).*?>', '\n', html_clean, flags=re.IGNORECASE)
    text = re.sub(r'<.*?>', '', html_clean, flags=re.DOTALL)
    import html as html_parser
    text = html_parser.unescape(text)
    
    cleaned_lines = []
    for line in text.split('\n'):
      stripped = line.strip()
      if stripped:
        cleaned_lines.append(stripped)
      elif cleaned_lines and cleaned_lines[-1] != "":
        cleaned_lines.append("")
    return "\n".join(cleaned_lines).strip()
  except Exception as e:
    return f"Error fetching URL: {str(e)}"


def validate_file_syntax(path: str, content: str) -> Tuple[bool, str]:
  """
  Checks syntax of content based on file extension.
  Returns (True, "") if syntax is valid or language has no validator.
  Returns (False, "Error message") if syntax verification fails.
  """
  ext = os.path.splitext(path)[1].lower()
  
  if ext == ".py":
    import ast
    try:
      ast.parse(content)
    except SyntaxError as e:
      return False, f"Syntax Error: {e.msg} on line {e.lineno}, column {e.offset}\nLine content: {e.text}"
      
  elif ext == ".json":
    import json
    try:
      json.loads(content)
    except json.JSONDecodeError as e:
      return False, f"JSON Parsing Error: {e.msg} at line {e.lineno}, column {e.colno}"
      
  elif ext in (".yaml", ".yml"):
    try:
      import yaml
      yaml.safe_load(content)
    except Exception as e:
      return False, f"YAML Parsing Error: {str(e)}"
      
  elif ext in (".c", ".cpp", ".h", ".hpp"):
    import subprocess
    import tempfile
    import shutil
    try:
      with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode='w+t') as temp:
        temp.write(content)
        temp_name = temp.name
      try:
        is_cpp = ext in (".cpp", ".hpp")
        if is_cpp:
          compiler = "clang++" if shutil.which("clang++") else "g++"
        else:
          compiler = "clang" if shutil.which("clang") else "gcc"
          
        proc = subprocess.run(
          [compiler, "-fsyntax-only", temp_name],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          timeout=3
        )
        if proc.returncode != 0:
          err_msg = proc.stderr.replace(temp_name, os.path.basename(path))
          return False, f"C/C++ Compiler Error:\n{err_msg}"
      finally:
        try:
          os.unlink(temp_name)
        except Exception:
          pass
    except Exception:
      pass

  elif ext in (".v", ".sv", ".vh", ".svh"):
    import subprocess
    import tempfile
    import shutil
    try:
      with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode='w+t') as temp:
        temp.write(content)
        temp_name = temp.name
      try:
        if shutil.which("verilator"):
          proc = subprocess.run(
            ["verilator", "--lint-only", "-Wno-fatal", temp_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
          )
          if proc.returncode != 0:
            err_msg = (proc.stderr + proc.stdout).replace(temp_name, os.path.basename(path))
            return False, f"Verilator Lint Error:\n{err_msg}"
        elif shutil.which("iverilog"):
          proc = subprocess.run(
            ["iverilog", "-g2012", "-t", "null", temp_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
          )
          if proc.returncode != 0:
            err_msg = (proc.stderr + proc.stdout).replace(temp_name, os.path.basename(path))
            return False, f"Icarus Verilog Syntax Error:\n{err_msg}"
      finally:
        try:
          os.unlink(temp_name)
        except Exception:
          pass
    except Exception:
      pass

  elif ext in (".vhd", ".vhdl"):
    import subprocess
    import tempfile
    import shutil
    try:
      with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode='w+t') as temp:
        temp.write(content)
        temp_name = temp.name
      try:
        if shutil.which("ghdl"):
          proc = subprocess.run(
            ["ghdl", "-s", temp_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
          )
          if proc.returncode != 0:
            err_msg = (proc.stderr + proc.stdout).replace(temp_name, os.path.basename(path))
            return False, f"GHDL Syntax Error:\n{err_msg}"
      finally:
        try:
          os.unlink(temp_name)
        except Exception:
          pass
    except Exception:
      pass
      
  return True, ""


def print_diff(path: str, old_content: str, new_content: str):
  """Renders a beautiful color-coded diff of file changes to the console."""
  import difflib
  from rich.text import Text
  from rich.panel import Panel
  
  old_lines = old_content.splitlines(keepends=True)
  new_lines = new_content.splitlines(keepends=True)
  
  diff = list(difflib.unified_diff(
    old_lines,
    new_lines,
    fromfile=f"old/{path}",
    tofile=f"new/{path}",
    n=3
  ))
  
  if not diff:
    return
    
  text = Text()
  for line in diff:
    if line.startswith('+') and not line.startswith('+++'):
      text.append(line, style="green")
    elif line.startswith('-') and not line.startswith('---'):
      text.append(line, style="red")
    elif line.startswith('@@'):
      text.append(line, style="cyan")
    elif line.startswith('---') or line.startswith('+++'):
      text.append(line, style="bold white")
    else:
      text.append(line, style="dim white")
      
  console.print(Panel(
    text,
    title=f"📝 File Changes: {os.path.basename(path)}",
    border_style="magenta"
  ))


def tool_write_file(sandbox_dir: str, path: str, content: str) -> str:
  """Write text content to a file inside the sandbox."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    is_valid, err_msg = validate_file_syntax(safe_p, content)
    if not is_valid:
      return f"Error: Syntax verification failed. File was not saved.\n{err_msg}"
      
    rel_path = os.path.relpath(safe_p, sandbox_dir)
    old_content = ""
    if os.path.exists(safe_p):
      try:
        with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
          old_content = f.read()
      except Exception:
        pass
        
    os.makedirs(os.path.dirname(safe_p), exist_ok=True)
    with open(safe_p, 'w', encoding='utf-8') as f:
      f.write(content)
      
    if old_content:
      print_diff(rel_path, old_content, content)
      
    return f"Successfully wrote to file '{rel_path}'."
  except Exception as e:
    return f"Error writing file: {str(e)}"

def tool_search_grep(sandbox_dir: str, pattern: str, path: str = ".") -> str:
  """Search for a regex pattern inside files in the sandbox, ignoring files in .gitignore and common cache directories."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: Path '{path}' does not exist."
      
    regex = re.compile(pattern, re.IGNORECASE)
    results = []
    ignore_patterns = load_ignore_patterns(sandbox_dir)
    
    for root, dirs, files in os.walk(safe_p):
      for d in list(dirs):
        dir_path = os.path.join(root, d)
        try:
          safe_dir = get_safe_path(sandbox_dir, dir_path)
          rel_dir = os.path.relpath(safe_dir, sandbox_dir)
          if is_path_ignored(rel_dir, ignore_patterns):
            dirs.remove(d)
        except PermissionError:
          dirs.remove(d)
          
      for file in files:
        file_path = os.path.join(root, file)
        try:
          safe_file_path = get_safe_path(sandbox_dir, file_path)
          rel_path = os.path.relpath(safe_file_path, sandbox_dir)
          if is_path_ignored(rel_path, ignore_patterns):
            continue
            
          with open(safe_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
              if regex.search(line):
                results.append(f"{rel_path}:{line_num}: {line.strip()}")
        except Exception:
          continue
          
    return "\n".join(results) if results else "No matches found."
  except Exception as e:
    return f"Error searching files: {str(e)}"


def tool_patch_file(sandbox_dir: str, path: str, search: str, replace: str) -> str:
  """Replace a specific unique block of text/code in a file with new content."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: File '{path}' does not exist. Use write_file to create new files."
    if not os.path.isfile(safe_p):
      return f"Error: Path '{path}' is not a file."
      
    with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
      content = f.read()
      
    # Normalize carriage returns for more robust matching (handles \r\n vs \n differences)
    normalized_content = content.replace("\r\n", "\n")
    normalized_search = search.replace("\r\n", "\n")
    normalized_replace = replace.replace("\r\n", "\n")
    
    if normalized_search not in normalized_content:
      return (
        f"Error: The search block was not found in '{path}'. "
        "Make sure you specify the search text exactly including leading whitespace and indentation. "
        "Note: Line endings (CRLF vs LF) are automatically normalized and ignored."
      )
      
    occurrences = normalized_content.count(normalized_search)
    if occurrences > 1:
      return (
        f"Error: Found {occurrences} occurrences of the search block in '{path}'. "
        "Please provide more context lines to make the search block unique."
      )
      
    # Perform replacement on the normalized content
    updated_normalized = normalized_content.replace(normalized_search, normalized_replace, 1)
    
    # Restore the original file's dominant line ending style
    if "\r\n" in content:
      new_content = updated_normalized.replace("\n", "\r\n")
    else:
      new_content = updated_normalized
      
    is_valid, err_msg = validate_file_syntax(safe_p, new_content)
    if not is_valid:
      return f"Error: Syntax verification failed for updated file content. Patch was not applied.\n{err_msg}"
      
    with open(safe_p, 'w', encoding='utf-8') as f:
      f.write(new_content)
      
    rel_path = os.path.relpath(safe_p, sandbox_dir)
    print_diff(rel_path, content, new_content)
    return f"Successfully updated file '{rel_path}' using a target replacement patch."
  except Exception as e:
    return f"Error patching file: {str(e)}"


def tool_edit_lines(sandbox_dir: str, path: str, start_line: int, end_line: int, replacement: str) -> str:
  """Replace a range of lines in a file (1-indexed, inclusive) with new content."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: File '{path}' does not exist. Use write_file to create new files."
    if not os.path.isfile(safe_p):
      return f"Error: Path '{path}' is not a file."
      
    with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
      lines = f.readlines()
      
    original_content = "".join(lines)
    total_lines = len(lines)
    if start_line < 1 or start_line > total_lines:
      return f"Error: start_line {start_line} is out of range. The file '{path}' has {total_lines} lines."
    if end_line < start_line or end_line > total_lines:
      return f"Error: end_line {end_line} is invalid (must be between start_line {start_line} and total file lines {total_lines})."
      
    has_crlf = any("\r\n" in line for line in lines)
    
    if replacement == "":
      replacement_lines_formatted = []
    else:
      replacement_normalized = replacement.replace("\r\n", "\n")
      replacement_lines = replacement_normalized.split("\n")
      suffix = "\r\n" if has_crlf else "\n"
      if replacement_normalized.endswith("\n") and len(replacement_lines) > 1 and replacement_lines[-1] == "":
        replacement_lines = replacement_lines[:-1]
      replacement_lines_formatted = [line + suffix for line in replacement_lines]
      
    slice_start = start_line - 1
    slice_end = end_line
    lines[slice_start:slice_end] = replacement_lines_formatted
    
    new_content = "".join(lines)
    is_valid, err_msg = validate_file_syntax(safe_p, new_content)
    if not is_valid:
      return f"Error: Syntax verification failed for updated file content. Line edits were not applied.\n{err_msg}"
      
    with open(safe_p, 'w', encoding='utf-8') as f:
      f.writelines(lines)
      
    rel_path = os.path.relpath(safe_p, sandbox_dir)
    print_diff(rel_path, original_content, new_content)
    replaced_count = end_line - start_line + 1
    inserted_count = len(replacement_lines_formatted)
    return (
      f"Successfully updated file '{rel_path}': replaced lines {start_line}-{end_line} "
      f"({replaced_count} lines) with {inserted_count} new lines."
    )
  except Exception as e:
    return f"Error editing lines: {str(e)}"


def tool_format_file(sandbox_dir: str, path: str) -> str:
  """Automatically format a source code file using the appropriate formatter."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: File '{path}' does not exist."
    if not os.path.isfile(safe_p):
      return f"Error: Path '{path}' is not a file."

    rel_path = os.path.relpath(safe_p, sandbox_dir)
    
    with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
      old_content = f.read()

    ext = os.path.splitext(safe_p)[1].lower()
    formatted_content = None
    formatter_used = ""

    import subprocess
    import shutil

    if ext == ".py":
      # Try black, ruff, yapf, autopep8
      if shutil.which("black"):
        subprocess.run(["black", "-q", safe_p], capture_output=True, text=True)
        formatter_used = "black"
      elif shutil.which("ruff"):
        subprocess.run(["ruff", "format", safe_p], capture_output=True, text=True)
        formatter_used = "ruff format"
      elif shutil.which("yapf"):
        subprocess.run(["yapf", "-i", safe_p], capture_output=True, text=True)
        formatter_used = "yapf"
      elif shutil.which("autopep8"):
        subprocess.run(["autopep8", "-i", safe_p], capture_output=True, text=True)
        formatter_used = "autopep8"
      else:
        return (
          f"Error: No Python formatter found (black, ruff, yapf, or autopep8). "
          f"Please run a command like 'pip install black' or 'pip install ruff' in the project environment."
        )
      
      # For formatters running in-place, read updated file
      with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
        formatted_content = f.read()

    elif ext == ".json":
      import json
      try:
        data = json.loads(old_content)
        formatted_content = json.dumps(data, indent=2) + "\n"
        formatter_used = "built-in json.dumps"
      except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"

    elif ext in (".yaml", ".yml"):
      import yaml
      try:
        # We can try prettier first, as it preserves comments
        if shutil.which("prettier"):
          subprocess.run(["prettier", "--write", safe_p], capture_output=True, text=True)
          formatter_used = "prettier"
          with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
            formatted_content = f.read()
        else:
          data = yaml.safe_load(old_content)
          formatted_content = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
          formatter_used = "built-in PyYAML (note: comments may have been stripped)"
      except Exception as e:
        return f"Error formatting YAML: {str(e)}"

    elif ext in (".c", ".cpp", ".h", ".hpp", ".cs", ".java"):
      if shutil.which("clang-format"):
        subprocess.run(["clang-format", "-i", safe_p], capture_output=True, text=True)
        formatter_used = "clang-format"
        with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
          formatted_content = f.read()
      else:
        return "Error: clang-format is not installed on the system."

    elif ext == ".go":
      if shutil.which("gofmt"):
        subprocess.run(["gofmt", "-w", safe_p], capture_output=True, text=True)
        formatter_used = "gofmt"
        with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
          formatted_content = f.read()
      else:
        return "Error: gofmt is not installed on the system."

    elif ext in (".rs", ".rlib"):
      if shutil.which("rustfmt"):
        subprocess.run(["rustfmt", safe_p], capture_output=True, text=True)
        formatter_used = "rustfmt"
        with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
          formatted_content = f.read()
      else:
        return "Error: rustfmt is not installed on the system."

    elif ext in (".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".md"):
      if shutil.which("prettier"):
        subprocess.run(["prettier", "--write", safe_p], capture_output=True, text=True)
        formatter_used = "prettier"
        with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
          formatted_content = f.read()
      else:
        return "Error: prettier is not installed on the system."

    else:
      return f"Error: No formatter configured for files with extension '{ext}'."

    # If formatted content was generated programmatically (JSON, YAML fallback), write it back
    if formatted_content is not None and formatted_content != old_content:
      with open(safe_p, 'w', encoding='utf-8') as f:
        f.write(formatted_content)

    if formatted_content is None:
      # If formatted_content was read back from disk after in-place formatting
      with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
        formatted_content = f.read()

    if formatted_content == old_content:
      return f"File '{rel_path}' is already formatted correctly."

    print_diff(rel_path, old_content, formatted_content)
    return f"Successfully formatted file '{rel_path}' using {formatter_used}."

  except Exception as e:
    return f"Error formatting file: {str(e)}"


# (Deleted tool_run_command. Replaced by ChatbotSession.tool_run_command)


# Define standard schemas for tools
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "format_file",
            "description": "Format a source code file using the appropriate formatter (e.g. black/ruff for Python, clang-format for C/C++, prettier for JS/TS/HTML/CSS/MD, or built-in json/yaml tools). Shows a diff of changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path relative to the sandbox root."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": "Replace a unique, specific block of text/code inside a file in the sandbox. Preserves the rest of the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path relative to the sandbox root."
                    },
                    "search": {
                        "type": "string",
                        "description": "The exact block of code/text to be replaced. Must match a unique occurrence in the file including whitespace and indentation."
                    },
                    "replace": {
                        "type": "string",
                        "description": "The new code/text to replace the search block with."
                    }
                },
                "required": ["path", "search", "replace"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_lines",
            "description": "Replace a range of lines in a file (1-indexed, inclusive) with new content. Immunity to search text matching errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path relative to the sandbox root."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "The starting line number to replace (1-indexed, inclusive)."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "The ending line number to replace (1-indexed, inclusive)."
                    },
                    "replacement": {
                        "type": "string",
                        "description": "The new text/code content to insert in place of the specified lines."
                    }
                },
                "required": ["path", "start_line", "end_line", "replacement"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory contents inside the sandboxed file system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list relative to the sandbox root. Defaults to '.' (root of sandbox)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text contents of a file inside the sandboxed file system, optionally restricted to a specific line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read relative to the sandbox root."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional starting line number to read (1-indexed, inclusive)."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional ending line number to read (1-indexed, inclusive)."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file inside the sandboxed file system. Restricts modifications to only inside the sandbox folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The target file path relative to the sandbox root."
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete content to write into the file."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_grep",
            "description": "Search recursively inside files in the sandbox directory for a regular expression pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "The relative directory path in the sandbox to start searching from. Defaults to '.'."
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command. The command will run with its working directory (cwd) set to the sandbox folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command to execute."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_background_command",
            "description": "Check the status and read the output of a command running in the background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The Task ID returned by run_command (e.g. 'task_1')."
                    }
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "locate_files",
            "description": "Locate files recursively inside the sandbox directory matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match (e.g., '*.py', '**/tests/*.json')."
                    },
                    "path": {
                        "type": "string",
                        "description": "The relative directory path to search from. Defaults to '.'."
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_info",
            "description": "Get metadata info (size, last modified, type) about a path inside the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The target path relative to the sandbox root."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the text content of a public URL and convert it to clean text (removes HTML tags/scripts/styles).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The absolute HTTP/HTTPS URL to fetch."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run tests for the project. Auto-detects standard test runners (pytest, npm test, cargo test, go test, ctest, make test, meson test) or executes a custom test command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Optional custom command to run tests (e.g. 'pytest tests/test_math.py', 'make test'). If omitted, the chatbot will attempt to auto-detect and run the project's test suite."
                    }
                }
            }
        }
    }
]

def execute_tool(name: str, arguments: Dict[str, Any], session: "ChatbotSession") -> str:
  """Executes the specified tool with arguments in the sandbox directory."""
  if name == "run_tests":
    return session.tool_run_tests(arguments.get("command"))
  elif name == "list_dir":
    return tool_list_dir(session.sandbox, arguments.get("path", "."))
  elif name == "read_file":
    path = arguments.get("path")
    if not path:
      return "Error: Missing parameter 'path'."
    try:
      start_line = int(arguments.get("start_line")) if arguments.get("start_line") is not None else None
      end_line = int(arguments.get("end_line")) if arguments.get("end_line") is not None else None
    except (ValueError, TypeError):
      return "Error: start_line and end_line must be valid integers."
    return tool_read_file(session.sandbox, path, start_line, end_line)
  elif name == "write_file":
    path = arguments.get("path")
    content = arguments.get("content")
    if not path or content is None:
      return "Error: Missing parameters 'path' and 'content'."
    return tool_write_file(session.sandbox, path, content)
  elif name == "patch_file":
    path = arguments.get("path")
    search = arguments.get("search")
    replace = arguments.get("replace")
    if not path or search is None or replace is None:
      return "Error: Missing parameters 'path', 'search', or 'replace'."
    return tool_patch_file(session.sandbox, path, search, replace)
  elif name == "edit_lines":
    path = arguments.get("path")
    try:
      start_line = int(arguments.get("start_line"))
      end_line = int(arguments.get("end_line"))
    except (ValueError, TypeError):
      return "Error: start_line and end_line must be valid integers."
    replacement = arguments.get("replacement")
    if not path or replacement is None:
      return "Error: Missing parameters 'path' or 'replacement'."
    return tool_edit_lines(session.sandbox, path, start_line, end_line, replacement)
  elif name == "format_file":
    path = arguments.get("path")
    if not path:
      return "Error: Missing parameter 'path'."
    return tool_format_file(session.sandbox, path)
  elif name == "search_grep":
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")
    if not pattern:
      return "Error: Missing parameter 'pattern'."
    return tool_search_grep(session.sandbox, pattern, path)
  elif name == "run_command":
    command = arguments.get("command")
    if not command:
      return "Error: Missing parameter 'command'."
    return session.tool_run_command(command)
  elif name == "check_background_command":
    task_id = arguments.get("task_id")
    if not task_id:
      return "Error: Missing parameter 'task_id'."
    return session.tool_check_background_command(task_id)
  elif name == "locate_files":
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")
    if not pattern:
      return "Error: Missing parameter 'pattern'."
    return tool_locate_files(session.sandbox, pattern, path)
  elif name == "get_file_info":
    path = arguments.get("path")
    if not path:
      return "Error: Missing parameter 'path'."
    return tool_get_file_info(session.sandbox, path)
  elif name == "fetch_url":
    url = arguments.get("url")
    if not url:
      return "Error: Missing parameter 'url'."
    return tool_fetch_url(url)
  else:
    return f"Error: Tool '{name}' is not recognized."

# --- Helper Utilities ---


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
  """
  Parses YAML frontmatter from markdown.
  Returns (metadata_dict, body_content).
  """
  metadata = {}
  body = content
  
  if content.startswith("---"):
    parts = content.split("---", 2)
    if len(parts) >= 3:
      yaml_content = parts[1]
      body = parts[2].strip()
      
      for line in yaml_content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
          continue
        if ":" in line:
          key, val = line.split(":", 1)
          key = key.strip()
          val = val.strip().strip('"').strip("'")
          
          if val.startswith('[') and val.endswith(']'):
            import ast
            try:
              val = ast.literal_eval(val)
            except Exception:
              pass
          metadata[key] = val
  return metadata, body


def load_system_prompt_from_file(file_path: str) -> str:
    """Loads custom system prompt from a YAML configuration or raw text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file '{file_path}' does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            import yaml
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                if "system_prompt" in data:
                    return str(data["system_prompt"])
                else:
                    raise ValueError("YAML dictionary configuration must contain a 'system_prompt' key.")
            elif isinstance(data, str):
                return data
            else:
                f.seek(0)
                return f.read()
        except Exception as e:
            try:
                f.seek(0)
                return f.read()
            except Exception:
                raise ValueError(f"Failed to read prompt file: {str(e)}")


def count_tokens(text: str) -> int:
    """Counts tokens in a text block using tiktoken, falling back to approximation."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback approximation (roughly 4 characters per token for English text)
        return len(text) // 4

def get_ollama_models(url: str) -> List[str]:
    """Queries local Ollama instance for installed models."""
    try:
        parsed = urllib.parse.urlparse(url)
        base_api_url = f"{parsed.scheme}://{parsed.netloc}/api/tags"
        response = requests.get(base_api_url, timeout=2)
        if response.status_code == 200:
            models_data = response.json()
            return [m["name"] for m in models_data.get("models", [])]
    except Exception:
        pass
    return []

# --- Chatbot Session Manager ---

class ChatbotSession:
    def __init__(self, provider: str, model: str, context_size: int, sandbox: str, api_key: str = None, url: str = None, max_loops: int = 20, system_prompt_override: str = None, prompt_mode: str = "replace", skills_paths: List[str] = None):
        self.provider = provider
        self.model = model
        self.context_size = context_size
        self.sandbox = os.path.abspath(sandbox)
        self.api_key = api_key
        self.url = url
        self.max_loops = max_loops
        self.background_commands = {}
        self.next_task_id = 1
        self.skills_paths = skills_paths or []
        
        # Internal state
        self.messages: List[Dict[str, Any]] = []
        default_prompt = (
            "You are a helpful assistant with local sandboxed file access and shell execution capabilities.\n"
            "You have tools for: listing directories (list_dir), locating files (locate_files), checking file info (get_file_info), reading files (read_file), writing files (write_file), patching files (patch_file), editing line ranges (edit_lines), searching regex patterns (search_grep), fetching web content (fetch_url), executing shell commands (run_command), and checking background tasks (check_background_command).\n"
            "All paths provided to the tools will resolve relative to the sandbox directory.\n"
            "You are strictly prohibited from writing files outside the sandbox folder.\n"
            "For editing existing files, you should use the edit_lines tool (if you know the line numbers) or the patch_file tool (if replacing a unique text block) instead of overwriting the entire file with write_file. The edit_lines tool is highly recommended as it is completely immune to text-matching failures.\n"
            "WARNING: When inspecting files using shell commands, avoid using `cat -A` or `cat -E`. These commands append a dollar sign (`$`) to the end of every line as a line-end marker. The `$` character is NOT part of the file. Do NOT include `$` in replacement text when writing or editing files.\n"
            "When running shell commands using run_command, if a command takes longer than 10 seconds, it will automatically transition to run in the background and return a 'Task ID'. You must NOT block. Instead, check its output later by calling check_background_command with the Task ID to get progress or final output. Perform other file tasks (read, patch, edit) while waiting.\n"
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
        
        # Initialize client
        self.init_client()
        
        # Load active skills
        self.skills = {}
        self.load_skills()

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

    def get_active_system_prompt(self) -> str:
      """Returns system prompt integrated with dynamically activated skills."""
      active_skills = []
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
          
      if active_skills:
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
                api_key=key
            )

    def tool_run_tests(self, command: str = None) -> str:
      """Run tests in the sandbox, auto-detecting the testing framework if no command is provided."""
      import os
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

    def tool_run_command(self, command: str) -> str:
      """Execute a shell command, transitioning to background execution if it takes too long."""
      import subprocess
      import os
      import tempfile
      stdout_f = None
      stderr_f = None
      try:
        stdout_f = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
        stderr_f = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
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
          stdout_f.seek(0)
          stderr_f.seek(0)
          stdout = stdout_f.read()
          stderr = stderr_f.read()
          stdout_f.close()
          stderr_f.close()
          try:
            os.unlink(stdout_f.name)
          except Exception:
            pass
          try:
            os.unlink(stderr_f.name)
          except Exception:
            pass
          output = []
          if stdout:
            output.append(f"Stdout:\n{stdout}")
          if stderr:
            output.append(f"Stderr:\n{stderr}")
          status = f"Command exited with code {proc.returncode}."
          return "\n".join(output) + f"\n{status}" if output else status
        except subprocess.TimeoutExpired:
          task_id = f"task_{self.next_task_id}"
          self.next_task_id += 1
          self.background_commands[task_id] = {
            "proc": proc,
            "command": command,
            "stdout_path": stdout_f.name,
            "stderr_path": stderr_f.name,
            "stdout_file": stdout_f,
            "stderr_file": stderr_f
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
        import os
        task = self.background_commands.get(task_id)
        if not task:
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
        output = []
        if stdout_content:
            output.append(f"Stdout:\n{stdout_content}")
        if stderr_content:
            output.append(f"Stderr:\n{stderr_content}")
        if status is None:
            return (
                f"Status: Task '{task_id}' is STILL RUNNING.\n"
                + ("\n".join(output) if output else "(No output generated yet)")
            )
        else:
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

    def cleanup_background_commands(self):
        """Kills all active background tasks and removes temporary files."""
        import os
        import signal
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

    def prune_history(self) -> List[Dict[str, Any]]:
      """Prunes conversation history to respect the configured context size."""
      sys_prompt = self.get_active_system_prompt()
      system_msg = {"role": "system", "content": sys_prompt}
      sys_tokens = count_tokens(sys_prompt)
      
      if sys_tokens >= self.context_size:
        return [system_msg]
        
      pruned = []
      accumulated_tokens = sys_tokens
      
      # Process from newest to oldest
      for msg in reversed(self.messages):
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

    def run_llm_cycle(self):
        """Executes a full inference cycle, resolving tool calls recursively."""
        self.load_skills()
        max_tool_loops = self.max_loops
        loop_count = 0
        
        while loop_count < max_tool_loops:
            # Prepare message payloads based on limit settings
            active_messages = self.prune_history()
            
            # Start LLM stream call
            tool_calls_accumulated = []
            content_accumulated = ""
            
            try:
                # Live rendering console helper
                with Live(Panel("Connecting to LLM...", title="Assistant", border_style="green"), 
                          refresh_per_second=12, console=console) as live:
                    
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=active_messages,
                        tools=TOOLS_SCHEMA,
                        stream=True
                    )
                    
                    first_chunk = True
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        
                        # Process streaming content
                        if delta.content:
                            if first_chunk:
                                live.update(Panel("", title="Assistant", border_style="green"))
                                first_chunk = False
                            content_accumulated += delta.content
                            live.update(Panel(Markdown(content_accumulated), title="Assistant", border_style="green"))
                            
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
                                live.update(Panel(f"Accumulating tool arguments... ({len(tool_calls_accumulated)} call(s))", 
                                                  title="System", border_style="yellow"))
                                
            except Exception as e:
                console.print(f"[bold red]Error calling API:[/bold red] {str(e)}")
                break
                
            # If we didn't receive structured tool calls, try to extract them from text content
            if not tool_calls_accumulated and content_accumulated:
                parsed_calls = self.extract_tool_calls_from_text(content_accumulated)
                if parsed_calls:
                    tool_calls_accumulated = parsed_calls
                    content_accumulated = ""
                    
            # Ensure every accumulated tool call has a unique ID
            import uuid
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
                    console.print(Panel(
                        f"Name: [cyan]{t_name}[/cyan]\nArguments: [yellow]{json.dumps(args_parsed, indent=2)}[/yellow]",
                        title="🔧 Executing Tool",
                        border_style="yellow"
                    ))
                    t_result = execute_tool(t_name, args_parsed, self)
                    
                # Print result summary nicely
                console.print(Panel(
                    t_result,
                    title="🔧 Tool Result",
                    border_style="dim yellow"
                ))
                
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

    # --- Slash Commands Handling ---

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
            
        elif cmd == "/help":
            self.show_help()
            
        elif cmd == "/status":
            self.show_status()
            
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

    def show_help(self):
        """Displays formatted CLI usage guide."""
        table = Table(title="Slash Commands", show_header=True, header_style="bold magenta")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")
        table.add_row("/help", "Show this help screen")
        table.add_row("/status", "Display current session configuration")
        table.add_row("/provider [ollama|openrouter]", "View or switch the LLM backend provider")
        table.add_row("/model [name]", "View or switch the current LLM model")
        table.add_row("/sandbox [path]", "View or change the sandbox directory path")
        table.add_row("/context [tokens]", "View or modify the history token limit")
        table.add_row("/loops [iterations]", "View or modify the max sequential tool loops limit")
        table.add_row("/api_key [key]", "Configure the OpenRouter API Key")
        table.add_row("/system [text]", "View or edit the system instructions")
        table.add_row("/multiline", "Toggle multiline prompt input (Alt+Enter to send)")
        table.add_row("/history", "View message records and sizing details")
        table.add_row("/tools", "List available sandbox tools and schemas")
        table.add_row("/clear / /reset", "Clear conversation memory")
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

    def show_tools(self):
        """Lists available filesystem functions."""
        table = Table(title="Available Sandboxed Tools", show_header=True, header_style="bold yellow")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Description", style="white")
        for tool in TOOLS_SCHEMA:
            func = tool["function"]
            table.add_row(func["name"], func["description"])
        console.print(table)

    # --- CLI Input Loop ---

    def start_loop(self):
        """Runs the interactive input/output CLI loop."""
        # Create keybindings for multiline submissions
        kb = KeyBindings()
        
        @kb.add('escape', 'enter')
        def _(event):
            event.current_buffer.validate_and_handle()
            
        # File history tracking
        history_file = os.path.expanduser("~/.agent_chat_history")
        session = PromptSession(history=FileHistory(history_file), key_bindings=kb)
        
        # Display starting banner
        console.print(Panel(
            "[bold green]Welcome to the Sandboxed AI Chatbot CLI![/bold green]\n"
            "This script interfaces with Ollama and OpenRouter and restricts file write operations to the sandbox.\n"
            "Type [cyan]/help[/cyan] to display slash commands.\n"
            "Press [cyan]Ctrl+D[/cyan] or type [cyan]/exit[/cyan] to exit.",
            title="Antigravity Sandboxed Chatbot",
            border_style="magenta"
        ))
        
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
                    multiline=self.multiline_mode
                )
                
                # Check for empty input
                if not user_input.strip():
                    continue
                    
                # Check for slash commands
                if user_input.strip().startswith("/"):
                    should_continue = self.handle_command(user_input)
                    if not should_continue:
                        break
                    continue
                    
                # Append user query to history and execute loop
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
                console.print(f"[bold red]Unexpected error in CLI loop:[/bold red] {str(e)}")

# --- CLI Main Entrance ---

def main():
    parser = argparse.ArgumentParser(
        description="AI Chatbot CLI with advanced sandboxed text and file system interaction."
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["ollama", "openrouter"],
        default="ollama",
        help="The LLM backend provider to use (default: ollama)."
    )
    parser.add_argument(
        "--model", "-m",
        help="Model identifier to use. If omitted, default models will be resolved based on provider."
    )
    parser.add_argument(
        "--context-size", "-c",
        type=int,
        default=8192,
        help="Target context window length constraint in tokens (default: 8192)."
    )
    parser.add_argument(
        "--sandbox", "-s",
        default="./sandbox",
        help="Path to the sandboxed file system directory. Writes are strictly restricted here (default: ./sandbox)."
    )
    parser.add_argument(
        "--skills-path", "-k",
        action="append",
        default=[],
        help="Custom directories to search for skills. Can be specified multiple times."
    )
    parser.add_argument(
        "--max-loops", "-l",
        type=int,
        default=20,
        help="Maximum sequential tool execution loops allowed in a single turn (default: 20)."
    )
    parser.add_argument(
        "--config-prompt", "-f",
        help="Path to a YAML or text configuration file containing the custom system prompt."
    )
    parser.add_argument(
        "--prompt-mode", "-d",
        choices=["replace", "integrate"],
        default="replace",
        help="How to apply the custom system prompt file (replace default prompt, or integrate/append to it)."
    )
    parser.add_argument(
        "--api-key", "-a",
        help="OpenRouter API key. Overrides the OPENROUTER_API_KEY environment variable."
    )
    parser.add_argument(
        "--url", "-u",
        help="API Base URL override (defaults to Ollama local endpoint or OpenRouter base URL)."
    )
    
    args = parser.parse_args()
    
    # Load system prompt from file if specified
    custom_system_prompt = None
    if args.config_prompt:
        try:
            custom_system_prompt = load_system_prompt_from_file(args.config_prompt)
            console.print(f"[bold blue]Info:[/bold blue] Loaded custom system prompt from '{args.config_prompt}' (mode: {args.prompt_mode}).")
        except Exception as e:
            console.print(f"[bold red]Error loading prompt configuration:[/bold red] {e}")
            sys.exit(1)
            
    # Resolve default models
    model = args.model
    if not model:
        if args.provider == "ollama":
            # Attempt to auto-detect model from local Ollama tags
            ollama_url = args.url or "http://localhost:11434/v1"
            local_models = get_ollama_models(ollama_url)
            if local_models:
                # Pick the first matching model
                model = local_models[0]
                console.print(f"[bold blue]Info:[/bold blue] Auto-detected local Ollama model: [bold green]{model}[/bold green]")
            else:
                model = "qwen2.5-coder:7b"
                console.print(f"[bold blue]Info:[/bold blue] No local Ollama models detected. Fallback default: [bold green]{model}[/bold green]")
        else:
            model = "google/gemini-2.5-flash"
            console.print(f"[bold blue]Info:[/bold blue] OpenRouter provider selected. Default model: [bold green]{model}[/bold green]")
            
    # Initialize and execute chat session
    chat_session = ChatbotSession(
        provider=args.provider,
        model=model,
        context_size=args.context_size,
        sandbox=args.sandbox,
        api_key=args.api_key,
        url=args.url,
        max_loops=args.max_loops,
        system_prompt_override=custom_system_prompt,
        prompt_mode=args.prompt_mode,
        skills_paths=args.skills_path
    )
    
    chat_session.start_loop()

if __name__ == "__main__":
    main()
