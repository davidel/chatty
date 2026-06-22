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
import logging
import datetime
from typing import List, Dict, Any, Tuple, Optional

# Initialize module logger
logger = logging.getLogger("chatty")

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

def tool_list_dir(sandbox_dir: str, path: str = ".", max_items: int = 200) -> str:
    """List the contents of a directory path inside the sandbox."""
    try:
        safe_p = get_safe_path(sandbox_dir, path)
        if not os.path.exists(safe_p):
            return f"Error: Path '{path}' does not exist."
        if not os.path.isdir(safe_p):
            return f"Error: Path '{path}' is not a directory."
            
        items = os.listdir(safe_p)
        sorted_items = sorted(items)
        truncated = False
        if len(sorted_items) > max_items:
            sorted_items = sorted_items[:max_items]
            truncated = True
            
        result = []
        for item in sorted_items:
            full_path = os.path.join(safe_p, item)
            rel_path = os.path.relpath(full_path, sandbox_dir)
            if os.path.isdir(full_path):
                result.append(f"[DIR]  {rel_path}/")
            else:
                size = os.path.getsize(full_path)
                result.append(f"[FILE] {rel_path} ({size} bytes)")
        output_str = "\n".join(result) if result else "(Empty directory)"
        if truncated:
            output_str += f"\n\n[WARNING: Directory listing truncated. Showing first {max_items} of {len(items)} items.]"
        return output_str
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def tool_read_file(sandbox_dir: str, path: str, start_line: int = None, end_line: int = None, max_chars: int = 40000) -> str:
  """Read the contents of a file inside the sandbox, optionally specifying a 1-indexed line range."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: File '{path}' does not exist."
    if not os.path.isfile(safe_p):
      return f"Error: Path '{path}' is not a file."
      
    with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
      if start_line is None and end_line is None:
        content = f.read()
        if len(content) > max_chars:
          return content[:max_chars] + f"\n\n[WARNING: File '{path}' is too large ({len(content)} characters) and has been truncated. Use 'start_line' and 'end_line' parameters to read specific sections.]"
        return content
        
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


def tool_get_file_info(sandbox_dir: str, path: str) -> str:
  """Get metadata info (size, last modified, type, and line count for text files) about a path."""
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
    if not is_dir and is_text_file(safe_p):
      info.append(f"Lines: {count_lines(safe_p)}")
      
    return "\n".join(info)
  except Exception as e:
    return f"Error getting file info: {str(e)}"


def tool_fetch_url(url: str, max_chars: int = 24000) -> str:
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
    full_text = "\n".join(cleaned_lines).strip()
    if len(full_text) > max_chars:
      return full_text[:max_chars] + f"\n\n[WARNING: URL content truncated. Total length: {len(full_text)} characters.]"
    return full_text
  except Exception as e:
    return f"Error fetching URL: {str(e)}"


def validate_file_syntax(path: str, content: str, sandbox_dir: str = None, compile_paths: List[str] = None) -> Tuple[bool, str]:
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
        compiler = "clang++" if shutil.which("clang++") else "g++" if is_cpp else "clang" if shutil.which("clang") else "gcc"
          
        cmd_args = [compiler, "-fsyntax-only"]
        target_dir = os.path.dirname(path) or "."
        cmd_args.extend(["-I", target_dir])
        if sandbox_dir:
          for p in (compile_paths or []):
            abs_p = os.path.abspath(os.path.join(sandbox_dir, p))
            if os.path.isdir(abs_p):
              cmd_args.extend(["-I", abs_p])
            elif os.path.isfile(abs_p):
              cmd_args.extend(["-I", os.path.dirname(abs_p)])
        cmd_args.append(temp_name)
          
        proc = subprocess.run(
          cmd_args,
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
        dir_name = os.path.dirname(path) or "."
        search_dirs = {dir_name}
        extra_files = []
        if sandbox_dir:
          try:
            for root, dirs, files in os.walk(sandbox_dir):
              if any(f.endswith((".v", ".sv", ".vh", ".svh")) for f in files):
                search_dirs.add(root)
          except Exception:
            pass

          for p in (compile_paths or []):
            abs_p = os.path.abspath(os.path.join(sandbox_dir, p))
            if os.path.isdir(abs_p):
              search_dirs.add(abs_p)
            elif os.path.isfile(abs_p):
              extra_files.append(abs_p)
              search_dirs.add(os.path.dirname(abs_p))

        if shutil.which("verilator"):
          cmd_args = ["verilator", "--lint-only", "-Wno-fatal", "-Wno-MODMISSING"]
          for s_dir in sorted(search_dirs):
            cmd_args.extend(["-y", s_dir, f"-I{s_dir}"])
          cmd_args.extend(extra_files)
          cmd_args.append(temp_name)
          proc = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
          )
          if proc.returncode != 0:
            err_msg = (proc.stderr + proc.stdout).replace(temp_name, os.path.basename(path))
            return False, (
              f"Verilator Lint Error:\n{err_msg}\n"
              "Note: If Verilator cannot resolve dependencies, make sure the required module or include files are present. "
            )
        elif shutil.which("iverilog"):
          cmd_args = ["iverilog", "-g2012", "-t", "null"]
          for s_dir in sorted(search_dirs):
            cmd_args.extend(["-y", s_dir, f"-I{s_dir}"])
          cmd_args.extend(extra_files)
          cmd_args.append(temp_name)
          proc = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
          )
          if proc.returncode != 0:
            err_msg = (proc.stderr + proc.stdout).replace(temp_name, os.path.basename(path))
            return False, (
              f"Icarus Verilog Syntax Error:\n{err_msg}\n"
            )
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
          cmd_args = ["ghdl", "-s"]
          if sandbox_dir:
            for p in (compile_paths or []):
              abs_p = os.path.abspath(os.path.join(sandbox_dir, p))
              if os.path.isdir(abs_p):
                cmd_args.append(f"-P{abs_p}")
              elif os.path.isfile(abs_p):
                cmd_args.append(f"-P{os.path.dirname(abs_p)}")
          cmd_args.append(temp_name)
          proc = subprocess.run(
            cmd_args,
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


def tool_write_file(sandbox_dir: str, path: str, content: str, compile_paths: List[str] = None) -> str:
  """Write text content to a file inside the sandbox."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    is_valid, err_msg = validate_file_syntax(safe_p, content, sandbox_dir, compile_paths)
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

def tool_search_grep(sandbox_dir: str, pattern: str, path: str = ".", max_results: int = 100, line_numbers: bool = False) -> str:
  """Search for a regex pattern inside files in the sandbox, ignoring files in .gitignore and common cache directories."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: Path '{path}' does not exist."
      
    regex = re.compile(pattern, re.IGNORECASE)
    results = []
    ignore_patterns = load_ignore_patterns(sandbox_dir)
    
    if os.path.isfile(safe_p):
      rel_path = os.path.relpath(safe_p, sandbox_dir)
      if not is_path_ignored(rel_path, ignore_patterns):
        with open(safe_p, 'r', encoding='utf-8', errors='ignore') as f:
          for line_num, line in enumerate(f, 1):
            if regex.search(line):
              if line_numbers:
                results.append(f"{rel_path}:{line_num}: {line.strip()}")
              else:
                results.append(f"{rel_path}: {line.strip()}")
              if len(results) >= max_results:
                break
    else:
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
                  if line_numbers:
                    results.append(f"{rel_path}:{line_num}: {line.strip()}")
                  else:
                    results.append(f"{rel_path}: {line.strip()}")
                  if len(results) >= max_results:
                    break
          except Exception:
            continue
          if len(results) >= max_results:
            break
        if len(results) >= max_results:
          break
          
    if len(results) >= max_results:
      return "\n".join(results) + f"\n\n[WARNING: Search results truncated to {max_results} matches. Please refine your regex pattern to filter more specifically.]"
    return "\n".join(results) if results else "No matches found."
  except Exception as e:
    return f"Error searching files: {str(e)}"


def tool_patch_file(sandbox_dir: str, path: str, search: str, replace: str, compile_paths: List[str] = None) -> str:
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
      
    is_valid, err_msg = validate_file_syntax(safe_p, new_content, sandbox_dir, compile_paths)
    if not is_valid:
      return f"Error: Syntax verification failed for updated file content. Patch was not applied.\n{err_msg}"
      
    with open(safe_p, 'w', encoding='utf-8') as f:
      f.write(new_content)
      
    rel_path = os.path.relpath(safe_p, sandbox_dir)
    print_diff(rel_path, content, new_content)
    return f"Successfully updated file '{rel_path}' using a target replacement patch."
  except Exception as e:
    return f"Error patching file: {str(e)}"


def tool_edit_lines(sandbox_dir: str, path: str, start_line: int, end_line: int, replacement: str, compile_paths: List[str] = None) -> str:
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
    is_valid, err_msg = validate_file_syntax(safe_p, new_content, sandbox_dir, compile_paths)
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

def tool_move_file(sandbox_dir: str, src: str, dest: str) -> str:
  """Move or rename a file or directory inside the sandbox."""
  try:
    safe_src = get_safe_path(sandbox_dir, src)
    safe_dest = get_safe_path(sandbox_dir, dest)
    
    if not os.path.exists(safe_src):
      return f"Error: Source path '{src}' does not exist."
      
    dest_parent = os.path.dirname(safe_dest)
    if not os.path.exists(dest_parent):
      os.makedirs(dest_parent, exist_ok=True)
      
    import shutil
    shutil.move(safe_src, safe_dest)
    
    rel_src = os.path.relpath(safe_src, sandbox_dir)
    rel_dest = os.path.relpath(safe_dest, sandbox_dir)
    return f"Successfully moved '{rel_src}' to '{rel_dest}'."
  except Exception as e:
    return f"Error moving file/directory: {str(e)}"


def tool_copy_file(sandbox_dir: str, src: str, dest: str) -> str:
  """Copy a file or directory inside the sandbox."""
  try:
    safe_src = get_safe_path(sandbox_dir, src)
    safe_dest = get_safe_path(sandbox_dir, dest)
    
    if not os.path.exists(safe_src):
      return f"Error: Source path '{src}' does not exist."
      
    dest_parent = os.path.dirname(safe_dest)
    if not os.path.exists(dest_parent):
      os.makedirs(dest_parent, exist_ok=True)
      
    import shutil
    if os.path.isdir(safe_src):
      shutil.copytree(safe_src, safe_dest, dirs_exist_ok=True)
    else:
      shutil.copy2(safe_src, safe_dest)
      
    rel_src = os.path.relpath(safe_src, sandbox_dir)
    rel_dest = os.path.relpath(safe_dest, sandbox_dir)
    return f"Successfully copied '{rel_src}' to '{rel_dest}'."
  except Exception as e:
    return f"Error copying file/directory: {str(e)}"


def tool_delete_file(sandbox_dir: str, path: str) -> str:
  """Delete a file or directory inside the sandbox."""
  try:
    safe_path = get_safe_path(sandbox_dir, path)
    
    if not os.path.exists(safe_path):
      return f"Error: Path '{path}' does not exist."
      
    rel_path = os.path.relpath(safe_path, sandbox_dir)
    import shutil
    if os.path.isdir(safe_path):
      shutil.rmtree(safe_path)
      return f"Successfully deleted directory '{rel_path}'."
    else:
      os.remove(safe_path)
      return f"Successfully deleted file '{rel_path}'."
  except Exception as e:
    return f"Error deleting file/directory: {str(e)}"


def tool_make_directory(sandbox_dir: str, path: str) -> str:
  """Create a new directory (and any parent directories) inside the sandbox."""
  try:
    safe_path = get_safe_path(sandbox_dir, path)
    
    if os.path.exists(safe_path):
      if os.path.isdir(safe_path):
        return f"Directory '{path}' already exists."
      else:
        return f"Error: Path '{path}' already exists but is a file."
        
    os.makedirs(safe_path, exist_ok=True)
    rel_path = os.path.relpath(safe_path, sandbox_dir)
    return f"Successfully created directory '{rel_path}'."
  except Exception as e:
    return f"Error creating directory: {str(e)}"


# Define standard schemas for tools
TOOLS_SCHEMA = [
  {
    "type": "function",
    "function": {
      "name": "move_file",
      "description": "Move or rename a file or directory inside the sandboxed file system.",
      "parameters": {
        "type": "object",
        "properties": {
          "src": {
            "type": "string",
            "description": "The source file or directory path relative to the sandbox root."
          },
          "dest": {
            "type": "string",
            "description": "The destination file or directory path relative to the sandbox root."
          }
        },
        "required": ["src", "dest"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "copy_file",
      "description": "Copy a file or directory inside the sandboxed file system.",
      "parameters": {
        "type": "object",
        "properties": {
          "src": {
            "type": "string",
            "description": "The source file or directory path relative to the sandbox root."
          },
          "dest": {
            "type": "string",
            "description": "The destination file or directory path relative to the sandbox root."
          }
        },
        "required": ["src", "dest"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "delete_file",
      "description": "Delete a file or directory inside the sandboxed file system.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The path to the file or directory to delete relative to the sandbox root."
          }
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "make_directory",
      "description": "Create a new directory (and any parent directories recursively) inside the sandboxed file system.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The directory path to create relative to the sandbox root."
          }
        },
        "required": ["path"]
      }
    }
  },
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
            "description": "Replace a unique, specific block of text/code inside a file in the sandbox. Preserves the rest of the file. Use this for editing existing files when you can match a unique block of text. For editing where matching is difficult, use 'edit_lines'.",
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
                    },
                    "compile_paths": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional list of files or directories (relative to sandbox) containing compile-time dependencies, library search paths, or include paths needed for syntax verification."
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
            "description": "Replace a range of lines in a file (1-indexed, inclusive) with new content. This tool is highly recommended for editing existing files because it uses line numbers (which you can get from 'search_grep' or 'read_file') and is completely immune to text-matching failures.",
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
                    },
                    "compile_paths": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional list of files or directories (relative to sandbox) containing compile-time dependencies, library search paths, or include paths needed for syntax verification."
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
            "description": "Read the text contents of a file inside the sandboxed file system, optionally restricted to a specific line range. Use this tool instead of shell commands like 'cat', 'head', 'tail', 'sed', 'awk', 'less', or 'more' via run_command.",
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
            "description": "Write text content to a file inside the sandboxed file system. Restricts modifications to only inside the sandbox folder. WARNING: For editing existing files, you should use 'edit_lines' or 'patch_file' instead of overwriting the entire file.",
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
                    },
                    "compile_paths": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional list of files or directories (relative to sandbox) containing compile-time dependencies, library search paths, or include paths needed for syntax verification."
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
            "description": "Search for a regular expression pattern inside files in the sandbox directory (recursively) or inside a specific file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The python-compatible regular expression pattern to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "The relative directory or file path in the sandbox to start searching from. Defaults to '.'."
                    },
                    "line_numbers": {
                        "type": "boolean",
                        "description": "Set to true to include line numbers in the search results (formatted as 'file_path:line_number: content'). Set this to true if you plan to edit the matches later (e.g. using edit_lines or patch_file). Defaults to false."
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
            "description": "Execute a shell command. The command will run with its working directory (cwd) set to the sandbox folder. WARNING: You are strictly prohibited from using this tool to search files (use search_grep), find files (use locate_files), or view/inspect files (use read_file/get_file_info). Using commands like 'grep', 'find', 'cat', 'head', 'tail', 'sed', 'awk', 'less', or 'more' to search or view files directly will fail with an error.",
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
            "description": "Get metadata info (size, last modified, type, and line count for text files) about a path inside the sandbox.",
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
            "description": "Run tests, linting, or verification suites for the project (e.g., pytest, verilator, iverilog, npm test, cargo test, go test, ctest, make test, meson test) or executes a custom test command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Optional custom command to run tests/linting (e.g., 'pytest tests/test_math.py', 'make test', or 'verilator --lint-only -y src -Isrc/include src/top.sv'). Specify any required include paths, library search paths, or source files directly in the command string."
                    }
                }
            }
        }
    }
]

def execute_tool(name: str, arguments: Dict[str, Any], session: "ChatbotSession") -> str:
  """Executes the specified tool with arguments in the sandbox directory."""
  if name == "move_file":
    src = arguments.get("src")
    dest = arguments.get("dest")
    if not src or not dest:
      return "Error: Missing parameters 'src' and/or 'dest'."
    return tool_move_file(session.sandbox, src, dest)
  elif name == "copy_file":
    src = arguments.get("src")
    dest = arguments.get("dest")
    if not src or not dest:
      return "Error: Missing parameters 'src' and/or 'dest'."
    return tool_copy_file(session.sandbox, src, dest)
  elif name == "delete_file":
    path = arguments.get("path")
    if not path:
      return "Error: Missing parameter 'path'."
    return tool_delete_file(session.sandbox, path)
  elif name == "make_directory":
    path = arguments.get("path")
    if not path:
      return "Error: Missing parameter 'path'."
    return tool_make_directory(session.sandbox, path)
  elif name == "run_tests":
    return session.tool_run_tests(arguments.get("command"))
  elif name == "list_dir":
    return tool_list_dir(session.sandbox, arguments.get("path", "."), max_items=session.max_dir_items)
  elif name == "read_file":
    path = arguments.get("path")
    if not path:
      return "Error: Missing parameter 'path'."
    try:
      start_line = int(arguments.get("start_line")) if arguments.get("start_line") is not None else None
      end_line = int(arguments.get("end_line")) if arguments.get("end_line") is not None else None
    except (ValueError, TypeError):
      return "Error: start_line and end_line must be valid integers."
    return tool_read_file(session.sandbox, path, start_line, end_line, max_chars=session.max_read_chars)
  elif name == "write_file":
    path = arguments.get("path")
    content = arguments.get("content")
    compile_paths = arguments.get("compile_paths")
    if not path or content is None:
      return "Error: Missing parameters 'path' and 'content'."
    return tool_write_file(session.sandbox, path, content, compile_paths)
  elif name == "patch_file":
    path = arguments.get("path")
    search = arguments.get("search")
    replace = arguments.get("replace")
    compile_paths = arguments.get("compile_paths")
    if not path or search is None or replace is None:
      return "Error: Missing parameters 'path', 'search', or 'replace'."
    return tool_patch_file(session.sandbox, path, search, replace, compile_paths)
  elif name == "edit_lines":
    path = arguments.get("path")
    compile_paths = arguments.get("compile_paths")
    try:
      start_line = int(arguments.get("start_line"))
      end_line = int(arguments.get("end_line"))
    except (ValueError, TypeError):
      return "Error: start_line and end_line must be valid integers."
    replacement = arguments.get("replacement")
    if not path or replacement is None:
      return "Error: Missing parameters 'path' or 'replacement'."
    return tool_edit_lines(session.sandbox, path, start_line, end_line, replacement, compile_paths)
  elif name == "format_file":
    path = arguments.get("path")
    if not path:
      return "Error: Missing parameter 'path'."
    return tool_format_file(session.sandbox, path)
  elif name == "search_grep":
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")
    line_numbers = arguments.get("line_numbers", False)
    if not pattern:
      return "Error: Missing parameter 'pattern'."
    return tool_search_grep(session.sandbox, pattern, path, max_results=session.max_grep_results, line_numbers=line_numbers)
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
    return tool_fetch_url(url, max_chars=session.max_url_chars)
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


def truncate_output(text: str, max_chars: int = 16000) -> str:
  """Truncates output in the middle if it exceeds max_chars."""
  if len(text) <= max_chars:
    return text
  half = max_chars // 2
  truncated_chars = len(text) - max_chars
  return f"{text[:half]}\n\n... [TRUNCATED {truncated_chars} CHARACTERS OF OUTPUT] ...\n\n{text[-half:]}"

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
    def __init__(self, provider: str, model: str, context_size: int, sandbox: str, api_key: str = None, url: str = None, max_loops: int = 20, system_prompt_override: str = None, prompt_mode: str = "replace", skills_paths: List[str] = None, max_read_chars: int = 40000, max_grep_results: int = 100, max_command_chars: int = 16000, max_history_tool_chars: int = 1000, history_keep_messages: int = 4, max_url_chars: int = 24000, max_dir_items: int = 200):
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
        default_prompt = (
            "You are a helpful assistant with local sandboxed file access and shell execution capabilities.\n"
            "You have tools for: listing directories (list_dir), locating files (locate_files), checking file info (get_file_info), reading files (read_file), writing files (write_file), patching files (patch_file), editing line ranges (edit_lines), searching regex patterns (search_grep), fetching web content (fetch_url), executing shell commands (run_command), and checking background tasks (check_background_command).\n"
            "All paths provided to the tools will resolve relative to the sandbox directory.\n"
            "You are strictly prohibited from writing files outside the sandbox folder.\n"
            "CRITICAL: You MUST use the dedicated, high-level filesystem tools (like read_file, search_grep, locate_files) instead of running command-line utilities (like grep, find, cat, head, tail, sed, awk, less, more) inside run_command. Shell execution using run_command is blocked for these actions and will return an error.\n"
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
                  "X-Title": "Chatty",
                }
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

    def validate_command_safety(self, command: str) -> Optional[str]:
      """Validates that the shell command does not bypass dedicated tools."""
      import shlex
      import re
      import os
      
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

    def tool_run_command(self, command: str) -> str:
      """Execute a shell command, transitioning to background execution if it takes too long."""
      logger.info(f"Running shell command: '{command}'")
      validation_err = self.validate_command_safety(command)
      if validation_err:
        logger.warning(f"Rejected command '{command}': {validation_err}")
        return validation_err
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

    def cleanup_background_commands(self):
        """Kills all active background tasks and removes temporary files."""
        import os
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

    def prune_history(self) -> List[Dict[str, Any]]:
      """Prunes conversation history to respect the configured context size, compressing older tool outputs."""
      sys_prompt = self.get_active_system_prompt()
      system_msg = {"role": "system", "content": sys_prompt}
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
        
      logger.info(f"Pruning history: kept {len(pruned)} out of {total_msgs} messages (accumulated tokens: {accumulated_tokens})")
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
        logger.info(f"Starting LLM cycle. Max sequential tool loops: {max_tool_loops}")
        
        while loop_count < max_tool_loops:
            # Prepare message payloads based on limit settings
            active_messages = self.prune_history()
            
            # Start LLM stream call
            tool_calls_accumulated = []
            content_accumulated = ""
            
            logger.info(f"Loop {loop_count + 1}/{max_tool_loops}: Sending request to LLM (model={self.model}) with {len(active_messages)} messages")
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
                logger.info(f"LLM call succeeded. Content size: {len(content_accumulated)} chars, Tool calls count: {len(tool_calls_accumulated)}")
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
                    logger.info(f"Executing tool {t_name} (id={t_id}) with arguments: {args_parsed}")
                    t_result = execute_tool(t_name, args_parsed, self)
                    
                # Print result summary nicely
                console.print(Panel(
                    t_result,
                    title="🔧 Tool Result",
                    border_style="dim yellow"
                ))
                
                truncated = "TRUNCATED" in t_result or "truncated" in t_result.lower() or "WARNING" in t_result
                logger.info(f"Tool {t_name} (id={t_id}) completed. Result size: {len(t_result)} characters (truncated: {truncated})")
                
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
            
        elif cmd == "/compress":
          self.compress_context()
          
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


class GlogFormatter(logging.Formatter):
  """A logging formatter that formats messages like Google's glog,
  but using process ID instead of thread ID.
  Format: Lyyyymmdd hh:mm:ss.uuuuuu process file:line] msg
  """

  def format(self, record):
    level_char = "I"
    if record.levelname:
      if record.levelname == "CRITICAL":
        level_char = "F"
      elif record.levelname in ("DEBUG", "INFO", "WARNING", "ERROR"):
        level_char = record.levelname[0]
      else:
        level_char = record.levelname[0]
    dt = datetime.datetime.fromtimestamp(record.created)
    time_str = dt.strftime("%Y%m%d %H:%M:%S.%f")
    pid = record.process
    filename = record.filename
    lineno = record.lineno
    record.message = record.getMessage()
    prefix = f"{level_char}{time_str} {pid} {filename}:{lineno}]"
    s = f"{prefix} {record.message}"
    if record.exc_info:
      if not record.exc_text:
        record.exc_text = self.formatException(record.exc_info)
    if record.exc_text:
      if s[-1:] != "\n":
        s = s + "\n"
      s = s + record.exc_text
    if record.stack_info:
      if s[-1:] != "\n":
        s = s + "\n"
      s = s + self.formatStack(record.stack_info)
    return s


def setup_logging(log_file: str, log_level_str: str) -> None:
  """Configure logging with the custom GlogFormatter."""
  log_level = getattr(logging, log_level_str.upper(), logging.INFO)
  handler = logging.FileHandler(log_file, encoding="utf-8")
  handler.setFormatter(GlogFormatter())
  logging.basicConfig(
    level=log_level,
    handlers=[handler]
  )


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
    parser.add_argument(
        "--max-read-chars",
        type=int,
        default=40000,
        help="Max characters to read from a file during full read tool execution (default: 40000)."
    )
    parser.add_argument(
        "--max-grep-results",
        type=int,
        default=100,
        help="Max results returned by regex search tool (default: 100)."
    )
    parser.add_argument(
        "--max-command-chars",
        type=int,
        default=16000,
        help="Max characters returned from standard output/error of a shell command (default: 16000)."
    )
    parser.add_argument(
        "--max-history-tool-chars",
        type=int,
        default=1000,
        help="Max characters to keep in historical tool outputs before compression (default: 1000)."
    )
    parser.add_argument(
        "--history-keep-messages",
        type=int,
        default=4,
        help="Number of recent messages to keep fully uncompressed (default: 4)."
    )
    parser.add_argument(
        "--max-url-chars",
        type=int,
        default=24000,
        help="Max characters returned from fetched URLs (default: 24000)."
    )
    parser.add_argument(
        "--max-dir-items",
        type=int,
        default=200,
        help="Max items listed by the directory list tool (default: 200)."
    )
    parser.add_argument(
        "--log-file",
        default="chatty.log",
        help="Path to the file where operations will be logged (default: chatty.log). Set to empty string to disable logging."
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)."
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    if args.log_file:
        setup_logging(args.log_file, args.log_level)
        logger.info("==========================================")
        logger.info(f"Logging initialized to '{args.log_file}' (level: {args.log_level}).")
    
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
        skills_paths=args.skills_path,
        max_read_chars=args.max_read_chars,
        max_grep_results=args.max_grep_results,
        max_command_chars=args.max_command_chars,
        max_history_tool_chars=args.max_history_tool_chars,
        history_keep_messages=args.history_keep_messages,
        max_url_chars=args.max_url_chars,
        max_dir_items=args.max_dir_items
    )
    
    chat_session.start_loop()

if __name__ == "__main__":
    main()
