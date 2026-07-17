import os
import re
import fnmatch
from typing import List

from chatty.safety import (
  get_safe_path,
  load_ignore_patterns,
  is_path_ignored,
  is_text_file
)


def tool_locate_files(sandbox_dir: str, pattern: str, path: str = ".") -> str:
  """Locate files recursively inside the sandbox directory matching a glob pattern, ignoring files in .gitignore and common cache directories."""
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
          if is_path_ignored(rel_dir, ignore_patterns, is_dir=True):
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


def tool_search_grep(sandbox_dir: str, pattern: str, path: str = ".", max_results: int = 100, line_numbers: bool = False) -> str:
  """Search for a regex pattern inside files in the sandbox, ignoring binary files, files in .gitignore, and common cache directories."""
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
        if not is_text_file(safe_p):
          return f"Error: File '{path}' is a binary file; searching is skipped."
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
            if is_path_ignored(rel_dir, ignore_patterns, is_dir=True):
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
            if not is_text_file(safe_file_path):
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
