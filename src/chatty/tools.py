import os
import re
import json
import time
import shutil
import subprocess
from typing import List, Dict, Any, Tuple, Optional

from chatty.safety import (
  get_safe_path,
  load_ignore_patterns,
  is_path_ignored,
  is_text_file,
  count_lines
)
from chatty.utils import (
  record_command_binaries,
  print_diff,
  tool_fetch_url
)


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


def tool_read_file(sandbox_dir: str, path: str, start_line: int = None, end_line: int = None, max_chars: int = 40000, line_numbers: bool = False) -> str:
  """Read the contents of a file inside the sandbox, optionally specifying a 1-indexed line range."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: File '{path}' does not exist."
    if not os.path.isfile(safe_p):
      return f"Error: Path '{path}' is not a file."
      
    with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
      if start_line is None and end_line is None and not line_numbers:
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
        
      selected_lines = lines[s-1:e]
      if line_numbers:
        content = "".join(f"{s + idx}: {line}" for idx, line in enumerate(selected_lines))
      else:
        content = "".join(selected_lines)
        
      if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[WARNING: File '{path}' section is too large and has been truncated.]"
      return content
  except Exception as e:
    return f"Error reading file: {str(e)}"


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
  """Get metadata info (size, last modified, type, and line count for text files) about a path."""
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


def tool_write_file(sandbox_dir: str, path: str, content: str) -> str:
  """Write text content to a file inside the sandbox."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
      
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
      
    with open(safe_p, 'w', encoding='utf-8') as f:
      f.write(new_content)
      
    rel_path = os.path.relpath(safe_p, sandbox_dir)
    print_diff(rel_path, content, new_content)
    return f"Successfully updated file '{rel_path}' using a target replacement patch."
  except Exception as e:
    return f"Error patching file: {str(e)}"


def tool_multi_patch(sandbox_dir: str, path: str, patches: List[Dict[str, str]]) -> str:
  """Apply multiple non-contiguous exact text replacements to a file inside the sandbox."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: File '{path}' does not exist. Use write_file to create new files."
    if not os.path.isfile(safe_p):
      return f"Error: Path '{path}' is not a file."

    with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
      content = f.read()

    normalized_content = content.replace("\r\n", "\n")
    patch_ranges = []

    for idx, patch in enumerate(patches):
      if not isinstance(patch, dict):
        return f"Error: Patch at index {idx} must be an object/dictionary."
      search = patch.get("search")
      replace = patch.get("replace")
      if search is None or replace is None:
        return f"Error: Patch at index {idx} is missing 'search' or 'replace' key."

      normalized_search = search.replace("\r\n", "\n")
      normalized_replace = replace.replace("\r\n", "\n")

      if normalized_search not in normalized_content:
        return (
          f"Error: The search block in patch {idx + 1} was not found in '{path}'. "
          "Make sure you specify the search text exactly including leading whitespace and indentation. "
          "Note: Line endings (CRLF vs LF) are automatically normalized and ignored."
        )

      occurrences = normalized_content.count(normalized_search)
      if occurrences > 1:
        return (
          f"Error: Found {occurrences} occurrences of the search block in patch {idx + 1} inside '{path}'. "
          "Please provide more context lines to make the search block unique."
        )

      start_char = normalized_content.find(normalized_search)
      end_char = start_char + len(normalized_search)

      patch_ranges.append({
        "index": idx,
        "start": start_char,
        "end": end_char,
        "search": normalized_search,
        "replace": normalized_replace
      })

    # Check for overlaps
    sorted_ranges = sorted(patch_ranges, key=lambda x: x["start"])
    for i in range(len(sorted_ranges) - 1):
      if sorted_ranges[i]["end"] > sorted_ranges[i + 1]["start"]:
        return (
          f"Error: Overlapping patches detected. "
          f"Patch {sorted_ranges[i]['index'] + 1} overlaps with Patch {sorted_ranges[i+1]['index'] + 1}."
        )

    # Apply replacements from end to start to avoid shifting indices
    updated_normalized = normalized_content
    sorted_ranges_desc = sorted(patch_ranges, key=lambda x: x["start"], reverse=True)

    for r in sorted_ranges_desc:
      start = r["start"]
      end = r["end"]
      rep = r["replace"]
      updated_normalized = updated_normalized[:start] + rep + updated_normalized[end:]

    # Restore original dominant line ending style
    if "\r\n" in content:
      new_content = updated_normalized.replace("\n", "\r\n")
    else:
      new_content = updated_normalized

    with open(safe_p, 'w', encoding='utf-8') as f:
      f.write(new_content)

    rel_path = os.path.relpath(safe_p, sandbox_dir)
    print_diff(rel_path, content, new_content)
    return f"Successfully updated file '{rel_path}' by applying {len(patches)} patches."
  except Exception as e:
    return f"Error multi-patching file: {str(e)}"


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
    
    with open(safe_p, 'w', encoding='utf-8') as f:
      f.writelines(lines)
      
    rel_path = os.path.relpath(safe_p, sandbox_dir)
    print_diff(rel_path, original_content, "".join(lines))
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

    if ext == ".py":
      # Try black, ruff, yapf, autopep8
      if shutil.which("black"):
        cmd_args = ["black", "-q", safe_p]
        record_command_binaries(cmd_args)
        subprocess.run(cmd_args, capture_output=True, text=True)
        formatter_used = "black"
      elif shutil.which("ruff"):
        cmd_args = ["ruff", "format", safe_p]
        record_command_binaries(cmd_args)
        subprocess.run(cmd_args, capture_output=True, text=True)
        formatter_used = "ruff format"
      elif shutil.which("yapf"):
        cmd_args = ["yapf", "-i", safe_p]
        record_command_binaries(cmd_args)
        subprocess.run(cmd_args, capture_output=True, text=True)
        formatter_used = "yapf"
      elif shutil.which("autopep8"):
        cmd_args = ["autopep8", "-i", safe_p]
        record_command_binaries(cmd_args)
        subprocess.run(cmd_args, capture_output=True, text=True)
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
          cmd_args = ["prettier", "--write", safe_p]
          record_command_binaries(cmd_args)
          subprocess.run(cmd_args, capture_output=True, text=True)
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
        cmd_args = ["clang-format", "-i", safe_p]
        record_command_binaries(cmd_args)
        subprocess.run(cmd_args, capture_output=True, text=True)
        formatter_used = "clang-format"
        with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
          formatted_content = f.read()
      else:
        return "Error: clang-format is not installed on the system."

    elif ext == ".go":
      if shutil.which("gofmt"):
        cmd_args = ["gofmt", "-w", safe_p]
        record_command_binaries(cmd_args)
        subprocess.run(cmd_args, capture_output=True, text=True)
        formatter_used = "gofmt"
        with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
          formatted_content = f.read()
      else:
        return "Error: gofmt is not installed on the system."

    elif ext in (".rs", ".rlib"):
      if shutil.which("rustfmt"):
        cmd_args = ["rustfmt", safe_p]
        record_command_binaries(cmd_args)
        subprocess.run(cmd_args, capture_output=True, text=True)
        formatter_used = "rustfmt"
        with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
          formatted_content = f.read()
      else:
        return "Error: rustfmt is not installed on the system."

    elif ext in (".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".md"):
      if shutil.which("prettier"):
        cmd_args = ["prettier", "--write", safe_p]
        record_command_binaries(cmd_args)
        subprocess.run(cmd_args, capture_output=True, text=True)
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


def tool_sleep(seconds: float) -> str:
  """Sleep for a specified number of seconds to wait for background operations to progress."""
  try:
    sec = float(seconds)
    if sec < 0:
      return "Error: sleep duration cannot be negative."
    if sec > 60:
      return "Error: maximum sleep duration is 60 seconds."
    time.sleep(sec)
    return f"Successfully slept for {sec} seconds."
  except Exception as e:
    return f"Error sleeping: {str(e)}"


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
          }
        },
        "required": ["path", "search", "replace"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "multi_patch",
      "description": "Apply multiple non-contiguous exact text replacements to a file. The operation is atomic: if any patch fails to match uniquely, the entire operation is aborted.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The file path relative to the sandbox root."
          },
          "patches": {
            "type": "array",
            "description": "The list of patches to apply. Patches are matched against the original file content.",
            "items": {
              "type": "object",
              "properties": {
                "search": {
                  "type": "string",
                  "description": "The exact block of code/text to replace. Must be unique in the original file."
                },
                "replace": {
                  "type": "string",
                  "description": "The code/text to replace the search block with."
                }
              },
              "required": ["search", "replace"]
            }
          }
        },
        "required": ["path", "patches"]
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
          },
          "line_numbers": {
            "type": "boolean",
            "description": "Set to true to include 1-indexed line numbers at the beginning of each line (formatted as 'line_num: line_content'). Defaults to false."
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
      "description": "Execute a shell command, returning its stdout, stderr, and exit status code. The command will run with its working directory (cwd) set to the sandbox folder. WARNING: You are strictly prohibited from using this tool to search files (use search_grep), find files (use locate_files), view/inspect files (use read_file/get_file_info), count lines/words (use get_file_info), or pause execution (use sleep). Using commands like 'grep', 'find', 'cat', 'head', 'tail', 'sed', 'awk', 'less', 'more', or 'sleep' directly will fail with an error. Always use get_file_info instead of 'wc -l' to count lines in files, and use the 'sleep' tool to pause execution.",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "The exact shell command to execute."
          },
          "output_filter": {
            "type": "string",
            "description": "Optional regular expression pattern. If specified, only lines from the output matching this pattern will be returned. Use this to filter large command outputs (e.g. to search for 'FAIL', 'Error', or specific test/log names) and prevent context window truncation."
          },
          "tail_lines": {
            "type": "integer",
            "description": "Optional. Only return the last N lines of the command output (similar to 'tail -n N'). Useful for viewing the end of long execution or build logs."
          },
          "head_lines": {
            "type": "integer",
            "description": "Optional. Only return the first N lines of the command output (similar to 'head -n N'). Useful for viewing startup logs, headers, or initial error messages."
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
      "description": "Check the status, output, and exit status code of a command running in the background.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": {
            "type": "string",
            "description": "The Task ID returned by run_command (e.g. 'task_1')."
          },
          "timeout": {
            "type": "number",
            "description": "Optional. The number of seconds to wait/block for the background task to complete. If the task is still running after this timeout, the status will show that it is still running."
          }
        },
        "required": ["task_id"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "kill_process",
      "description": "Terminate a process running in the background using its Task ID.",
      "parameters": {
        "type": "object",
        "properties": {
          "task_id": {
            "type": "string",
            "description": "The Task ID of the background command to terminate (e.g. 'task_1')."
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
      "description": "Get metadata info (size, last modified, type, and line count for text files) about a path inside the sandbox. Use this tool instead of shell commands like 'wc' or 'wc -l' via run_command.",
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
  },
  {
    "type": "function",
    "function": {
      "name": "sleep",
      "description": "Sleep for a specified number of seconds to wait for background operations to progress.",
      "parameters": {
        "type": "object",
        "properties": {
          "seconds": {
            "type": "number",
            "description": "The number of seconds to sleep."
          }
        },
        "required": ["seconds"]
      }
    }
  }
]


def execute_tool(name: str, arguments: Dict[str, Any], session: Any) -> str:
  """Executes the specified tool with arguments in the sandbox directory."""
  if not hasattr(session, "tool_calls_count"):
    session.tool_calls_count = {}
  session.tool_calls_count[name] = session.tool_calls_count.get(name, 0) + 1

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
    line_numbers = bool(arguments.get("line_numbers")) if arguments.get("line_numbers") is not None else False
    return tool_read_file(session.sandbox, path, start_line, end_line, max_chars=session.max_read_chars, line_numbers=line_numbers)
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
  elif name == "multi_patch":
    path = arguments.get("path")
    patches = arguments.get("patches")
    if not path or not isinstance(patches, list):
      return "Error: Missing parameter 'path' or 'patches' must be a list of patch objects."
    return tool_multi_patch(session.sandbox, path, patches)
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
    line_numbers = arguments.get("line_numbers", False)
    if not pattern:
      return "Error: Missing parameter 'pattern'."
    return tool_search_grep(session.sandbox, pattern, path, max_results=session.max_grep_results, line_numbers=line_numbers)
  elif name == "run_command":
    command = arguments.get("command")
    if not command:
      return "Error: Missing parameter 'command'."
    output_filter = arguments.get("output_filter")
    tail_lines = arguments.get("tail_lines")
    head_lines = arguments.get("head_lines")
    if tail_lines is not None:
      try:
        tail_lines = int(tail_lines)
      except (ValueError, TypeError):
        return "Error: tail_lines must be a valid integer."
    if head_lines is not None:
      try:
        head_lines = int(head_lines)
      except (ValueError, TypeError):
        return "Error: head_lines must be a valid integer."
    return session.tool_run_command(command, output_filter=output_filter, tail_lines=tail_lines, head_lines=head_lines)
  elif name == "check_background_command":
    task_id = arguments.get("task_id")
    if not task_id:
      return "Error: Missing parameter 'task_id'."
    timeout = arguments.get("timeout")
    if timeout is not None:
      try:
        timeout = float(timeout)
      except (ValueError, TypeError):
        return "Error: timeout must be a valid number."
    return session.tool_check_background_command(task_id, timeout=timeout)
  elif name == "kill_process":
    task_id = arguments.get("task_id")
    if not task_id:
      return "Error: Missing parameter 'task_id'."
    return session.tool_kill_process(task_id)
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
  elif name == "sleep":
    seconds = arguments.get("seconds")
    if seconds is None:
      return "Error: Missing parameter 'seconds'."
    try:
      seconds = float(seconds)
    except (ValueError, TypeError):
      return "Error: seconds must be a valid number."
    return tool_sleep(seconds)
  else:
    return f"Error: Tool '{name}' is not recognized."
