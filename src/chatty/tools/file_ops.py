import os
import time
import shutil
import json
import subprocess
from typing import List, Dict, Any, Tuple, Optional

from chatty.safety import (
  get_safe_path,
  is_text_file,
  count_lines
)
from chatty.utils import print_diff, record_command_binaries


def make_file_preview(safe_p: str, highlight_ranges: List[Tuple[int, int]], context_lines: int = 5) -> str:
  """Generates a line-numbered preview of the file.

  If the file has <= 100 lines, shows the whole file with line numbers.
  Otherwise, shows context around the specified highlight ranges.
  """
  try:
    with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
      lines = f.readlines()

    total_lines = len(lines)
    if total_lines <= 100:
      preview_content = "".join(f"{idx + 1}: {line}" for idx, line in enumerate(lines))
      return f"File '{os.path.basename(safe_p)}' now has {total_lines} lines:\n```\n{preview_content}```"

    # Large file, compile windows
    show_lines = set()
    for start, end in highlight_ranges:
      s = max(1, start - context_lines)
      e = min(total_lines, end + context_lines)
      for i in range(s, e + 1):
        show_lines.add(i)

    sorted_show = sorted(list(show_lines))
    if not sorted_show:
      return f"File '{os.path.basename(safe_p)}' now has {total_lines} lines."

    chunks = []
    current_chunk = []
    for line_num in sorted_show:
      if not current_chunk:
        current_chunk.append(line_num)
      elif line_num == current_chunk[-1] + 1:
        current_chunk.append(line_num)
      else:
        chunks.append(current_chunk)
        current_chunk = [line_num]
    if current_chunk:
      chunks.append(current_chunk)

    preview_parts = []
    last_end = 0
    for chunk in chunks:
      start = chunk[0]
      end = chunk[-1]
      if start > last_end + 1:
        preview_parts.append(f"... (lines {last_end+1}-{start-1} truncated) ...\n")
      chunk_str = "".join(f"{num}: {lines[num-1]}" for num in chunk)
      preview_parts.append(chunk_str)
      last_end = end

    if last_end < total_lines:
      preview_parts.append(f"... (lines {last_end+1}-{total_lines} truncated) ...\n")

    preview_content = "".join(preview_parts)
    return f"File '{os.path.basename(safe_p)}' now has {total_lines} lines. Preview of changed sections:\n```\n{preview_content}```"
  except Exception as e:
    return f"File updated, but failed to generate preview: {str(e)}"


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
    
    start_char = normalized_content.find(normalized_search)
    start_line = normalized_content[:start_char].count('\n') + 1
    new_lines_count = len(normalized_replace.split('\n'))
    end_line = start_line + new_lines_count - 1
    
    preview = make_file_preview(safe_p, [(start_line, end_line)])
    return f"Successfully updated file '{rel_path}' using a target replacement patch.\n\n{preview}"
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
    
    sorted_ranges = sorted(patch_ranges, key=lambda x: x["start"])
    shift = 0
    new_highlight_ranges = []
    for r in sorted_ranges:
      new_start_char = r["start"] + shift
      new_start_line = updated_normalized[:new_start_char].count('\n') + 1
      new_lines_count = len(r["replace"].split('\n'))
      new_end_line = new_start_line + new_lines_count - 1
      new_highlight_ranges.append((new_start_line, new_end_line))
      shift += len(r["replace"]) - len(r["search"])
      
    preview = make_file_preview(safe_p, new_highlight_ranges)
    return f"Successfully updated file '{rel_path}' by applying {len(patches)} patches.\n\n{preview}"
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
    
    new_end_line = start_line + max(0, inserted_count - 1) if inserted_count > 0 else start_line
    preview = make_file_preview(safe_p, [(start_line, new_end_line)])
    return (
      f"Successfully updated file '{rel_path}': replaced lines {start_line}-{end_line} "
      f"({replaced_count} lines) with {inserted_count} new lines.\n\n{preview}"
    )
  except Exception as e:
    return f"Error editing lines: {str(e)}"


def tool_multi_edit_lines(sandbox_dir: str, path: str, edits: List[Dict[str, Any]]) -> str:
  """Apply multiple line range edits to a file inside the sandbox.

  All start_line and end_line coordinates are 1-indexed, inclusive, and refer to
  the original file content before any edits are applied. The edits must not
  overlap.
  """
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

    # Validate and parse edits
    parsed_edits = []
    for idx, edit in enumerate(edits):
      if not isinstance(edit, dict):
        return f"Error: Edit at index {idx} must be an object/dictionary."
      start_line = edit.get("start_line")
      end_line = edit.get("end_line")
      replacement = edit.get("replacement")
      if start_line is None or end_line is None or replacement is None:
        return f"Error: Edit at index {idx} is missing 'start_line', 'end_line', or 'replacement'."
      try:
        start_line = int(start_line)
        end_line = int(end_line)
      except (ValueError, TypeError):
        return f"Error: Edit at index {idx} start_line and end_line must be valid integers."

      if start_line < 1 or start_line > total_lines:
        return f"Error: Edit {idx + 1} start_line {start_line} is out of range. The file '{path}' has {total_lines} lines."
      if end_line < start_line or end_line > total_lines:
        return f"Error: Edit {idx + 1} end_line {end_line} is invalid (must be between start_line {start_line} and total file lines {total_lines})."

      parsed_edits.append({
        "index": idx,
        "start": start_line,
        "end": end_line,
        "replacement": replacement
      })

    # Check for overlaps
    sorted_edits = sorted(parsed_edits, key=lambda x: x["start"])
    for i in range(len(sorted_edits) - 1):
      if sorted_edits[i]["end"] >= sorted_edits[i + 1]["start"]:
        return (
          f"Error: Overlapping edits detected. "
          f"Edit {sorted_edits[i]['index'] + 1} (lines {sorted_edits[i]['start']}-{sorted_edits[i]['end']}) "
          f"overlaps with Edit {sorted_edits[i+1]['index'] + 1} (lines {sorted_edits[i+1]['start']}-{sorted_edits[i+1]['end']})."
        )

    has_crlf = any("\r\n" in line for line in lines)
    suffix = "\r\n" if has_crlf else "\n"

    # Modify the lines list from bottom to top (reverse sorted_edits)
    sorted_desc = sorted(parsed_edits, key=lambda x: x["start"], reverse=True)
    updated_lines = list(lines)

    for edit in sorted_desc:
      start = edit["start"]
      end = edit["end"]
      rep = edit["replacement"]

      if rep == "":
        rep_lines_formatted = []
      else:
        rep_normalized = rep.replace("\r\n", "\n")
        rep_lines = rep_normalized.split("\n")
        if rep_normalized.endswith("\n") and len(rep_lines) > 1 and rep_lines[-1] == "":
          rep_lines = rep_lines[:-1]
        rep_lines_formatted = [line + suffix for line in rep_lines]

      slice_start = start - 1
      slice_end = end
      updated_lines[slice_start:slice_end] = rep_lines_formatted

    new_content = "".join(updated_lines)
    with open(safe_p, 'w', encoding='utf-8') as f:
      f.writelines(updated_lines)

    # Calculate highlight ranges in the updated file
    sorted_asc = sorted(parsed_edits, key=lambda x: x["start"])
    line_shift = 0
    new_highlight_ranges = []
    for edit in sorted_asc:
      start = edit["start"]
      end = edit["end"]
      rep = edit["replacement"]

      if rep == "":
        rep_lines_count = 0
      else:
        rep_normalized = rep.replace("\r\n", "\n")
        rep_lines = rep_normalized.split("\n")
        if rep_normalized.endswith("\n") and len(rep_lines) > 1 and rep_lines[-1] == "":
          rep_lines = rep_lines[:-1]
        rep_lines_count = len(rep_lines)

      new_start = start + line_shift
      new_end = new_start + rep_lines_count - 1
      if rep_lines_count > 0:
        new_highlight_ranges.append((new_start, new_end))
      else:
        new_highlight_ranges.append((new_start, new_start))

      line_shift += rep_lines_count - (end - start + 1)

    rel_path = os.path.relpath(safe_p, sandbox_dir)
    print_diff(rel_path, original_content, new_content)
    preview = make_file_preview(safe_p, new_highlight_ranges)

    return f"Successfully updated file '{rel_path}' by applying {len(edits)} line range edits.\n\n{preview}"
  except Exception as e:
    return f"Error multi-editing lines: {str(e)}"


def get_available_formatters() -> List[str]:
  """Returns a list of formatting tools currently available on the system path."""
  formatters = []
  for tool in ["black", "ruff", "clang-format", "prettier", "gofmt", "rustfmt", "yapf", "autopep8"]:
    if shutil.which(tool):
      formatters.append(tool)
  return formatters


def tool_format_file(sandbox_dir: str, path: str, formatter: str = None, config_path: str = None) -> str:
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
    
    # 1. Resolve custom configuration file path if specified
    config_abs_path = None
    if config_path:
      config_abs_path = get_safe_path(sandbox_dir, config_path)
      if not os.path.exists(config_abs_path):
        return f"Error: Configuration file '{config_path}' does not exist."

    # 2. Determine which formatter to use
    chosen_formatter = None
    if formatter:
      formatter_lower = formatter.lower()
      if not shutil.which(formatter_lower):
        if formatter_lower in ("built-in-json", "built-in-yaml", "built-in"):
          chosen_formatter = formatter_lower
        else:
          return f"Error: Formatter '{formatter}' is not installed or not found in system path."
      else:
        chosen_formatter = formatter_lower
    else:
      # Auto-select based on file extension
      if ext == ".py":
        for tool in ["black", "ruff", "yapf", "autopep8"]:
          if shutil.which(tool):
            chosen_formatter = tool
            break
        if not chosen_formatter:
          return "Error: No Python formatter found (black, ruff, yapf, or autopep8)."
      elif ext in (".c", ".cpp", ".h", ".hpp", ".cs", ".java", ".sv", ".svh", ".v"):
        if shutil.which("clang-format"):
          chosen_formatter = "clang-format"
        else:
          return "Error: clang-format is not installed on the system."
      elif ext == ".go":
        if shutil.which("gofmt"):
          chosen_formatter = "gofmt"
        else:
          return "Error: gofmt is not installed on the system."
      elif ext in (".rs", ".rlib"):
        if shutil.which("rustfmt"):
          chosen_formatter = "rustfmt"
        else:
          return "Error: rustfmt is not installed on the system."
      elif ext in (".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".md", ".json", ".yaml", ".yml"):
        if shutil.which("prettier"):
          chosen_formatter = "prettier"
        elif ext == ".json":
          chosen_formatter = "built-in-json"
        elif ext in (".yaml", ".yml"):
          chosen_formatter = "built-in-yaml"
        else:
          return "Error: prettier is not installed on the system."
      else:
        return f"Error: No formatter configured for files with extension '{ext}'."

    # 3. Run the chosen formatter command
    formatted_content = None
    formatter_used = chosen_formatter

    if chosen_formatter == "black":
      cmd_args = ["black", "-q"]
      if config_abs_path:
        cmd_args.extend(["--config", config_abs_path])
      cmd_args.append(safe_p)
      record_command_binaries(cmd_args, None) # wait, record_command_binaries signature? We will check.
      subprocess.run(cmd_args, capture_output=True, text=True)

    elif chosen_formatter == "ruff":
      cmd_args = ["ruff", "format"]
      if config_abs_path:
        cmd_args.extend(["--config", config_abs_path])
      cmd_args.append(safe_p)
      record_command_binaries(cmd_args, None)
      subprocess.run(cmd_args, capture_output=True, text=True)

    elif chosen_formatter == "clang-format":
      cmd_args = ["clang-format", "-i"]
      if config_abs_path:
        cmd_args.append(f"-style=file:{config_abs_path}")
      cmd_args.append(safe_p)
      record_command_binaries(cmd_args, None)
      subprocess.run(cmd_args, capture_output=True, text=True)

    elif chosen_formatter == "prettier":
      cmd_args = ["prettier", "--write"]
      if config_abs_path:
        cmd_args.extend(["--config", config_abs_path])
      cmd_args.append(safe_p)
      record_command_binaries(cmd_args, None)
      subprocess.run(cmd_args, capture_output=True, text=True)

    elif chosen_formatter == "gofmt":
      cmd_args = ["gofmt", "-w", safe_p]
      record_command_binaries(cmd_args, None)
      subprocess.run(cmd_args, capture_output=True, text=True)

    elif chosen_formatter == "rustfmt":
      cmd_args = ["rustfmt"]
      if config_abs_path:
        cmd_args.extend(["--config-path", config_abs_path])
      cmd_args.append(safe_p)
      record_command_binaries(cmd_args, None)
      subprocess.run(cmd_args, capture_output=True, text=True)

    elif chosen_formatter == "yapf":
      cmd_args = ["yapf", "-i"]
      if config_abs_path:
        cmd_args.extend(["--style", config_abs_path])
      cmd_args.append(safe_p)
      record_command_binaries(cmd_args, None)
      subprocess.run(cmd_args, capture_output=True, text=True)

    elif chosen_formatter == "autopep8":
      cmd_args = ["autopep8", "-i"]
      if config_abs_path:
        cmd_args.extend(["--global-config", config_abs_path])
      cmd_args.append(safe_p)
      record_command_binaries(cmd_args, None)
      subprocess.run(cmd_args, capture_output=True, text=True)

    elif chosen_formatter in ("built-in-json", "built-in"):
      try:
        data = json.loads(old_content)
        formatted_content = json.dumps(data, indent=2) + "\n"
        formatter_used = "built-in JSON formatter"
      except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"

    elif chosen_formatter in ("built-in-yaml", "built-in"):
      import yaml
      try:
        data = yaml.safe_load(old_content)
        formatted_content = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        formatter_used = "built-in PyYAML formatter"
      except Exception as e:
        return f"Error formatting YAML: {str(e)}"

    else:
      return f"Error: Unsupported formatter '{chosen_formatter}'."

    # 4. Save and return results
    if formatted_content is not None and formatted_content != old_content:
      with open(safe_p, 'w', encoding='utf-8') as f:
        f.write(formatted_content)

    if formatted_content is None:
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
  """Delete a file inside the sandbox. Fails if the path is a directory."""
  try:
    safe_path = get_safe_path(sandbox_dir, path)
    
    if not os.path.exists(safe_path):
      return f"Error: Path '{path}' does not exist."
      
    if os.path.isdir(safe_path):
      return f"Error: Path '{path}' is a directory. Use 'delete_directory' instead."
      
    rel_path = os.path.relpath(safe_path, sandbox_dir)
    os.remove(safe_path)
    return f"Successfully deleted file '{rel_path}'."
  except Exception as e:
    return f"Error deleting file: {str(e)}"


def tool_delete_directory(sandbox_dir: str, path: str, recursive: bool = False) -> str:
  """Delete a directory inside the sandbox."""
  try:
    safe_path = get_safe_path(sandbox_dir, path)
    
    if not os.path.exists(safe_path):
      return f"Error: Path '{path}' does not exist."
      
    if not os.path.isdir(safe_path):
      return f"Error: Path '{path}' is a file. Use 'delete_file' instead."
      
    rel_path = os.path.relpath(safe_path, sandbox_dir)
    
    if not recursive:
      try:
        contents = os.listdir(safe_path)
        if contents:
          return f"Error: Directory '{rel_path}' is not empty. Set recursive=True to delete it and all its contents."
      except Exception:
        pass
        
    if recursive:
      shutil.rmtree(safe_path)
    else:
      os.rmdir(safe_path)
    return f"Successfully deleted directory '{rel_path}'."
  except Exception as e:
    return f"Error deleting directory: {str(e)}"


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


def tool_hex_dump(
    sandbox_dir: str,
    path: str,
    start_offset: int = 0,
    size: int = 256,
    format: str = "canonical",
    word_size: int = 8,
    endian: str = "little",
    signed: bool = False
) -> str:
  """Inspect a binary file by dumping its content as formatted hex/integers."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    if not os.path.exists(safe_p):
      return f"Error: File '{path}' does not exist."
    if not os.path.isfile(safe_p):
      return f"Error: Path '{path}' is not a file."
      
    file_size = os.path.getsize(safe_p)
    if start_offset < 0 or start_offset >= file_size:
      if file_size == 0 and start_offset == 0:
        return f"File '{path}' is empty (0 bytes)."
      return f"Error: start_offset {start_offset} is out of range. File '{path}' is {file_size} bytes."
      
    if size <= 0:
      return "Error: size must be greater than 0."
    
    # Cap size to prevent huge dumps (e.g. max 16KB)
    max_dump_bytes = 16384
    if size > max_dump_bytes:
      size = max_dump_bytes
      truncated_msg = f"\n\n[WARNING: Hex dump size truncated to max limit of {max_dump_bytes} bytes.]"
    else:
      truncated_msg = ""
      
    with open(safe_p, "rb") as f:
      f.seek(start_offset)
      data = f.read(size)
      
    if not data:
      return f"No bytes read from file '{path}' at offset {start_offset}."
      
    # Validate word size
    word_bytes = word_size // 8
    if word_bytes not in (1, 2, 4, 8) or word_size % 8 != 0:
      return f"Error: Unsupported word_size {word_size}-bit. Must be 8, 16, 32, or 64."
      
    if endian not in ("little", "big"):
      return f"Error: Unsupported endianness '{endian}'. Must be 'little' or 'big'."
      
    # Validate format parameter
    valid_formats = ("canonical", "hex", "dec", "int", "raw")
    if format not in valid_formats:
      return f"Error: Unsupported format '{format}'. Supported formats: {', '.join(valid_formats)}."

    # If raw format is requested, return simple hex of raw bytes (only makes sense for 8-bit/byte level)
    if format == "raw":
      if word_size != 8:
        return "Error: format 'raw' is only supported with word_size=8."
      return data.hex() + truncated_msg

    # Handling canonical 8-bit layout (hexdump -C layout)
    if word_size == 8 and format == "canonical":
      lines = []
      for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        addr = start_offset + i
        h1 = " ".join(f"{b:02x}" for b in chunk[:8])
        h2 = " ".join(f"{b:02x}" for b in chunk[8:])
        hex_str = f"{h1:<23}  {h2:<23}"
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{addr:08x}  {hex_str}  |{ascii_str}|")
      return "\n".join(lines) + truncated_msg

    # Otherwise, parse bytes into integers of requested word size
    num_words = len(data) // word_bytes
    leftover = len(data) % word_bytes
    
    chunks = [data[i * word_bytes : (i + 1) * word_bytes] for i in range(num_words)]
    values = [int.from_bytes(chunk, byteorder=endian, signed=signed) for chunk in chunks]
    
    leftover_msg = ""
    if leftover > 0:
      leftover_msg = f"\n\n[WARNING: Ignored {leftover} trailing byte(s) because they do not fit into the {word_size}-bit word size.]"

    lines = []
    for i, val in enumerate(values):
      addr = start_offset + i * word_bytes
      
      if format == "canonical":
        # Formatted details including address, hex representation, and decimal value
        if word_size == 16:
          lines.append(f"{addr:08x}  0x{val:04x}  ({val})")
        elif word_size == 32:
          lines.append(f"{addr:08x}  0x{val:08x}  ({val})")
        elif word_size == 64:
          lines.append(f"{addr:08x}  0x{val:016x}  ({val})")
        else: # word_size == 8 (non-canonical hex list with dec)
          lines.append(f"{addr:08x}  0x{val:02x}  ({val})")
          
      elif format == "hex":
        if word_size == 8:
          lines.append(f"{addr:08x}: 0x{val:02x}")
        elif word_size == 16:
          lines.append(f"{addr:08x}: 0x{val:04x}")
        elif word_size == 32:
          lines.append(f"{addr:08x}: 0x{val:08x}")
        elif word_size == 64:
          lines.append(f"{addr:08x}: 0x{val:016x}")
          
      elif format in ("dec", "int"):
        lines.append(f"{addr:08x}: {val}")

    return "\n".join(lines) + truncated_msg + leftover_msg
  except Exception as e:
    return f"Error performing hex dump: {str(e)}"

