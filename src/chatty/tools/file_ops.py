import os
import time
import shutil
import json
import subprocess
import re
import difflib
from typing import List, Dict, Any, Tuple, Optional, Union

from chatty.safety import (
  get_safe_path,
  is_text_file,
  count_lines
)
from chatty.utils import print_diff, record_command_binaries
from chatty.backup import backup_file


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
    safe_p = get_safe_path(sandbox_dir, path, write=True)

    rel_path = os.path.relpath(safe_p, sandbox_dir)
    backup_file(sandbox_dir, rel_path)
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


def parse_aider_patches(patch_text: str) -> List[Tuple[str, str]]:
  """Parses a string containing one or more Aider-style search/replace blocks.

  Format:
  <<<<<<< SEARCH
  old code
  =======
  new code
  >>>>>>> REPLACE
  """
  lines = patch_text.splitlines()
  patches = []
  
  in_search = False
  in_replace = False
  search_lines = []
  replace_lines = []
  
  for idx, line in enumerate(lines):
    if line.startswith("<<<<<<< SEARCH"):
      if in_search or in_replace:
        raise ValueError(f"Nested or malformed SEARCH block at line {idx+1}")
      in_search = True
      search_lines = []
    elif line.startswith("======="):
      if not in_search:
        raise ValueError(f"Unexpected ======= marker without SEARCH block at line {idx+1}")
      in_search = False
      in_replace = True
      replace_lines = []
    elif line.startswith(">>>>>>> REPLACE"):
      if not in_replace:
        raise ValueError(f"Unexpected >>>>>>> REPLACE marker without SEARCH/REPLACE block at line {idx+1}")
      in_replace = False
      patches.append(("\n".join(search_lines), "\n".join(replace_lines)))
    else:
      if in_search:
        search_lines.append(line)
      elif in_replace:
        replace_lines.append(line)
        
  if in_search or in_replace:
    raise ValueError("Unclosed SEARCH or REPLACE block in patch text")
    
  return patches


def get_indent_info(line: str) -> Tuple[str, int]:
  """Determines the indentation character and count for a line."""
  indent_chars = ""
  for char in line:
    if char in (' ', '\t'):
      indent_chars += char
    else:
      break
  if '\t' in indent_chars:
    return '\t', indent_chars.count('\t')
  else:
    return ' ', len(indent_chars)


def apply_shift(line: str, shift: int, indent_char: str) -> str:
  """Shifts the indentation of a line by a specified amount."""
  if not line.strip():
    return ""
  if shift == 0:
    return line
  if shift > 0:
    return (indent_char * shift) + line
  else:
    to_remove = abs(shift)
    while to_remove > 0 and line.startswith(indent_char):
      line = line[1:]
      to_remove -= 1
    return line


def compute_shift(file_slice_lines: List[str], search_slice_lines: List[str]) -> Tuple[int, str]:
  """Computes indentation shift between file lines and search lines."""
  for f_line, s_line in zip(file_slice_lines, search_slice_lines):
    if f_line.strip() and s_line.strip():
      char_f, count_f = get_indent_info(f_line)
      _, count_s = get_indent_info(s_line)
      if count_f > 0 or count_s > 0:
        return count_f - count_s, char_f
  f_line = next((l for l in file_slice_lines if l.strip()), "")
  s_line = next((l for l in search_slice_lines if l.strip()), "")
  if f_line and s_line:
    char_f, count_f = get_indent_info(f_line)
    _, count_s = get_indent_info(s_line)
    return count_f - count_s, char_f
  return 0, " "


def normalize_code_line(line: str) -> str:
  """Collapses multiple whitespace characters to a single space and strips ends."""
  return re.sub(r'\s+', ' ', line).strip()


def find_block_in_file(file_content: str, search_block: str) -> Tuple[str, Any]:
  """Finds the search block in the file content using cascading exact and fuzzy matching.

  Returns (status, match_info):
  - If found: ("found", (start_char, end_char, shift, indent_char))
  - If not unique: ("not_unique", list_of_matching_line_numbers)
  - If not found: ("not_found", closest_match_dict_or_None)
  - If empty: ("empty_search", None)
  """
  file_content_norm = file_content.replace("\r\n", "\n")
  search_block_norm = search_block.replace("\r\n", "\n")

  file_lines = file_content_norm.split("\n")
  search_lines = search_block_norm.split("\n")

  file_lines_rstripped = [re.sub(r'\\+', r'\\', line.rstrip()) for line in file_lines]
  search_lines_rstripped = [re.sub(r'\\+', r'\\', line.rstrip()) for line in search_lines]

  line_start_chars = []
  curr = 0
  for line in file_lines:
    line_start_chars.append(curr)
    curr += len(line) + 1

  num_file_lines = len(file_lines)
  num_search_lines = len(search_lines)

  if num_search_lines == 0 or (num_search_lines == 1 and search_lines[0] == ""):
    return "empty_search", None

  def get_end_char(end_idx: int) -> int:
    if end_idx >= num_file_lines - 1:
      return len(file_content_norm)
    return line_start_chars[end_idx + 1]

  # --- Tier 1: Exact Match (modulo trailing whitespace) ---
  exact_matches = []
  for i in range(num_file_lines - num_search_lines + 1):
    if file_lines_rstripped[i:i + num_search_lines] == search_lines_rstripped:
      exact_matches.append(i)

  if len(exact_matches) == 1:
    start_line = exact_matches[0]
    end_line = start_line + num_search_lines - 1
    return "found", (line_start_chars[start_line], get_end_char(end_line), 0, " ")
  elif len(exact_matches) > 1:
    return "not_unique", exact_matches

  # --- Tier 2: Normalized Indentation (line-by-line stripped equality) ---
  file_lines_stripped = [line.strip() for line in file_lines_rstripped]
  search_lines_stripped = [line.strip() for line in search_lines_rstripped]

  fuzzy_matches = []
  for i in range(num_file_lines - num_search_lines + 1):
    if file_lines_stripped[i:i + num_search_lines] == search_lines_stripped:
      fuzzy_matches.append(i)

  if len(fuzzy_matches) == 1:
    start_line = fuzzy_matches[0]
    end_line = start_line + num_search_lines - 1
    shift, indent_char = compute_shift(file_lines[start_line:end_line + 1], search_lines)
    return "found", (line_start_chars[start_line], get_end_char(end_line), shift, indent_char)
  elif len(fuzzy_matches) > 1:
    return "not_unique", fuzzy_matches

  # --- Tier 3: Whitespace & Blank-Line Collapsed Sublist Match ---
  search_non_empty = [(idx, normalize_code_line(l)) for idx, l in enumerate(search_lines) if normalize_code_line(l)]
  if search_non_empty:
    file_non_empty = [(idx, normalize_code_line(l)) for idx, l in enumerate(file_lines) if normalize_code_line(l)]
    search_tokens = [tok for _, tok in search_non_empty]
    file_tokens = [tok for _, tok in file_non_empty]

    tier3_matches = []
    n_tokens = len(search_tokens)
    for j in range(len(file_tokens) - n_tokens + 1):
      if file_tokens[j:j + n_tokens] == search_tokens:
        f_start = file_non_empty[j][0]
        f_end = file_non_empty[j + n_tokens - 1][0]
        # Include leading blank lines if search_lines had them
        s_lead_blanks = 0
        for l in search_lines:
          if not l.strip():
            s_lead_blanks += 1
          else:
            break
        while s_lead_blanks > 0 and f_start > 0 and not file_lines[f_start - 1].strip():
          f_start -= 1
          s_lead_blanks -= 1
        # Include trailing blank lines if search_lines had them
        s_trail_blanks = 0
        for l in reversed(search_lines):
          if not l.strip():
            s_trail_blanks += 1
          else:
            break
        while s_trail_blanks > 0 and f_end < num_file_lines - 1 and not file_lines[f_end + 1].strip():
          f_end += 1
          s_trail_blanks -= 1
        tier3_matches.append((f_start, f_end))

    if len(tier3_matches) == 1:
      start_line, end_line = tier3_matches[0]
      shift, indent_char = compute_shift(file_lines[start_line:end_line + 1], search_lines)
      return "found", (line_start_chars[start_line], get_end_char(end_line), shift, indent_char)
    elif len(tier3_matches) > 1:
      return "not_unique", [m[0] for m in tier3_matches]

  # --- Tier 4: Windowed Fuzzy Match (SequenceMatcher) ---
  candidates = []
  if len(search_non_empty) >= 2:
    def norm_block(text: str) -> str:
      lines = text.splitlines()
      return "\n".join(normalize_code_line(l) for l in lines if l.strip())

    norm_search = norm_block(search_block_norm)
    M = len(search_lines)
    min_w = max(2, M - 3)
    max_w = min(num_file_lines, M + 4)

    for W in range(min_w, max_w + 1):
      for i in range(num_file_lines - W + 1):
        cand_lines = file_lines[i:i + W]
        cand_norm = norm_block("\n".join(cand_lines))
        if not cand_norm:
          continue
        sm = difflib.SequenceMatcher(None, norm_search, cand_norm)
        if sm.quick_ratio() >= 0.70:
          r = sm.ratio()
          if r >= 0.50:
            candidates.append((r, i, i + W - 1))

  if candidates:
    candidates.sort(key=lambda c: c[0], reverse=True)
    best_score, best_start, best_end = candidates[0]

    # Find competing non-overlapping candidates
    competing = []
    for c in candidates:
      score, c_start, c_end = c
      overlap = max(0, min(best_end, c_end) - max(best_start, c_start) + 1)
      if overlap <= 1 and score >= 0.70:
        competing.append(c)

    second_best_score = competing[0][0] if competing else 0.0

    SIMILARITY_THRESHOLD = 0.85
    MARGIN_THRESHOLD = 0.15

    if best_score >= SIMILARITY_THRESHOLD:
      if (best_score - second_best_score) >= MARGIN_THRESHOLD:
        # Trim unmatched edge blank lines
        while best_start < best_end and not file_lines[best_start].strip() and search_lines and search_lines[0].strip():
          best_start += 1
        while best_end > best_start and not file_lines[best_end].strip() and search_lines and search_lines[-1].strip():
          best_end -= 1

        shift, indent_char = compute_shift(file_lines[best_start:best_end + 1], search_lines)
        return "found", (line_start_chars[best_start], get_end_char(best_end), shift, indent_char)
      else:
        return "not_unique", [best_start, competing[0][1]]

  closest_info = None
  if candidates:
    best_c = max(candidates, key=lambda c: c[0])
    if best_c[0] >= 0.50:
      s_l = best_c[1] + 1
      e_l = best_c[2] + 1
      snippet = "".join(f"{s_l + idx}: {file_lines[s_l - 1 + idx]}\n" for idx in range(e_l - s_l + 1))
      closest_info = {
        "start_line": s_l,
        "end_line": e_l,
        "similarity": best_c[0],
        "text": snippet
      }

  return "not_found", closest_info


def tool_patch_file(sandbox_dir: str, path: str, patch: str) -> str:
  """Replace one or more unique blocks of text in a file using Aider-style SEARCH/REPLACE blocks."""
  try:
    safe_p = get_safe_path(sandbox_dir, path, write=True)
    if not os.path.exists(safe_p):
      return f"Error: File '{path}' does not exist. Use write_file to create new files."
    if not os.path.isfile(safe_p):
      return f"Error: Path '{path}' is not a file."

    rel_path = os.path.relpath(safe_p, sandbox_dir)
    backup_file(sandbox_dir, rel_path)

    try:
      patches = parse_aider_patches(patch)
    except Exception as e:
      return (
        f"Error parsing Aider-style patches: {str(e)}\n"
        "Make sure you use the exact format:\n"
        "<<<<<<< SEARCH\n"
        "...\n"
        "=======\n"
        "...\n"
        ">>>>>>> REPLACE"
      )

    if not patches:
      return "Error: No SEARCH/REPLACE blocks found in the patch parameter."

    with open(safe_p, 'r', encoding='utf-8', errors='replace') as f:
      content = f.read()

    original_content = content
    highlight_ranges = []

    for idx, (search_block, replace_block) in enumerate(patches):
      status, match_info = find_block_in_file(content, search_block)

      if status == "empty_search":
        return f"Error in patch block {idx+1}: SEARCH block is empty."
      elif status == "not_unique":
        line_info = ""
        if isinstance(match_info, (list, tuple)) and match_info:
          line_info = f" (similar matches near line(s): {', '.join(str(m + 1) for m in match_info[:5])})"
        return f"Error in patch block {idx+1}: SEARCH block is not unique. Please provide more context lines.{line_info}"
      elif status == "not_found":
        msg = f"Error in patch block {idx+1}: SEARCH block not found in file. Make sure it matches the file content."
        if isinstance(match_info, dict) and match_info.get("text"):
          pct = int(match_info["similarity"] * 100)
          msg += (
            f"\nClosest match found at lines {match_info['start_line']}-{match_info['end_line']} "
            f"(similarity {pct}%):\n```\n{match_info['text']}```\n"
            f"You can use the exact lines above for your SEARCH block."
          )
        return msg

      start_char, end_char, shift = match_info[:3]
      indent_char = match_info[3] if len(match_info) > 3 else " "

      replace_lines = replace_block.replace("\r\n", "\n").split("\n")
      if shift != 0:
        shifted_replace_lines = [apply_shift(line, shift, indent_char) for line in replace_lines]
        replace_block_shifted = "\n".join(shifted_replace_lines)
      else:
        replace_block_shifted = "\n".join(replace_lines)

      # Preserve trailing newline of the matched block
      if end_char > start_char and content[end_char - 1] in ('\n', '\r'):
        if not replace_block_shifted.endswith("\n"):
          replace_block_shifted += "\n"

      has_crlf = "\r\n" in content
      if has_crlf:
        replace_block_final = replace_block_shifted.replace("\n", "\r\n")
      else:
        replace_block_final = replace_block_shifted

      content = content[:start_char] + replace_block_final + content[end_char:]

      replaced_lines_count = len(replace_lines)
      start_line_num = content[:start_char].count('\n') + 1
      end_line_num = start_line_num + max(0, replaced_lines_count - 1)
      highlight_ranges.append((start_line_num, end_line_num))

    with open(safe_p, 'w', encoding='utf-8') as f:
      f.write(content)

    rel_path = os.path.relpath(safe_p, sandbox_dir)
    print_diff(rel_path, original_content, content)

    preview = make_file_preview(safe_p, highlight_ranges)
    return f"Successfully updated file '{rel_path}' by applying {len(patches)} patch block(s).\n\n{preview}"

  except Exception as e:
    return f"Error patching file: {str(e)}"


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
    safe_p = get_safe_path(sandbox_dir, path, write=True)
    if not os.path.exists(safe_p):
      return f"Error: File '{path}' does not exist."
    if not os.path.isfile(safe_p):
      return f"Error: Path '{path}' is not a file."

    rel_path = os.path.relpath(safe_p, sandbox_dir)
    backup_file(sandbox_dir, rel_path)
    
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
    safe_src = get_safe_path(sandbox_dir, src, write=True)
    safe_dest = get_safe_path(sandbox_dir, dest, write=True)
    
    if not os.path.exists(safe_src):
      return f"Error: Source path '{src}' does not exist."
      
    if os.path.exists(safe_dest) and os.path.isfile(safe_dest):
      backup_file(sandbox_dir, os.path.relpath(safe_dest, sandbox_dir))
    if os.path.isfile(safe_src):
      backup_file(sandbox_dir, os.path.relpath(safe_src, sandbox_dir))
      
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
    safe_src = get_safe_path(sandbox_dir, src, write=False)
    safe_dest = get_safe_path(sandbox_dir, dest, write=True)
    
    if not os.path.exists(safe_src):
      return f"Error: Source path '{src}' does not exist."
      
    if os.path.exists(safe_dest) and os.path.isfile(safe_dest):
      backup_file(sandbox_dir, os.path.relpath(safe_dest, sandbox_dir))
      
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

def tool_delete_file(sandbox_dir: str, path: Union[str, List[str]]) -> str:
  """Delete a file or multiple files inside the sandbox. Fails if a path is a directory."""
  if isinstance(path, list):
    if not path:
      return "Error: No paths provided."
    results = []
    for p in path:
      if not isinstance(p, str):
        results.append(f"Error: Invalid path type: {type(p).__name__}")
        continue
      results.append(tool_delete_file(sandbox_dir, p))
    return "\n".join(results)

  try:
    safe_path = get_safe_path(sandbox_dir, path, write=True)
    
    if not os.path.exists(safe_path):
      return f"Error: Path '{path}' does not exist."
      
    if os.path.isdir(safe_path):
      return f"Error: Path '{path}' is a directory. Use 'delete_directory' instead."
      
    rel_path = os.path.relpath(safe_path, sandbox_dir)
    backup_file(sandbox_dir, rel_path)
    os.remove(safe_path)
    return f"Successfully deleted file '{rel_path}'."
  except Exception as e:
    return f"Error deleting file: {str(e)}"


def tool_delete_directory(sandbox_dir: str, path: Union[str, List[str]], recursive: bool = False) -> str:
  """Delete a directory or multiple directories inside the sandbox."""
  if isinstance(path, list):
    if not path:
      return "Error: No paths provided."
    results = []
    for p in path:
      if not isinstance(p, str):
        results.append(f"Error: Invalid path type: {type(p).__name__}")
        continue
      results.append(tool_delete_directory(sandbox_dir, p, recursive=recursive))
    return "\n".join(results)

  try:
    safe_path = get_safe_path(sandbox_dir, path, write=True)
    
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
      for root, _, files in os.walk(safe_path):
        for file in files:
          abs_f = os.path.join(root, file)
          rel_f = os.path.relpath(abs_f, sandbox_dir)
          backup_file(sandbox_dir, rel_f)
      shutil.rmtree(safe_path)
    else:
      os.rmdir(safe_path)
    return f"Successfully deleted directory '{rel_path}'."
  except Exception as e:
    return f"Error deleting directory: {str(e)}"


def tool_make_directory(sandbox_dir: str, path: Union[str, List[str]]) -> str:
  """Create a new directory or multiple directories (and any parent directories) inside the sandbox."""
  if isinstance(path, list):
    if not path:
      return "Error: No paths provided."
    results = []
    for p in path:
      if not isinstance(p, str):
        results.append(f"Error: Invalid path type: {type(p).__name__}")
        continue
      results.append(tool_make_directory(sandbox_dir, p))
    return "\n".join(results)

  try:
    safe_path = get_safe_path(sandbox_dir, path, write=True)
    
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


def tool_list_file_backups(sandbox_dir: str, path: str) -> str:
  """Lists all backups for the given path."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    rel_path = os.path.relpath(safe_p, sandbox_dir)
    
    from chatty.backup import list_backups
    backups = list_backups(sandbox_dir, rel_path)
    if not backups:
      return f"No backups found for file '{rel_path}'."
      
    lines = [f"Backups for file '{rel_path}' (newest first):"]
    for idx, (ts, time_str) in enumerate(backups, 1):
      lines.append(f"  {idx}. Timestamp: {ts} | Time: {time_str}")
    return "\n".join(lines)
  except Exception as e:
    return f"Error listing backups: {str(e)}"


def tool_read_file_backup(
    sandbox_dir: str,
    path: str,
    timestamp: int,
    start_line: int = None,
    end_line: int = None,
    max_chars: int = 40000,
    line_numbers: bool = False
) -> str:
  """Reads the contents of a specific backup file for path and timestamp."""
  try:
    safe_p = get_safe_path(sandbox_dir, path)
    rel_path = os.path.relpath(safe_p, sandbox_dir)
    
    from chatty.backup import get_backups_dir
    backup_file_path = os.path.join(get_backups_dir(sandbox_dir), rel_path, f"{timestamp}.bak")
    
    if not os.path.exists(backup_file_path):
      return f"Error: Backup with timestamp '{timestamp}' for '{rel_path}' does not exist."
      
    if not os.path.isfile(backup_file_path):
      return f"Error: Backup path is not a file."
      
    with open(backup_file_path, 'r', encoding='utf-8', errors='replace') as f:
      if start_line is None and end_line is None and not line_numbers:
        content = f.read()
        if len(content) > max_chars:
          return content[:max_chars] + f"\n\n[WARNING: Backup file is too large ({len(content)} characters) and has been truncated. Use 'start_line' and 'end_line' parameters to read specific sections.]"
        return content
        
      lines = f.readlines()
      total_lines = len(lines)
      
      s = 1 if start_line is None else start_line
      e = total_lines if end_line is None else end_line
      
      if s < 1 or s > total_lines:
        return f"Error: start_line {start_line} is out of range. The backup file has {total_lines} lines."
      if e < s or e > total_lines:
        return f"Error: end_line {end_line} is invalid (must be between start_line {s} and total backup file lines {total_lines})."
        
      selected_lines = lines[s-1:e]
      if line_numbers:
        content = "".join(f"{s + idx}: {line}" for idx, line in enumerate(selected_lines))
      else:
        content = "".join(selected_lines)
        
      if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[WARNING: Backup file section is too large and has been truncated.]"
      return content
      
  except Exception as e:
    return f"Error reading backup file: {str(e)}"


