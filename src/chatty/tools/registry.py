from typing import Dict, Any, Callable
import os

from chatty.tools.file_ops import (
  tool_move_file,
  tool_copy_file,
  tool_delete_file,
  tool_delete_directory,
  tool_make_directory,
  tool_list_dir,
  tool_read_file,
  tool_hex_dump,
  tool_write_file,
  tool_patch_file,
  tool_format_file,
  tool_get_file_info
)
from chatty.tools.search_ops import tool_search_grep, tool_locate_files
from chatty.tools.web_ops import tool_fetch_url, tool_search_web
from chatty.tools.system_ops import tool_sleep, tool_ask_question


def handle_move_file(arguments: Dict[str, Any], session: Any) -> str:
  src = arguments.get("src")
  dest = arguments.get("dest")
  if not src or not dest:
    return "Error: Missing parameters 'src' and/or 'dest'."
  return tool_move_file(session.sandbox, src, dest)


def handle_copy_file(arguments: Dict[str, Any], session: Any) -> str:
  src = arguments.get("src")
  dest = arguments.get("dest")
  if not src or not dest:
    return "Error: Missing parameters 'src' and/or 'dest'."
  return tool_copy_file(session.sandbox, src, dest)


def handle_delete_file(arguments: Dict[str, Any], session: Any) -> str:
  path = arguments.get("path")
  if not path:
    return "Error: Missing parameter 'path'."
  return tool_delete_file(session.sandbox, path)


def handle_delete_directory(arguments: Dict[str, Any], session: Any) -> str:
  path = arguments.get("path")
  if not path:
    return "Error: Missing parameter 'path'."
  recursive = bool(arguments.get("recursive", False))
  return tool_delete_directory(session.sandbox, path, recursive)


def handle_make_directory(arguments: Dict[str, Any], session: Any) -> str:
  path = arguments.get("path")
  if not path:
    return "Error: Missing parameter 'path'."
  return tool_make_directory(session.sandbox, path)


def handle_run_tests(arguments: Dict[str, Any], session: Any) -> str:
  return session.tool_run_tests(arguments.get("command"))


def handle_list_dir(arguments: Dict[str, Any], session: Any) -> str:
  return tool_list_dir(session.sandbox, arguments.get("path", "."), max_items=session.max_dir_items)


def handle_read_file(arguments: Dict[str, Any], session: Any) -> str:
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


def handle_hex_dump(arguments: Dict[str, Any], session: Any) -> str:
  path = arguments.get("path")
  if not path:
    return "Error: Missing parameter 'path'."
  try:
    start_offset = int(arguments.get("start_offset")) if arguments.get("start_offset") is not None else 0
    size = int(arguments.get("size")) if arguments.get("size") is not None else 256
  except (ValueError, TypeError):
    return "Error: start_offset and size must be valid integers."
  format_type = arguments.get("format", "canonical")
  try:
    word_size = int(arguments.get("word_size")) if arguments.get("word_size") is not None else 8
  except (ValueError, TypeError):
    return "Error: word_size must be a valid integer."
  endian = arguments.get("endian", "little")
  signed = bool(arguments.get("signed", False))
  return tool_hex_dump(session.sandbox, path, start_offset, size, format_type, word_size, endian, signed)


def handle_write_file(arguments: Dict[str, Any], session: Any) -> str:
  path = arguments.get("path")
  content = arguments.get("content")
  if not path or content is None:
    return "Error: Missing parameters 'path' and 'content'."
  return tool_write_file(session.sandbox, path, content)


def handle_patch_file(arguments: Dict[str, Any], session: Any) -> str:
  path = arguments.get("path")
  patch = arguments.get("patch")
  if not path or patch is None:
    return "Error: Missing parameters 'path' or 'patch'."
  return tool_patch_file(session.sandbox, path, patch)


def handle_multi_patch(arguments: Dict[str, Any], session: Any) -> str:
  return (
    "Error: The 'multi_patch' tool has been deprecated. "
    "Please use the unified 'patch_file' tool with a single string parameter 'patch' "
    "containing one or more Aider-style SEARCH/REPLACE blocks."
  )


def handle_edit_lines(arguments: Dict[str, Any], session: Any) -> str:
  return (
    "Error: The 'edit_lines' tool has been deprecated. "
    "Please use the unified 'patch_file' tool with Aider-style SEARCH/REPLACE blocks instead."
  )


def handle_multi_edit_lines(arguments: Dict[str, Any], session: Any) -> str:
  return (
    "Error: The 'multi_edit_lines' tool has been deprecated. "
    "Please use the unified 'patch_file' tool with Aider-style SEARCH/REPLACE blocks instead."
  )


def handle_format_file(arguments: Dict[str, Any], session: Any) -> str:
  path = arguments.get("path")
  formatter = arguments.get("formatter")
  config_path = arguments.get("config_path")
  if not path:
    return "Error: Missing parameter 'path'."
  return tool_format_file(session.sandbox, path, formatter, config_path)


def handle_search_grep(arguments: Dict[str, Any], session: Any) -> str:
  pattern = arguments.get("pattern")
  path = arguments.get("path", ".")
  line_numbers = arguments.get("line_numbers", False)
  if not pattern:
    return "Error: Missing parameter 'pattern'."
  return tool_search_grep(session.sandbox, pattern, path, max_results=session.max_grep_results, line_numbers=line_numbers)


def handle_run_command(arguments: Dict[str, Any], session: Any) -> str:
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
  combine_stderr = arguments.get("combine_stderr", False)
  if isinstance(combine_stderr, str):
    combine_stderr = combine_stderr.lower() in ("true", "1", "yes")
  else:
    combine_stderr = bool(combine_stderr)
  return session.tool_run_command(command, output_filter=output_filter, tail_lines=tail_lines, head_lines=head_lines, combine_stderr=combine_stderr)


def handle_check_background_command(arguments: Dict[str, Any], session: Any) -> str:
  task_id = arguments.get("task_id")
  if not task_id:
    return "Error: Missing parameter 'task_id'."
  timeout = arguments.get("timeout")
  if timeout is not None:
    try:
      timeout = float(timeout)
    except (ValueError, TypeError):
      return "Error: timeout must be a valid number."
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
  return session.tool_check_background_command(
    task_id,
    timeout=timeout,
    output_filter=output_filter,
    tail_lines=tail_lines,
    head_lines=head_lines
  )


def handle_kill_process(arguments: Dict[str, Any], session: Any) -> str:
  task_id = arguments.get("task_id")
  if not task_id:
    return "Error: Missing parameter 'task_id'."
  return session.tool_kill_process(task_id)


def handle_peek_task_output(arguments: Dict[str, Any], session: Any) -> str:
  task_id = arguments.get("task_id")
  if not task_id:
    return "Error: Missing parameter 'task_id'."
  tail_lines = arguments.get("tail_lines", 20)
  if tail_lines is not None:
    try:
      tail_lines = int(tail_lines)
    except (ValueError, TypeError):
      return "Error: tail_lines must be a valid integer."
  output_filter = arguments.get("output_filter")
  return session.tool_peek_task_output(task_id, tail_lines=tail_lines, output_filter=output_filter)


def handle_locate_files(arguments: Dict[str, Any], session: Any) -> str:
  pattern = arguments.get("pattern")
  path = arguments.get("path", ".")
  if not pattern:
    return "Error: Missing parameter 'pattern'."
  return tool_locate_files(session.sandbox, pattern, path)


def handle_get_file_info(arguments: Dict[str, Any], session: Any) -> str:
  path = arguments.get("path")
  if not path:
    return "Error: Missing parameter 'path'."
  return tool_get_file_info(session.sandbox, path)


def handle_fetch_url(arguments: Dict[str, Any], session: Any) -> str:
  url = arguments.get("url")
  if not url:
    return "Error: Missing parameter 'url'."
  return tool_fetch_url(url, max_chars=session.max_url_chars, sandbox_path=session.sandbox)


def handle_sleep(arguments: Dict[str, Any], session: Any) -> str:
  if session and getattr(session, "background_commands", None):
    active_tasks = list(session.background_commands.keys())
    return (
      f"Error: Using the 'sleep' tool is prohibited while background tasks ({', '.join(active_tasks)}) are active. "
      "To wait for background commands to progress or finish, you MUST call 'check_background_command' "
      "with the 'timeout' parameter instead."
    )
  seconds = arguments.get("seconds")
  if seconds is None:
    return "Error: Missing parameter 'seconds'."
  try:
    seconds = float(seconds)
  except (ValueError, TypeError):
    return "Error: seconds must be a valid number."
  return tool_sleep(seconds)


def handle_ask_question(arguments: Dict[str, Any], session: Any) -> str:
  question = arguments.get("question")
  if not question:
    return "Error: Missing parameter 'question'."
  options = arguments.get("options")
  multiple = bool(arguments.get("multiple", False))
  return tool_ask_question(question, options, multiple)


def handle_search_web(arguments: Dict[str, Any], session: Any) -> str:
  query = arguments.get("query")
  if not query:
    return "Error: Missing parameter 'query'."
  try:
    max_results = int(arguments.get("max_results", 10))
  except (ValueError, TypeError):
    max_results = 10
  return tool_search_web(query, max_results)


def handle_ask_oracle(arguments: Dict[str, Any], session: Any) -> str:
  query = arguments.get("query")
  if not query:
    return "Error: Missing parameter 'query'."
  return session.consult_oracle(query)


TOOL_REGISTRY: Dict[str, Callable[[Dict[str, Any], Any], str]] = {
  "move_file": handle_move_file,
  "copy_file": handle_copy_file,
  "delete_file": handle_delete_file,
  "delete_directory": handle_delete_directory,
  "make_directory": handle_make_directory,
  "run_tests": handle_run_tests,
  "list_dir": handle_list_dir,
  "read_file": handle_read_file,
  "hex_dump": handle_hex_dump,
  "write_file": handle_write_file,
  "patch_file": handle_patch_file,
  "multi_patch": handle_multi_patch,
  "edit_lines": handle_edit_lines,
  "multi_edit_lines": handle_multi_edit_lines,
  "format_file": handle_format_file,
  "search_grep": handle_search_grep,
  "run_command": handle_run_command,
  "check_background_command": handle_check_background_command,
  "kill_process": handle_kill_process,
  "peek_task_output": handle_peek_task_output,
  "locate_files": handle_locate_files,
  "get_file_info": handle_get_file_info,
  "fetch_url": handle_fetch_url,
  "sleep": handle_sleep,
  "ask_question": handle_ask_question,
  "search_web": handle_search_web,
  "ask_oracle": handle_ask_oracle,
}
