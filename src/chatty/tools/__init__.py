from typing import Dict, Any

# Import everything from the modular sub-modules to preserve public APIs
from chatty.tools.file_ops import (
  make_file_preview,
  tool_list_dir,
  tool_read_file,
  tool_get_file_info,
  tool_write_file,
  tool_patch_file,
  tool_multi_patch,
  tool_edit_lines,
  tool_multi_edit_lines,
  get_available_formatters,
  tool_format_file,
  tool_move_file,
  tool_copy_file,
  tool_delete_file,
  tool_delete_directory,
  tool_make_directory,
  tool_hex_dump
)
from chatty.tools.search_ops import (
  tool_locate_files,
  tool_search_grep
)
from chatty.tools.web_ops import (
  tool_search_web,
  tool_fetch_url
)
from chatty.tools.system_ops import (
  tool_sleep,
  tool_ask_question
)

from chatty.tools.schemas import TOOLS_SCHEMA
from chatty.tools.registry import TOOL_REGISTRY


def execute_tool(name: str, arguments: Dict[str, Any], session: Any) -> str:
  """Executes the specified tool with arguments in the sandbox directory."""
  if not hasattr(session, "tool_calls_count"):
    session.tool_calls_count = {}
  session.tool_calls_count[name] = session.tool_calls_count.get(name, 0) + 1

  handler = TOOL_REGISTRY.get(name)
  if handler:
    return handler(arguments, session)
  return f"Error: Tool '{name}' is not recognized."
