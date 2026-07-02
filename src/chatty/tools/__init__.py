from typing import List, Dict, Any, Tuple, Optional

# Import everything from the modular sub-modules
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
      "description": "Delete a file inside the sandboxed file system. Fails if the path is a directory.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The path to the file to delete relative to the sandbox root."
          }
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "delete_directory",
      "description": "Delete a directory inside the sandboxed file system.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The path to the directory to delete relative to the sandbox root."
          },
          "recursive": {
            "type": "boolean",
            "description": "If true, deletes the directory and all of its contents recursively. If false, fails if the directory is not empty. Defaults to false."
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
      "description": "Format a source code file using the appropriate formatter (e.g. black/ruff for Python, clang-format for C/C++/SystemVerilog/Verilog, prettier for JS/TS/HTML/CSS/MD, or built-in json/yaml tools). Shows a diff of changes.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The file path relative to the sandbox root."
          },
          "formatter": {
            "type": "string",
            "description": "Optional name of the formatter tool to use (e.g. 'clang-format', 'black', 'ruff', 'prettier', etc.). If omitted, chatty will auto-select the best available tool based on file extension."
          },
          "config_path": {
            "type": "string",
            "description": "Optional path to the tool-specific configuration file relative to the sandbox root (e.g. '.clang-format', 'pyproject.toml', 'prettier.config.js', etc.)."
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
      "name": "multi_edit_lines",
      "description": "Apply multiple non-contiguous line range edits to a file. The operation is atomic: if any edit fails (e.g. invalid line range or overlapping range), the entire operation is aborted. All line numbers (start_line and end_line) are 1-indexed, inclusive, and refer to the ORIGINAL content of the file before any edits are applied.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The file path relative to the sandbox root."
          },
          "edits": {
            "type": "array",
            "description": "The list of line range edits to apply. Edits refer to the original file content and must not overlap.",
            "items": {
              "type": "object",
              "properties": {
                "start_line": {
                  "type": "integer",
                  "description": "The starting line number of the range to replace in the original file (1-indexed, inclusive)."
                },
                "end_line": {
                  "type": "integer",
                  "description": "The ending line number of the range to replace in the original file (1-indexed, inclusive)."
                },
                "replacement": {
                  "type": "string",
                  "description": "The new text/code content to insert in place of the specified line range."
                }
              },
              "required": ["start_line", "end_line", "replacement"]
            }
          }
        },
        "required": ["path", "edits"]
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
      "name": "hex_dump",
      "description": "Inspect binary files by performing a hex dump or parsing slices of data into integers of various widths (8/16/32/64-bit), endianness, and signedness.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The file path relative to the sandbox root."
          },
          "start_offset": {
            "type": "integer",
            "description": "Optional starting byte offset (0-indexed). Defaults to 0."
          },
          "size": {
            "type": "integer",
            "description": "Optional number of bytes to read. Defaults to 256. Maximum is 16384."
          },
          "format": {
            "type": "string",
            "description": "Output representation. 'canonical' (hex + ASCII for 8-bit, or formatted address lines for larger widths), 'hex' (prefixed hex values e.g. 0x0f1a), 'dec' (decimal integers), 'raw' (raw continuous hex string, only valid for 8-bit). Defaults to 'canonical'."
          },
          "word_size": {
            "type": "integer",
            "description": "Group width in bits. Supported values: 8, 16, 32, 64. Defaults to 8."
          },
          "endian": {
            "type": "string",
            "description": "Byte order for multi-byte integers. Supported: 'little', 'big'. Defaults to 'little'."
          },
          "signed": {
            "type": "boolean",
            "description": "Whether to treat multi-byte integers as signed (true) or unsigned (false). Defaults to false."
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
      "description": "Search for a regular expression pattern inside files in the sandbox directory (recursively) or inside a specific file. Binary files are automatically ignored.",
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
      "description": "Execute a shell command, returning its stdout, stderr, and exit status code. The command will run with its working directory (cwd) set to the sandbox folder. WARNING: You are strictly prohibited from using this tool to list directories (use list_dir), search files (use search_grep), find files (use locate_files), view/inspect files (use read_file/get_file_info), count lines/words (use get_file_info), or pause execution (use sleep). Using commands like 'ls', 'dir', 'grep', 'find', 'cat', 'head', 'tail', 'sed', 'awk', 'less', 'more', or 'sleep' directly will fail with an error. Always use get_file_info instead of 'wc -l' to count lines in files, and use the 'sleep' tool to pause execution.",
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
          },
          "combine_stderr": {
            "type": "boolean",
            "description": "Optional. If true, standard error (stderr) is merged into standard output (stdout) and returned chronologically interleaved. Use this instead of appending '2>&1' to the command. Defaults to false."
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
          },
          "output_filter": {
            "type": "string",
            "description": "Optional regular expression pattern to filter the output. If specified, only lines matching this pattern will be returned. Pass an empty string to clear any existing filter."
          },
          "tail_lines": {
            "type": "integer",
            "description": "Optional. Only return the last N lines of the command output. Pass -1 to clear any existing tail limit."
          },
          "head_lines": {
            "type": "integer",
            "description": "Optional. Only return the first N lines of the command output. Pass -1 to clear any existing head limit."
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
      "description": "Sleep for a specified number of seconds. Do NOT use this tool to wait for background commands/tasks to progress or finish; instead, use 'check_background_command' with the 'timeout' parameter.",
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
  },
  {
    "type": "function",
    "function": {
      "name": "ask_question",
      "description": "Prompt the user with a free-form question or a list of options to select from in order to resolve ambiguity, confirm decisions, or clarify instructions.",
      "parameters": {
        "type": "object",
        "properties": {
          "question": {
            "type": "string",
            "description": "The question to present to the user."
          },
          "options": {
            "type": "array",
            "description": "Optional list of selection choices for the user to pick from.",
            "items": {
              "type": "string"
            }
          },
          "multiple": {
            "type": "boolean",
            "description": "Optional. If true, the user can select multiple options (comma-separated). Only applicable if 'options' is provided. Defaults to false."
          }
        },
        "required": ["question"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "search_web",
      "description": "Search the web for a given query and return a list of matching results with titles, URLs, and text snippets.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The search query to look up on the web."
          },
          "max_results": {
            "type": "integer",
            "description": "Optional. The maximum number of search results to return. Defaults to 10."
          }
        },
        "required": ["query"]
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
  elif name == "delete_directory":
    path = arguments.get("path")
    if not path:
      return "Error: Missing parameter 'path'."
    recursive = bool(arguments.get("recursive", False))
    return tool_delete_directory(session.sandbox, path, recursive)
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
  elif name == "hex_dump":
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
  elif name == "multi_edit_lines":
    path = arguments.get("path")
    edits = arguments.get("edits")
    if not path or not isinstance(edits, list):
      return "Error: Missing parameter 'path' or 'edits' must be a list of edit objects."
    return tool_multi_edit_lines(session.sandbox, path, edits)
  elif name == "format_file":
    path = arguments.get("path")
    formatter = arguments.get("formatter")
    config_path = arguments.get("config_path")
    if not path:
      return "Error: Missing parameter 'path'."
    return tool_format_file(session.sandbox, path, formatter, config_path)
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
    combine_stderr = arguments.get("combine_stderr", False)
    if isinstance(combine_stderr, str):
      combine_stderr = combine_stderr.lower() in ("true", "1", "yes")
    else:
      combine_stderr = bool(combine_stderr)
    return session.tool_run_command(command, output_filter=output_filter, tail_lines=tail_lines, head_lines=head_lines, combine_stderr=combine_stderr)
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
  elif name == "ask_question":
    question = arguments.get("question")
    if not question:
      return "Error: Missing parameter 'question'."
    options = arguments.get("options")
    multiple = bool(arguments.get("multiple", False))
    return tool_ask_question(question, options, multiple)
  elif name == "search_web":
    query = arguments.get("query")
    if not query:
      return "Error: Missing parameter 'query'."
    try:
      max_results = int(arguments.get("max_results", 10))
    except (ValueError, TypeError):
      max_results = 10
    return tool_search_web(query, max_results)
  else:
    return f"Error: Tool '{name}' is not recognized."
