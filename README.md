# Chatty

An advanced AI Chatbot CLI with a rich terminal interface, sandboxed file system tools, multi-provider LLM support (Ollama and OpenRouter), syntax verification, dynamic skills, and cost/token optimization.

Chatty provides a local terminal loop that allows an LLM agent to interact with your codebase and system tools in a safe, sandboxed directory. It utilizes `rich` and `prompt_toolkit` to deliver a premium user experience, complete with formatting, status indicators, and autocompletion.

---

## Folder Structure

The project is structured as a standard Python package using a modern `src/` layout:

```text
chatty/
├── LICENSE                 # License terms (Apache 2.0)
├── pyproject.toml          # Packaging metadata, entrypoints, and dependencies
├── README.md               # User manual and technical documentation
├── src/
│   └── chatty/             # Main source package
│       ├── __init__.py     # Module initialization & top-level interface
│       ├── __main__.py     # Command-line entrypoint wrapper
│       ├── cli.py          # Primary application logic & ChatbotSession execution loop
│       └── skills/         # Default skills / extensions directory
│           └── greetings/  # Standard greetings plugin
│               ├── SKILL.md  # System instructions guidelines for greeting users
│               └── dummy.json# Metadata structure placeholder
└── tests/                  # Unittest suite
    ├── test_cutoff.py      # Validation of message token truncation, history pruning, and prompt caching
    ├── test_format.py      # Verification of json, yaml, and clang-format code styling tools
    ├── test_logging.py     # Checks for Google-style Logging (glog) file outputs
    ├── test_safety.py      # Ensures commands and processes are validated for sandboxing
    ├── test_sandbox_ops.py # Verification of file copy, move, delete, make-dir, search, and info tools
    ├── test_syntax.py      # Checks syntax parsers (Python, C++, Verilog, etc.) and compilation paths
    └── test_tool_stats.py  # Checks tracking metrics for tool and external binary usage
```

### Main Code Components
- [pyproject.toml](file:///tmp/chatty/pyproject.toml): Defines Python package requirements and targets [cli.py:main](file:///tmp/chatty/src/chatty/cli.py#L3034) as the execution binary entrypoint.
- [cli.py](file:///tmp/chatty/src/chatty/cli.py): Contains the core [ChatbotSession](file:///tmp/chatty/src/chatty/cli.py#L1696) logic, input loops, slash commands, LLM provider clients, and tool execution routines.
- [__init__.py](file:///tmp/chatty/src/chatty/__init__.py): Exposes the package API.
- [__main__.py](file:///tmp/chatty/src/chatty/__main__.py): Enables execution using `python3 -m chatty`.

---

## Features

- **Multi-Provider Flexibility**: Switch seamlessly between local Ollama instances (offline models) and remote cloud-based OpenRouter endpoints.
- **Sandboxed Operations**: Restricts file modifications and commands strictly to a sandbox root (default `./sandbox`).
- **Interactive UI**: Status bars showing the provider, active model, token counter, active loop count, and sandbox directory path. Autocomplete and multiline inputs are fully integrated.
- **Slash Commands**: Modify settings dynamically mid-session, view tool metrics, or compress history to save tokens.
- **Dynamic & Static Skills**: Dynamically import system prompts and guidelines from external files or directories via keyword/tag triggers or statically on initialization.
- **Syntax Pre-verification**: Prevents writing broken scripts (Python, C++, Verilog, JSON, YAML) by verifying code syntax prior to saving updates.
- **Context Optimization**: Automates context truncation and history compression. Implements `cache_control` to maximize OpenRouter prompt caching efficiency.
- **Google-Style Logging**: Standardized process tracking via custom `glog` styling.

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher.
- A running Ollama instance locally (optional, for local models).
- An OpenRouter API Key (optional, for remote models).

### Installation
Install the package and all dependencies (defined in [pyproject.toml](file:///tmp/chatty/pyproject.toml)) from the root folder:

```bash
pip install .
```

### Developer Mode (Editable Install)
To develop or modify the codebase and have changes immediately visible without reinstalling:

```bash
pip install -e .
```

---

## CLI Usage & Arguments

Once installed, invoke the chatbot using the `chatty` command or via Python:

```bash
chatty [options]
# OR
python3 -m chatty [options]
```

### Full List of CLI Arguments

| Parameter | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--provider` | `-p` | string | `ollama` | Backend provider to use (`ollama` or `openrouter`). |
| `--model` | `-m` | string | *Auto-resolved* | Model identifier. Ollama: auto-detects first local model (falls back to `qwen2.5-coder:7b`). OpenRouter: defaults to `google/gemini-2.5-flash`. |
| `--context-size` | `-c` | integer | `8192` | Target context window length constraint in tokens. |
| `--sandbox` | `-s` | string | `./sandbox` | Path to the sandboxed folder. All writes and runs are jailed inside this directory. |
| `--skills-path` | `-k` | string | *None* | Custom directory paths to scan for Skills (can be specified multiple times). |
| `--static-skills` | *None* | bool | *Auto* | Load all skills statically into system instructions (defaults to `True` for OpenRouter to maximize cache hit rates, `False` for Ollama). |
| `--max-loops` | `-l` | integer | `20` | Maximum sequential tool executions allowed in a single user turn. |
| `--config-prompt` | `-f` | string | *None* | Path to a YAML or plain text file containing custom system prompt guidelines. |
| `--prompt-mode` | `-d` | string | `replace` | How to apply custom system prompt configuration (`replace` default prompt, or `integrate`/append to it). |
| `--api-key` | `-a` | string | *None* | OpenRouter API Key. Overrides `OPENROUTER_API_KEY` environment variable. |
| `--url` | `-u` | string | *None* | Custom Base URL override for Ollama/OpenRouter APIs. |
| `--max-read-chars` | *None* | integer | `40000` | Maximum character limit when reading a text file to prevent context explosion. |
| `--max-grep-results`| *None* | integer | `100` | Limit on matching results returned from the regular expression search tool. |
| `--max-command-chars`|*None* | integer | `16000` | Maximum characters returned from stdout/stderr of executing commands. |
| `--max-history-tool-chars`|*None*| integer| `1000` | Token saving limit: compresses old/historical tool output messages to this size. |
| `--history-keep-messages`| *None*| integer| `4` | Number of recent messages to keep fully raw and uncompressed. |
| `--max-url-chars` | *None* | integer | `24000` | Limit on fetched website character outputs. |
| `--max-dir-items` | *None* | integer | `200` | Maximum number of directory items listed by directory explorer tool. |
| `--log-file` | *None* | string | `chatty.log` | File path where execution statements are logged. Set to `""` to disable. |
| `--log-level` | *None* | string | `info` | Logging verbosity (`debug`, `info`, `warning`, `error`). |

---

## Interactive Interface & Slash Commands

During a session, you can input direct queries to the model, or use **Slash Commands** to inspect/adjust configurations on the fly:

| Command | Expected Arguments | Description |
| :--- | :--- | :--- |
| `/help` | None | Displays a formatted usage table of all slash commands. |
| `/status` | None | Shows active session variables (provider, model, sandbox, tokens, loops, multiline, etc.). |
| `/tool_stats` | None | Renders execution statistics (call counts, failures, and breakdowns for tools and binaries). |
| `/provider` | `[ollama\|openrouter]` | View current provider or switch backend on the fly. Re-authenticates APIs dynamically. |
| `/model` | `[name]` | View active model name or switch to another available model. |
| `/sandbox` | `[path]` | View sandbox path or change it. Instantly loads any skills found in the new sandbox. |
| `/context` | `[tokens]` | View or update target context memory window limit in tokens. |
| `/loops` | `[iterations]` | View or modify the limit of sequential agent loops allowed per turn. |
| `/api_key` | `[key]` | Configure your OpenRouter cloud client token dynamically. |
| `/system` | `[text]` | Inspect or update the base system prompt instructions directly. |
| `/load` | `<path> [append\|replace]` | Read system instructions from a local YAML or text file, appending or replacing. |
| `/multiline` | None | Toggle multiline mode. When enabled, use `Alt+Enter` or `Esc+Enter` to submit. |
| `/history` | None | Renders message records, estimated token counts, roles, and tool calls. |
| `/tools` | None | Lists available sandboxed tools and their schema definitions. |
| `/clear` / `/reset`| None | Clears conversational context history. |
| `/compress` | None | Directs the model to summarize current conversational state, resets history, and appends summary. |
| `/exit` / `/quit` | None | Cleanly terminates background processes and exits Chatty. |

---

## Sandboxed File System Tools

The chatbot uses function-calling to interface with the sandbox workspace. Directly invoking command-line file manipulation programs (e.g. `cat`, `grep`, `find`) in `run_command` is blocked by safety filters in [validate_command_safety](file:///tmp/chatty/src/chatty/cli.py#L1946). The agent is required to use the appropriate structured tools:

### File Manipulation Tools
- **`list_dir`**: Explores directories inside the sandbox. Truncates output above `--max-dir-items` to prevent token flooding.
- **`read_file`**: Reads text files. Accepts optional `start_line` and `end_line` parameters (1-indexed) and honors `--max-read-chars`.
- **`write_file`**: Writes full text contents to a file. Triggers automated syntax checking.
- **`patch_file`**: Replaces a unique specific block of code. Safest tool for small, contiguous code changes.
- **`edit_lines`**: Modifies a specific line range (1-indexed, inclusive) with new text. Immune to text-matching failures.
- **`format_file`**: Styles source files using formatters: `black`/`ruff` for Python, `clang-format` for C/C++, `prettier` for frontend, or custom JSON/YAML encoders. Displays diff results.
- **`move_file`**: Renames or moves files and directories safely inside the sandbox boundaries.
- **`copy_file`**: Recursively copies file system structures.
- **`delete_file`**: Permanently removes files or directories.
- **`make_directory`**: Recursively builds directory trees.
- **`get_file_info`**: Retrieves file system metadata (modification dates, sizes, line counts for text documents).

### Code Search & Diagnostics
- **`search_grep`**: Performs recursive regular expression string matching on files. Can report line numbers (`line_numbers: true`) to aid editing.
- **`locate_files`**: Finds files recursively matching glob configurations (e.g., `**/*.py`).
- **`run_tests`**: Runs test scripts (`pytest`, `npm test`, custom targets).

### Command & Background Execution
- **`run_command`**: Runs shell commands from the sandbox directory.
  - *Safety Restrictions*: Monitored by safety checks to block commands that attempt directory escapes, or attempt to circumvent tool guidelines by calling commands like `cat`, `grep`, `find`, `sed`, `awk`, `less`, `more`, `wc`, `kill`, `pkill`, `killall`.
  - *Output Controls*: Supports `output_filter` (regex matching), `head_lines`, and `tail_lines` parameters to prevent token overflow.
  - *Asynchronous Process Execution*: Commands that block or run indefinitely are automatically backgrounded by the session, returning a `Task ID` (e.g., `task_1`).
- **`check_background_command`**: Inspects status or reads output of background processes using their `task_id`.
- **`kill_process`**: Kills a running background subprocess.

---

## Syntax Verification & Dependencies

Updating code via `write_file`, `patch_file`, or `edit_lines` calls the [validate_file_syntax](file:///tmp/chatty/src/chatty/cli.py#L413) helper before saving. If syntax errors occur, the tool blocks the file modifications and returns compiler/lint messages back to the model:
1. **Python**: Parsed using the python standard `ast` compiler.
2. **JSON & YAML**: Checked via `json.loads` and `yaml.safe_load`.
3. **C & C++**: Syntactically validated using local compilation tools.
4. **Verilog & SystemVerilog**: Checked using system-installed linting tools like `verilator` or `iverilog`.

### Resolving Dependencies with `compile_paths`
For languages that compile or check references across files (e.g., C/C++, Verilog), the write tools accept an optional `compile_paths` list parameter. You can specify header include folders, library directories, or explicit source dependencies relative to the sandbox:
```json
{
  "path": "src/top.sv",
  "search": "...old...",
  "replace": "...new...",
  "compile_paths": ["deps/my_dep.sv", "include/"]
}
```

---

## Extending Chatty with Skills

**Skills** are modular system prompt extensions that can be imported to give the LLM custom domain knowledge. A skill is a subdirectory located inside a registered skill path containing a [SKILL.md](file:///tmp/chatty/src/chatty/skills/greetings/SKILL.md) file:

```text
my_skills_dir/
└── database_guidelines/
    ├── SKILL.md
    └── sample_schema.sql
```

### SKILL.md Structure
The `SKILL.md` must start with YAML frontmatter:
```markdown
---
name: Database Operations
description: Instructions for sql query formatting and schemas.
tags: [sql, postgres, sqlite, db]
---
Always format queries in uppercase... (Rest of system guidelines)
```

### Loading Skills
Skills are compiled from multiple locations:
1. Default system path: `src/chatty/skills/`
2. Environment Variable: `CHATBOT_SKILLS_PATH` (colon/path-separated values)
3. Dynamic arguments: `--skills-path` / `-k` CLI options.
4. Local sandbox path: `<sandbox>/skills/`

### Activation Modes
- **Static Mode (`--static-skills`)**: Combines all loaded skills directly into the base system prompt. This ensures all rules are known at all times and optimizes prompt caching for APIs like OpenRouter.
- **Dynamic Mode (Default for Ollama)**: Scans the user prompt for the skill's `name` or any of its `tags`. If a match is found (case-insensitive), the skill's instructions are appended to the system instructions specifically for that conversational turn.

---

## Context Optimization

To operate efficiently over long conversations, Chatty implements two main context optimization mechanisms:

### 1. Smart History Pruning
Before sending messages to the LLM, the [prune_history](file:///tmp/chatty/src/chatty/cli.py#L2245) method limits the total context:
- Old tool outputs exceeding `--max-history-tool-chars` characters are truncated in memory and annotated with a `[TRUNCATED]` note.
- The latest messages (configured via `--history-keep-messages`) are kept fully raw to ensure the model retains precise local context.

### 2. Prompt Caching (OpenRouter)
For the OpenRouter provider, prompt caching is configured using `cache_control: {"type": "ephemeral"}` parameters:
- Applied to system prompt messages.
- Applied to the tools schema structure.
- Applied to the last two messages in the conversation queue.
This helps minimize token costs and reduces API response latencies.

---

## Logging & Diagnostics

Chatty records session activity in a log file (default: `chatty.log`). The logging mechanism (configured in [setup_logging](file:///tmp/chatty/src/chatty/cli.py#L3021)) uses the [GlogFormatter](file:///tmp/chatty/src/chatty/cli.py#L2984) to output in standard Google logging syntax:

`Lyyyymmdd hh:mm:ss.uuuuuu process file:line] message`

- **`L`**: Level identifier letter (`D` for Debug, `I` for Info, `W` for Warning, `E` for Error, `F` for Critical/Fatal).
- **`yyyyymmdd hh:mm:ss.uuuuuu`**: Precise timestamps.
- **`process`**: OS Process ID.
- **`file:line`**: Source code location reporting the logging output.

*To disable file logging, run with `--log-file ""`.*

---

## Testing

Chatty comes with a comprehensive unittest suite located in `tests/`. You can execute tests from the project root:

```bash
python3 -m unittest discover tests
```

Tests cover:
- Sandbox operation security ([test_safety.py](file:///tmp/chatty/tests/test_safety.py))
- File layout tools ([test_sandbox_ops.py](file:///tmp/chatty/tests/test_sandbox_ops.py))
- AST syntax validations ([test_syntax.py](file:///tmp/chatty/tests/test_syntax.py))
- Token truncation limits ([test_cutoff.py](file:///tmp/chatty/tests/test_cutoff.py))
- Log formatting schemas ([test_logging.py](file:///tmp/chatty/tests/test_logging.py))
- Formatter execution ([test_format.py](file:///tmp/chatty/tests/test_format.py))
- Session metrics ([test_tool_stats.py](file:///tmp/chatty/tests/test_tool_stats.py))
