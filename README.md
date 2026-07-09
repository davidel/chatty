# Chatty

<p align="center">
  <img src="https://raw.githubusercontent.com/davidel/chatty/main/assets/LOGO.png" alt="Chatty" width="600">
</p>

An advanced AI Chatbot CLI with a rich terminal interface, sandboxed file system tools, multi-provider LLM support (Ollama and OpenRouter), syntax verification, dynamic skills, and cost/token optimization.

Chatty provides a local terminal loop that allows an LLM agent to interact with your codebase and system tools in a safe, sandboxed directory. It utilizes `rich` and `prompt_toolkit` to deliver a premium user experience, complete with formatting, status indicators, and autocompletion.

---

## Features

- **Multi-Provider Flexibility**: Switch seamlessly between local Ollama instances (offline models) and remote cloud-based OpenRouter endpoints.
- **Oracle Query Delegation**: Consult a more advanced oracle model dynamically during a session via the [ask_oracle](file:///tmp/chatty/src/chatty/tools/__init__.py#L668) tool or the `/oracle` slash command.
- **Sandboxed Operations**: Restricts file modifications and commands strictly to a sandbox root (default `./sandbox`), enhanced by Linux Landlock kernel-level isolation when running on Linux.
- **User-Space AST Validation**: Parses Python scripts executed via shell commands and blocks direct filesystem operations unless explicitly permitted or whitelisted by the user.
- **Interactive UI**: Status bars showing the provider, active model, token counter, active loop count, and sandbox directory path. Autocomplete and multiline inputs are fully integrated.
- **Slash Commands**: Modify settings dynamically mid-session, view tool metrics, or compress history to save tokens.
- **Dynamic & Static Skills**: Dynamically import system prompts and guidelines from external files or directories via keyword/tag triggers or statically on initialization.
- **Syntax Pre-verification**: Prevents writing broken scripts (Python, C++, Verilog, JSON, YAML) by verifying code syntax prior to saving updates.
- **Context Optimization**: Automates context truncation and history compression. Implements ephemeral `cache_control` tagging to maximize prompt caching efficiency for OpenRouter.
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
| `--model` | `-m` | string | *Auto-resolved* | Model identifier(s) to load. Can be specified multiple times or as comma-separated values. The first becomes the active model. Ollama: auto-detects first local model (falls back to `qwen2.5-coder:7b`). OpenRouter: defaults to `google/gemini-2.5-flash`. |
| `--oracle-model` | *None* | string | *Auto-resolved* | Model identifier to use as the oracle. Default determines based on provider. |
| `--context-size` | `-c` | integer | `8192` | Target context window length constraint in tokens. |
| `--sandbox` | `-s` | string | `./sandbox` | Path to the sandboxed folder. All writes and runs are jailed inside this directory. |
| `--skills-path` | `-k` | string | *None* | Custom directory paths to scan for Skills (can be specified multiple times). |
| `--whitelist` | `-w` | string | *None* | Add an out-of-sandbox path to the initial whitelist. Can end with `:ro` or `:rw` to set mode (defaults to `ro`). Can be specified multiple times. |
| `--static-skills` | *None* | flag | *Auto* | Load all skills statically into system instructions (defaults to `True` for OpenRouter, `False` for Ollama). |
| `--prompt-caching` | *None* | flag | `False` | Explicitly enable prompt caching for compatible models (adds `cache_control` tagging). |
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
| `--headless` | *None* | flag | `False` | Run the chatbot in headless mode (no console printing or terminal interactive loop). |

---

## Custom OpenAI-Compatible Providers

Since Chatty uses the standard OpenAI SDK client under the hood, you can connect to any third-party provider that offers an OpenAI-compatible endpoint. To do this, specify `--provider openrouter` and override both `--url` and `--api-key` along with your desired `--model`:

```bash
# Run using DeepSeek V3
chatty --provider openrouter --url https://api.deepseek.com --api-key YOUR_DEEPSEEK_KEY --model deepseek-chat

# Run using Groq
chatty --provider openrouter --url https://api.groq.com/openai/v1 --api-key YOUR_GROQ_KEY --model llama-3.3-70b-versatile

# Run using Together AI
chatty --provider openrouter --url https://api.together.xyz/v1 --api-key YOUR_TOGETHER_KEY --model Qwen/Qwen2.5-Coder-32B-Instruct
```

---

## Interactive Interface & Slash Commands

During a session, you can input direct queries to the model, or use **Slash Commands** to inspect/adjust configurations on the fly:

| Command | Expected Arguments | Description |
| :--- | :--- | :--- |
| `/help` | None | Displays a formatted usage table of all slash commands. |
| `/status` | None | Shows active session variables (provider, model, oracle, sandbox, tokens, loops, multiline, etc.). |
| `/tool_stats` | None | Renders execution statistics (call counts, failures, and breakdowns for tools and binaries). |
| `/provider` | `[ollama\|openrouter]` | View current provider or switch backend on the fly. Re-authenticates APIs dynamically. |
| `/model` | `[ID\|name]` | View active model name or switch to another model by name or 1-based index/ID. |
| `/models` | `[add <name>\|remove <ID\|name>]` | List currently loaded models, or add/remove them dynamically. |
| `/oracle` | `[name]` | View active oracle model name or switch to another oracle model by name. |
| `/sandbox` | `[path]` | View sandbox path or change it. Instantly loads any skills found in the new sandbox. |
| `/whitelist` / `/permissions` | `[add <path> [ro\|rw] \| remove <path> \| clear]` | View or manage whitelisted out-of-sandbox paths. |
| `/context` | `[tokens]` | View or update target context memory window limit in tokens. |
| `/loops` | `[iterations]` | View or modify the limit of sequential agent loops allowed per turn. |
| `/api_key` | `[key]` | Configure your OpenRouter cloud client token dynamically. |
| `/system` | `[text]` | Inspect or update the base system prompt instructions directly. |
| `/load` | `<path> [append\|replace]` | Read system instructions from a local YAML or text file, appending or replacing. |
| `/save` / `/save_session` | `<path>` | Save the whole status of the current conversation/session to a JSON file. |
| `/load_session` | `<path>` | Load a saved conversation/session status from a JSON file. |
| `/multiline` | None | Toggle multiline mode. When enabled, use `Alt+Enter` or `Esc+Enter` to submit. |
| `/history` | None | Renders message records, estimated token counts, roles, and tool calls. |
| `/undo` | `[count]` | Reverts the last conversation turn(s), removing assistant responses, tool outputs, and the user prompt. |
| `/pop` | `<index>` | Truncates the conversation history by deleting all messages from the specified 1-based index onwards. |
| `/tools` | None | Lists available sandboxed tools and their schema definitions. |
| `/clear` / `/reset`| None | Clears conversational context history. |
| `/compress` | `[N]` | Directs the model to summarize current conversational state using a structured format, resets older history, and keeps N (default 4) recent messages intact. |
| `/exit` / `/quit` | None | Cleanly terminates background processes and exits Chatty. |

---

## Sandboxed File System Tools

The chatbot uses function-calling to interface with the sandbox workspace. Directly invoking command-line file manipulation programs (e.g. `cat`, `grep`, `find`) in `run_command` is blocked by safety filters in [validate_command_safety](file:///tmp/chatty/src/chatty/safety.py#L168). The agent is required to use the appropriate structured tools:

### File Manipulation Tools
- **`list_dir`**: Explores directories inside the sandbox. Truncates output above `--max-dir-items` to prevent token flooding.
- **`read_file`**: Reads text files. Accepts optional `start_line` and `end_line` parameters (1-indexed), supports displaying line numbers, and honors `--max-read-chars`.
- **`write_file`**: Writes full text contents to a file. Triggers automated syntax checking.
- **`patch_file`**: Replaces a unique specific block of code. Safest tool for small, contiguous code changes.
- **`multi_patch`**: Replaces multiple non-contiguous exact blocks of code atomically.
- **`edit_lines`**: Modifies a specific line range (1-indexed, inclusive) with new text. Immune to text-matching failures.
- **`multi_edit_lines`**: Modifies multiple non-contiguous line ranges atomically using line numbers.
- **`format_file`**: Styles source files using formatters: `black`/`ruff` for Python, `clang-format` for C/C++, `prettier` for frontend, or custom JSON/YAML encoders. Displays diff results.
- **`move_file`**: Renames or moves files and directories safely inside the sandbox boundaries.
- **`copy_file`**: Recursively copies file system structures.
- **`delete_file`**: Permanently removes a file. Fails on directories.
- **`delete_directory`**: Permanently removes a directory (optionally recursively).
- **`make_directory`**: Recursively builds directory trees.
- **`get_file_info`**: Retrieves file system metadata (modification dates, sizes, type, and line counts for text documents).
- **`hex_dump`**: Performs a hex dump or parses slices of binary files into integers of various widths (8/16/32/64-bit), endianness, and signedness.

### Code Search & Diagnostics
- **`search_grep`**: Performs recursive regular expression string matching on files. Can report line numbers (`line_numbers: true`) to aid editing.
- **`locate_files`**: Finds files recursively matching glob configurations (e.g., `**/*.py`).
- **`run_tests`**: Runs test scripts (`pytest`, `npm test`, custom targets).

### Web & Information Retrieval
- **`search_web`**: Searches the web for a query and returns titles, URLs, and snippets. Supports multiple backends via environment variables (checked in priority order):
  - **Tavily**: Set `TAVILY_API_KEY` (highly recommended for clean, parsed AI search results).
  - **Brave Search**: Set `BRAVE_API_KEY` (independent, privacy-focused search).
  - **Google Custom Search**: Set `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` (legacy Google search engine).
  - **Serper**: Set `SERPER_API_KEY` (Google search proxy).
  - **SerpApi**: Set `SERPAPI_API_KEY` (Google search proxy).
  - **Yahoo Scraper**: Default fallback if no keys are provided (unreliable for heavy use).
- **`fetch_url`**: Fetches the text content of a public URL (converting HTML to clean text).

### Command & Background Execution
- **`run_command`**: Runs shell commands from the sandbox directory.
  - *Safety Restrictions*: Monitored by safety checks to block commands that attempt directory escapes, or attempt to circumvent tool guidelines by calling commands like `cat`, `grep`, `find`, `sed`, `awk`, `less`, `more`, `wc`, `kill`, `pkill`, `killall`, `cp`, `mv`, `rm`, `rmdir`, `mkdir`, `ls`, `dir`.
  - *Output Controls*: Supports `output_filter` (regex matching), `head_lines`, and `tail_lines` parameters to prevent token overflow.
  - *Asynchronous Process Execution*: Commands that block or run indefinitely are automatically backgrounded by the session, returning a `Task ID` (e.g., `task_1`).
- **`check_background_command`**: Inspects status, reads output, and checks exit status code of background processes using their `task_id`.
- **`peek_task_output`**: Peeks at the currently accumulated output of a background task without blocking or changing its running status.
- **`kill_process`**: Kills a running background subprocess.
- **`sleep`**: Pauses execution for a specified number of seconds.
- **`ask_question`**: Prompts the user with a question or selectable options to confirm decisions or resolve ambiguity.
- **`ask_oracle`**: Consults a more advanced oracle model for advice or suggestions when stuck on a difficult reasoning step, logic problem, or complex code generation.

---

## Sandbox & Landlock Security Architecture

Chatty implements a layered security approach to sandbox tool executions and protect the host system.

### 1. User-Space Safety Filtering
Every command sent to `run_command` is statically parsed and verified by [validate_command_safety](file:///tmp/chatty/src/chatty/safety.py#L168). It blocks common shell commands (such as `cat`, `grep`, `find`, `cp`, `mv`, `rm`, `ls`, etc.) to force the LLM to use structured sandbox API tools rather than arbitrary shell execution.

Additionally, Python script executions (either run inline with `python -c` or loaded from `.py` files) are parsed into an Abstract Syntax Tree (AST) to extract their structural signature. If direct filesystem operations (like `open`, `write`, `os.remove`, etc.) are detected, the execution is blocked, and the chatbot prompts the user interactively to allow or deny the script, or whitelist its signature.

### 2. Kernel-Level Linux Landlock Sandboxing
On Linux (kernel version 5.13+), Chatty provides transparent, compile-on-demand process isolation using the Linux **Landlock LSM (Linux Security Module)**. 

#### Compilation Flow
When a chatbot session is initialized (see [ChatbotSession](file:///tmp/chatty/src/chatty/session.py#L216)), Chatty checks if:
1. The operating system is Linux.
2. The `--sandbox` option is enabled.
3. GCC or Clang is installed.

If these conditions are met, [compile_landlock_binary](file:///tmp/chatty/src/chatty/landlock.py#L13) builds the C-based wrapper helper [landlock_exec.c](file:///tmp/chatty/src/chatty/landlock_exec.c) into an executable binary `landlock_exec`. The binary is cached in the package directory or under the user's cache folder (`~/.cache/chatty/`) to avoid compilation overhead on subsequent startup.

#### Restrictive Access Rules
When executing a shell command via `run_command`, the execution argument is wrapped using [wrap_command_with_landlock](file:///tmp/chatty/src/chatty/landlock.py#L71), transforming it into:
```bash
landlock_exec --ro / --rw <sandbox_dir> --rw <temp_dir> -- /bin/sh -c "<command>"
```
The [landlock_exec](file:///tmp/chatty/src/chatty/landlock_exec.c) binary interacts with the Linux kernel to apply rules before spawning the target process:
* **Read-Only Paths (`--ro`)**: Allows reading from `/` (the root filesystem). This enables reading standard system executables, Python runtimes, shared libraries (`/lib`, `/usr/lib`), and general command dependencies.
* **Read-Write Paths (`--rw`)**: Specifically permits file writes, directory creation, and removals only inside the target `<sandbox_dir>` and the system's temporary directory. Any attempts to write elsewhere on the filesystem will fail with a `Permission denied` error.
* **No New Privileges**: Configures `prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)` so that processes cannot escalate privileges.
* **Kernel Enforcement**: Enforces the ruleset using the `landlock_restrict_self` system call, sealing the sandbox rules for the current process and all future sub-processes.

#### Fallback Behavior
If Landlock is unavailable (non-Linux systems, older kernels, or missing compilers), Chatty automatically falls back to standard user-space process execution limited by the working directory (`cwd`) configuration and command regex validations.

#### Out-of-Sandbox Path Whitelisting
To allow the chatbot to access files or directories outside the sandboxed folder:
* **CLI Startup Whitelist**: Use the `--whitelist` (or `-w`) option to whitelist paths at startup.
  ```bash
  chatty -w /usr/include:ro -w /home/user/project
  ```
  Paths default to Read-Only (`ro`) unless suffixed with `:rw` (Read-Write).
* **Interactive Whitelisting**: If the agent attempts to read or write a file outside the sandbox that is not whitelisted, Chatty will prompt you interactively:
  * `[y]es`: Allow access for this operation once.
  * `[n]o`: Deny access.
  * `[a]lways`: Whitelist the specific file for the rest of the session.
  * `[p]arents`: Show a menu to select and recursively whitelist a parent directory.
* **Session Management**: Use the `/whitelist` (or `/permissions`) slash command to inspect or manage whitelisted paths on the fly.

---

## Extending Chatty with Skills

**Skills** are modular system prompt extensions that can be imported to give the LLM custom domain knowledge. A skill is a subdirectory located inside a registered skill path containing a `SKILL.md` file (for example, see [skills/greetings/SKILL.md](file:///tmp/chatty/src/chatty/skills/greetings/SKILL.md)):

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
Before sending messages to the LLM, the [prune_history](file:///tmp/chatty/src/chatty/session.py#L1161) method limits the total context:
- Old tool outputs exceeding `--max-history-tool-chars` characters are truncated in memory and annotated with a `[TRUNCATED]` note.
- The latest messages (configured via `--history-keep-messages`) are kept fully raw to ensure the model retains precise local context.

### 2. Prompt Caching
When `--prompt-caching` is explicitly enabled or when using static skills with OpenRouter, Chatty injects `cache_control: {"type": "ephemeral"}` metadata parameters:
- Applied to the system prompt message.
- Applied to the tools schema structure.
- Applied to the last two messages in the conversation queue.
This helps minimize token costs and reduces API response latencies for compatible model endpoints.

---

## Logging & Diagnostics

Chatty records session activity in a log file (default: `chatty.log`). The logging mechanism (configured in [setup_logging](file:///tmp/chatty/src/chatty/logging_setup.py#L42)) uses the [GlogFormatter](file:///tmp/chatty/src/chatty/logging_setup.py#L5) to output in standard Google logging syntax:

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

The test suite validates the following components:
- [test_caching_and_repeats.py](file:///tmp/chatty/tests/test_caching_and_repeats.py): Verifies prompt caching efficiency, EPHEMERAL headers, and handling of repeated prompts.
- [test_commands.py](file:///tmp/chatty/tests/test_commands.py): Exercises interactive slash commands (switching provider, model, modifying context parameters, system prompts).
- [test_cutoff.py](file:///tmp/chatty/tests/test_cutoff.py): Validation of message token truncation, history pruning, and prompt caching.
- [test_format.py](file:///tmp/chatty/tests/test_format.py): Verification of json, yaml, and clang-format code styling tools.
- [test_headless.py](file:///tmp/chatty/tests/test_headless.py): Validates running the chatbot session in headless mode.
- [test_landlock.py](file:///tmp/chatty/tests/test_landlock.py): Unittests for the Landlock sandboxing mechanism on Linux.
- [test_logging.py](file:///tmp/chatty/tests/test_logging.py): Checks for Google-style Logging (glog) file outputs.
- [test_oracle.py](file:///tmp/chatty/tests/test_oracle.py): Tests the oracle query delegation logic, the `ask_oracle` tool, and oracle resolution.
- [test_safety.py](file:///tmp/chatty/tests/test_safety.py): Ensures commands and processes are validated for sandboxing.
- [test_sandbox_ops.py](file:///tmp/chatty/tests/test_sandbox_ops.py): Verification of file copy, move, delete, make-dir, search, and info tools.
- [test_session_persist.py](file:///tmp/chatty/tests/test_session_persist.py): Verifies saving and loading session states to/from JSON.
- [test_tool_stats.py](file:///tmp/chatty/tests/test_tool_stats.py): Checks tracking metrics for tool and external binary usage.
- [test_tools_modularity.py](file:///tmp/chatty/tests/test_tools_modularity.py): Checks the registration and separation of modular sandbox tools.
- [test_whitelist.py](file:///tmp/chatty/tests/test_whitelist.py): Asserts correct management of out-of-sandbox directory whitelisting (ro/rw permissions) and interactive whitelist prompts.
