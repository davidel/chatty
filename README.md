# Chatty

An advanced AI Chatbot CLI with a rich terminal interface, sandboxed file system tools, and support for multiple LLM providers (Ollama and OpenRouter).

## Features

- **Multi-Provider Support**: Switch seamlessly between local Ollama instances and remote OpenRouter models.
- **Sandboxed Operations**: Safe execution environment with restricted file system tools.
- **Interactive UI**: Powered by `rich` and `prompt_toolkit` to provide a beautiful terminal interface, autocomplete, and history.
- **Slash Commands**: Customize your chat session on the fly (e.g., loading prompts, toggling multiline, checking token counts).
- **Custom System Prompts**: Import system prompts via YAML or raw text files.

## Folder Structure

The project has been converted to a standard Python package with a `src/` layout:

```text
chatty/
├── pyproject.toml         # Package metadata, dependencies, and entrypoints
├── README.md              # Documentation
└── src/
    └── chatty/            # Package source code
        ├── __init__.py    # Exposes the main function
        ├── __main__.py    # Enables running with `python -m chatty`
        └── cli.py         # Main chatbot application logic
```

## Installation

### Prerequisites

- Python 3.8 or higher.
- An Ollama instance running locally (optional, for local models).
- An OpenRouter API Key (optional, for cloud models).

### Installing the Package

To install `chatty` and all of its dependencies, run the following command from the root of the project:

```bash
pip install .
```

### Developer Mode (Editable Install)

If you are developing or modifying the script and want changes to reflect immediately, install it in editable mode:

```bash
pip install -e .
```

## Usage

Once installed, you can start the chatbot directly from any terminal using the `chatty` command:

```bash
chatty [options]
```

Or run it using the Python module syntax:

```bash
python -m chatty [options]
```

### CLI Arguments

- `--provider`, `-p`: The LLM backend provider to use (`ollama` or `openrouter`). Default is `ollama`.
- `--model`, `-m`: Model identifier to use. If omitted, default models will be resolved based on provider (auto-detects local Ollama models).
- `--context-size`, `-c`: Target context window length constraint in tokens (default: `8192`).
- `--sandbox`, `-s`: Path to the sandboxed file system directory. Writes are restricted to this folder (default: `./sandbox`).
- `--max-loops`, `-l`: Maximum sequential tool execution loops allowed in a single turn (default: `20`).
- `--config-prompt`, `-f`: Path to a YAML or text configuration file containing a custom system prompt.
- `--prompt-mode`, `-d`: How to apply the custom system prompt (`replace` or `integrate`). Default is `replace`.
- `--api-key`, `-a`: OpenRouter API key. Overrides the `OPENROUTER_API_KEY` environment variable.
- `--url`, `-u`: API Base URL override.

### Example Commands

Run with the default local Ollama setup:
```bash
chatty
```

Run using OpenRouter with a specific model:
```bash
chatty -p openrouter -m google/gemini-2.5-flash -a YOUR_API_KEY
```

Run with a custom sandbox directory:
```bash
chatty --sandbox ./my_safe_sandbox
```

## License

This project is licensed under the Apache License, Version 2.0.
