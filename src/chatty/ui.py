import contextlib
import json
import logging
import os
from typing import Any, List, Dict

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns

from chatty.tools import TOOLS_SCHEMA
from chatty.utils import count_tokens

logger = logging.getLogger("chatty")
console = Console()


class LazyMarkdown:
  """A helper that wraps a Markdown string and only parses it when rendered.

  This prevents high CPU usage caused by parsing Markdown on every LLM token chunk.
  """

  def __init__(self, text: str):
    self.text = text

  def __rich_console__(self, console: Console, options: Any) -> Any:
    md = Markdown(self.text)
    return md.__rich_console__(console, options)

  def __rich_measure__(self, console: Console, options: Any) -> Any:
    md = Markdown(self.text)
    return md.__rich_measure__(console, options)


@contextlib.contextmanager
def optional_live(renderable, console, enabled=True, **kwargs):
  if enabled:
    from rich.live import Live
    with Live(renderable, console=console, **kwargs) as live:
      yield live
  else:
    class DummyLive:
      def update(self, *args, **kwargs):
        pass

      def stop(self):
        pass

      def start(self):
        pass
    yield DummyLive()


class ChattyCompleter(Completer):
  def __init__(self, commands):
    self.commands = sorted(commands)
    self.path_completer = PathCompleter(expanduser=True)

  def _safe_listdir(self):
    try:
      return os.listdir('.')
    except Exception:
      return []

  def get_completions(self, document, complete_event):
    text = document.text_before_cursor
    if text.startswith('/'):
      if ' ' not in text:
        for cmd in self.commands:
          if cmd.startswith(text):
            yield Completion(cmd, start_position=-len(text))
      else:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        if cmd in ('/load', '/save', '/save_session', '/load_session'):
          path_text = parts[1] if len(parts) > 1 else ""
          sub_doc = Document(path_text, cursor_position=len(path_text))
          for completion in self.path_completer.get_completions(sub_doc, complete_event):
            yield completion
    else:
      if text and not text.endswith(' ') and not text.endswith('\n'):
        words = text.split()
        if words:
          last_word = words[-1]
          if '/' in last_word or '.' in last_word or '~' in last_word or any(
              entry.startswith(last_word) for entry in self._safe_listdir()
          ):
            sub_doc = Document(last_word, cursor_position=len(last_word))
            for completion in self.path_completer.get_completions(sub_doc, complete_event):
              yield completion


def show_whitelist(session: Any):
  """Displays whitelisted paths and their permissions."""
  table = Table(title="Whitelisted Paths", show_header=True, header_style="bold magenta")
  table.add_column("Path", style="cyan")
  table.add_column("Permissions", style="green")

  if not session.allowed_ro_paths and not session.allowed_rw_paths:
    table.add_row("(No whitelisted paths)", "")
  else:
    for path in sorted(session.allowed_ro_paths):
      table.add_row(path, "Read-Only (RO)")
    for path in sorted(session.allowed_rw_paths):
      table.add_row(path, "Read-Write (RW)")

  session._print(table)


def show_help(session: Any):
  """Displays formatted CLI usage guide."""
  table = Table(title="Slash Commands", show_header=True, header_style="bold magenta")
  table.add_column("Command", style="cyan")
  table.add_column("Description", style="white")
  table.add_row("/help", "Show this help screen")
  table.add_row("/status", "Display current session configuration")
  table.add_row("/tool_stats", "Show statistics on tool and external binary calls")
  table.add_row("/provider [ollama|openrouter]", "View or switch the LLM backend provider")
  table.add_row("/model [ID|name]", "View or switch the current LLM model by ID or name")
  table.add_row("/models [add <name> | remove <ID/name>]", "List, add, or remove LLM models in the session")
  table.add_row("/oracle [name]", "View or switch the oracle model used for suggestions")
  table.add_row("/sandbox [path]", "View or change the sandbox directory path")
  table.add_row("/context [tokens]", "View or modify the history token limit")
  table.add_row("/loops [iterations]", "View or modify the max sequential tool loops limit")
  table.add_row("/api_key [key]", "Configure the OpenRouter API Key")
  table.add_row("/system [text]", "View or edit the system instructions")
  table.add_row("/load <path> [append|replace]", "Load system prompt guidelines from a file")
  table.add_row("/save_session <path>", "Save the entire session status to a JSON file")
  table.add_row("/load_session <path>", "Load a saved session status from a JSON file")
  table.add_row("/multiline", "Toggle multiline prompt input (Alt+Enter to send)")
  table.add_row("/history", "View message records and sizing details")
  table.add_row("/undo [count]", "Undo the last conversation turn(s)")
  table.add_row("/pop <index>", "Truncate history from index (1-based) onwards")
  table.add_row("/tools", "List available sandbox tools and schemas")
  table.add_row("/whitelist [add <path> [ro|rw] | remove <path> | clear]", "Manage whitelisted out-of-sandbox paths")
  table.add_row("/config [key=value]", "List, view, or change configuration parameters live")
  table.add_row("/clear / /reset", "Clear conversation memory")
  table.add_row("/compress [N]", "Summarize history, clear context, reload summary, keeping N (default 4) recent messages")
  table.add_row("/exit / /quit", "Exit the application")
  session._print(table)


def show_status(session: Any):
  """Displays configured status parameters."""
  table = Table(title="Active Session Status", show_header=False)
  table.add_column("Parameter", style="bold cyan")
  table.add_column("Value", style="green")
  table.add_row("Provider", session.provider)
  table.add_row("Model", session.model)
  table.add_row("Oracle Model", session.oracle_model or f"Not set (Default: {session.get_oracle_model()})")
  table.add_row("Sandbox Path", session.sandbox)
  table.add_row("Context Limit", f"{session.context_size} tokens")
  table.add_row("Max Loop Iterations", f"{session.max_loops} loops")
  table.add_row("API Request Delay", f"{session.api_delay} seconds")
  table.add_row("Total Messages", str(len(session.messages)))
  table.add_row("Multiline Input", "Enabled" if session.multiline_mode else "Disabled")
  session._print(table)


def show_tool_stats(session: Any):
  """Displays statistics on tool and external binary calls."""
  # Tool calls table
  tool_table = Table(title="Tool Execution Stats", show_header=True, header_style="bold yellow")
  tool_table.add_column("Tool Name", style="cyan")
  tool_table.add_column("Call Count", style="green", justify="right")

  sorted_tools = sorted(session.tool_calls_count.items(), key=lambda x: (-x[1], x[0]))
  total_tool_calls = sum(session.tool_calls_count.values())

  for name, count in sorted_tools:
    tool_table.add_row(name, str(count))

  if not sorted_tools:
    tool_table.add_row("[dim]No tools called yet[/dim]", "0")
  else:
    tool_table.add_section()
    tool_table.add_row("[bold]Total Tool Calls[/bold]", f"[bold]{total_tool_calls}[/bold]")

  # External binary table
  bin_table = Table(title="External Binary Execution Stats", show_header=True, header_style="bold magenta")
  bin_table.add_column("Binary Name", style="cyan")
  bin_table.add_column("Call Count", style="green", justify="right")

  sorted_bins = sorted(session.external_binaries_breakdown.items(), key=lambda x: (-x[1], x[0]))

  for name, count in sorted_bins:
    bin_table.add_row(name, str(count))

  if not sorted_bins:
    bin_table.add_row("[dim]No external binaries executed yet[/dim]", "0")
  else:
    bin_table.add_section()
    bin_table.add_row("[bold]Total Binary Calls[/bold]", f"[bold]{session.external_binaries_count}[/bold]")

  session._print(Columns([tool_table, bin_table], equal=False, expand=True))


def show_tools(session: Any):
  """Lists available filesystem functions."""
  table = Table(title="Available Sandboxed Tools", show_header=True, header_style="bold yellow")
  table.add_column("Tool Name", style="cyan")
  table.add_column("Description", style="white")
  sorted_tools = sorted(TOOLS_SCHEMA, key=lambda t: t["function"]["name"])
  for tool in sorted_tools:
    func = tool["function"]
    table.add_row(func["name"], func["description"])
  session._print(table)


def get_rich_status_bar(session: Any):
  """Returns a Rich Table rendering the status bar."""
  if not hasattr(session, "_cached_history_tokens") or session._cached_history_tokens is None:
    session._cached_history_tokens = session._calculate_tokens_for_messages(session.prune_history(log=False))

  total_tokens = session._cached_history_tokens

  table = Table(
    show_header=False,
    show_edge=False,
    show_lines=False,
    box=None,
    padding=0,
    expand=True,
    style="#e0e0e0 on #222222"
  )
  table.add_column()
  table.add_row(Text.from_markup(
    f" [bold]Chatty CLI[/bold] |"
    f" [bold]Provider:[/bold] [green]{session.provider}[/green] |"
    f" [bold]Model:[/bold] [yellow]{session.model}[/yellow] |"
    f" [bold]Tokens:[/bold] {total_tokens}/{session.context_size} |"
    f" [bold]Loops:[/bold] [cyan]{session.current_loop}/{session.max_loops}[/cyan] |"
    f" [bold]Sandbox:[/bold] {session.sandbox} "
  ))
  return table


def start_interactive_loop(session: Any):
  """Runs the interactive input/output CLI loop."""
  if session.headless:
    raise RuntimeError("Cannot start interactive prompt loop in headless mode.")
  # Create keybindings for multiline submissions
  kb = KeyBindings()

  @kb.add('escape', 'enter')
  def _(event):
    event.current_buffer.validate_and_handle()

  # File history tracking
  history_file = os.path.expanduser("~/.agent_chat_history")
  toolbar_style = Style.from_dict({
    'bottom-toolbar': 'bg:#222222 fg:#e0e0e0 noreverse',
  })
  completer = ChattyCompleter(session._commands.keys())
  prompt_session = PromptSession(
    history=FileHistory(history_file),
    key_bindings=kb,
    style=toolbar_style,
    completer=completer
  )

  # Display starting banner
  session._print(Panel(
    "[bold green]Welcome to the Sandboxed AI Chatbot CLI![/bold green]\n"
    "This script interfaces with Ollama and OpenRouter and restricts file write operations to the sandbox.\n"
    "Type [cyan]/help[/cyan] to display slash commands.\n"
    "Press [cyan]Ctrl+D[/cyan] or type [cyan]/exit[/cyan] to exit.",
    title="Chatty Sandboxed Chatbot",
    border_style="magenta"
  ))

  def get_bottom_toolbar():
    total_tokens = 0
    active_messages = session.prune_history(log=False)
    if active_messages:
      sys_msg = active_messages[0]
      total_tokens += count_tokens(sys_msg.get("content") or "")
      for msg in active_messages[1:]:
        content = msg.get("content") or ""
        if msg.get("tool_calls"):
          content += json.dumps(msg["tool_calls"])
        if msg.get("tool_call_id"):
          content += msg["tool_call_id"]
        total_tokens += count_tokens(content) + 12

    return HTML(
      f" <b>Chatty CLI</b> |"
      f" <b>Provider:</b> <ansigreen>{session.provider}</ansigreen> |"
      f" <b>Model:</b> <ansiyellow>{session.model}</ansiyellow> |"
      f" <b>Tokens:</b> {total_tokens}/{session.context_size} |"
      f" <b>Loops:</b> <ansicyan>{session.current_loop}/{session.max_loops}</ansicyan> |"
      f" <b>Sandbox:</b> {session.sandbox} "
    )

  session.show_status()

  while True:
    # Format interactive prompt dynamically
    multiline_indicator = " [ML]" if session.multiline_mode else ""
    prompt_html = (
      f"<ansicyan><b>AI-Sandbox</b></ansicyan> "
      f"(<ansigreen>{session.provider}</ansigreen>:<ansiyellow>{session.model}</ansiyellow>)"
      f"{multiline_indicator} &gt; "
    )

    try:
      # Read user input
      user_input = prompt_session.prompt(
        HTML(prompt_html),
        multiline=session.multiline_mode,
        bottom_toolbar=get_bottom_toolbar
      )

      # Check for empty input
      if not user_input.strip():
        continue

      # Check for slash commands
      if user_input.strip().startswith("/"):
        logger.info(f"Slash Command: {user_input.strip()}")
        should_continue = session.handle_command(user_input)
        if not should_continue:
          break
        continue

      # Append user query to history and execute loop
      logger.info(f"User Input: {user_input}")
      session.messages.append({"role": "user", "content": user_input})
      session.run_llm_cycle()

    except KeyboardInterrupt:
      # Handle Ctrl+C (clear current input or confirm exit)
      session._print("\n[yellow]KeyboardInterrupt (Ctrl+C). Type /exit to quit.[/yellow]")
    except EOFError:
      # Handle Ctrl+D
      session.cleanup_background_commands()
      session._print("\n[bold green]Goodbye![/bold green]")
      break
    except Exception as e:
      logger.exception("Unexpected error in CLI loop")
      session._print(f"[bold red]Unexpected error in CLI loop:[/bold red] {str(e)}")
