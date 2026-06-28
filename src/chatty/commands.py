import json
import os
from typing import Dict, Callable, Any

from rich.console import Console
from rich.panel import Panel
from chatty.utils import load_system_prompt_from_file, count_tokens

console = Console()


def cmd_exit(session: Any, arg: str) -> bool:
  session.cleanup_background_commands()
  console.print("[bold green]Goodbye![/bold green]")
  return False


def cmd_clear(session: Any, arg: str) -> bool:
  session.messages.clear()
  console.print("[bold green]Conversation history cleared.[/bold green]")
  return True


def cmd_compress(session: Any, arg: str) -> bool:
  session.compress_context()
  return True


def cmd_help(session: Any, arg: str) -> bool:
  session.show_help()
  return True


def cmd_status(session: Any, arg: str) -> bool:
  session.show_status()
  return True


def cmd_tool_stats(session: Any, arg: str) -> bool:
  session.show_tool_stats()
  return True


def cmd_provider(session: Any, arg: str) -> bool:
  if not arg:
    console.print(f"Current provider: [bold cyan]{session.provider}[/bold cyan]")
  elif arg in ("ollama", "openrouter"):
    session.provider = arg
    session.init_client()
    console.print(f"Switched provider to: [bold green]{session.provider}[/bold green]")
  else:
    console.print("[bold red]Error: Provider must be 'ollama' or 'openrouter'.[/bold red]")
  return True


def cmd_model(session: Any, arg: str) -> bool:
  if not arg:
    console.print(f"Current model: [bold cyan]{session.model}[/bold cyan]")
  else:
    session.model = arg
    console.print(f"Model updated to: [bold green]{session.model}[/bold green]")
  return True


def cmd_sandbox(session: Any, arg: str) -> bool:
  if not arg:
    console.print(f"Current sandbox path: [bold cyan]{session.sandbox}[/bold cyan]")
  else:
    abs_p = os.path.abspath(arg)
    os.makedirs(abs_p, exist_ok=True)
    session.sandbox = abs_p
    session.load_skills()
    console.print(f"Sandbox updated to: [bold green]{session.sandbox}[/bold green]")
  return True


def cmd_context(session: Any, arg: str) -> bool:
  if not arg:
    console.print(f"Current context size: [bold cyan]{session.context_size}[/bold cyan] tokens")
  else:
    try:
      session.context_size = int(arg)
      console.print(f"Context size updated to: [bold green]{session.context_size}[/bold green] tokens")
    except ValueError:
      console.print("[bold red]Error: Context size must be an integer.[/bold red]")
  return True


def cmd_loops(session: Any, arg: str) -> bool:
  if not arg:
    console.print(f"Current max loop limit: [bold cyan]{session.max_loops}[/bold cyan]")
  else:
    try:
      session.max_loops = int(arg)
      console.print(f"Max loop limit updated to: [bold green]{session.max_loops}[/bold green]")
    except ValueError:
      console.print("[bold red]Error: Max loops must be an integer.[/bold red]")
  return True


def cmd_api_key(session: Any, arg: str) -> bool:
  if not arg:
    console.print("API Key: [dim](hidden)[/dim]")
  else:
    session.api_key = arg
    session.init_client()
    console.print("[bold green]API key updated successfully.[/bold green]")
  return True


def cmd_multiline(session: Any, arg: str) -> bool:
  session.multiline_mode = not session.multiline_mode
  status = "enabled" if session.multiline_mode else "disabled"
  console.print(f"Multiline mode [bold cyan]{status}[/bold cyan].")
  if session.multiline_mode:
    console.print("[dim]Use Alt+Enter or Esc+Enter to submit message.[/dim]")
  return True


def cmd_system(session: Any, arg: str) -> bool:
  if not arg:
    console.print(Panel(session.system_prompt, title="Current System Prompt", border_style="cyan"))
  else:
    session.system_prompt = arg
    console.print("[bold green]System prompt updated.[/bold green]")
  return True


def cmd_load(session: Any, arg: str) -> bool:
  if not arg:
    console.print("[bold red]Error: Usage: /load <file_path> [append|replace][/bold red]")
  else:
    parts = arg.strip().rsplit(maxsplit=1)
    opt = "append"
    file_path = arg.strip()
    if len(parts) == 2 and parts[1].lower() in ("append", "replace"):
      file_path = parts[0].strip()
      opt = parts[1].lower()
    file_path = os.path.expanduser(file_path)
    try:
      loaded_prompt = load_system_prompt_from_file(file_path)
      if opt == "replace":
        session.system_prompt = loaded_prompt
        console.print(f"[bold green]System prompt replaced with content from {file_path}[/bold green]")
      else:
        session.system_prompt += f"\n\n{loaded_prompt}"
        console.print(f"[bold green]Appended prompt content from {file_path} to system prompt.[/bold green]")
    except Exception as e:
      console.print(f"[bold red]Error loading prompt file: {str(e)}[/bold red]")
  return True


def cmd_save(session: Any, arg: str) -> bool:
  if not arg:
    console.print("[bold red]Error: Usage: /save_session <file_path>[/bold red]")
  else:
    file_path = os.path.expanduser(arg.strip())
    if not os.path.isabs(file_path):
      file_path = os.path.join(session.sandbox, file_path)
    dir_name = os.path.dirname(file_path)
    if dir_name:
      os.makedirs(dir_name, exist_ok=True)
    session_data = {
      "provider": session.provider,
      "model": session.model,
      "context_size": session.context_size,
      "sandbox": session.sandbox,
      "max_loops": session.max_loops,
      "system_prompt": session.system_prompt,
      "messages": session.messages,
      "tool_calls_count": session.tool_calls_count,
      "external_binaries_count": session.external_binaries_count,
      "external_binaries_breakdown": session.external_binaries_breakdown,
    }
    if session.api_key:
      session_data["api_key"] = session.api_key
    if session.url:
      session_data["url"] = session.url
    try:
      with open(file_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, default=str)
      console.print(f"[bold green]Session saved successfully to {file_path}[/bold green]")
    except Exception as e:
      console.print(f"[bold red]Error saving session: {str(e)}[/bold red]")
  return True


def cmd_load_session(session: Any, arg: str) -> bool:
  if not arg:
    console.print("[bold red]Error: Usage: /load_session <file_path>[/bold red]")
  else:
    file_path = os.path.expanduser(arg.strip())
    if not os.path.isabs(file_path):
      file_path = os.path.join(session.sandbox, file_path)
    try:
      with open(file_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)
      if "provider" in session_data:
        session.provider = session_data["provider"]
      if "model" in session_data:
        session.model = session_data["model"]
      if "context_size" in session_data:
        session.context_size = session_data["context_size"]
      if "sandbox" in session_data:
        sandbox_path = os.path.abspath(session_data["sandbox"])
        if os.path.exists(sandbox_path):
          session.sandbox = sandbox_path
      if "max_loops" in session_data:
        session.max_loops = session_data["max_loops"]
      if "system_prompt" in session_data:
        session.system_prompt = session_data["system_prompt"]
      if "messages" in session_data:
        session.messages = session_data["messages"]
      if "tool_calls_count" in session_data:
        session.tool_calls_count = session_data["tool_calls_count"]
      if "external_binaries_count" in session_data:
        session.external_binaries_count = session_data["external_binaries_count"]
      if "external_binaries_breakdown" in session_data:
        session.external_binaries_breakdown = session_data["external_binaries_breakdown"]
      if "api_key" in session_data:
        session.api_key = session_data["api_key"]
      if "url" in session_data:
        session.url = session_data["url"]
      session.init_client()
      session.load_skills()
      console.print(f"[bold green]Session loaded successfully from {file_path}[/bold green]")
    except Exception as e:
      console.print(f"[bold red]Error loading session: {str(e)}[/bold red]")
  return True


def cmd_tools(session: Any, arg: str) -> bool:
  session.show_tools()
  return True


def cmd_history(session: Any, arg: str) -> bool:
  console.print("[bold cyan]Conversation History (estimated tokens):[/bold cyan]")
  for idx, msg in enumerate(session.messages):
    role = msg["role"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    display_text = ""
    if reasoning:
      display_text += f"[Thinking: {reasoning[:60]}...]\n"
    display_text += content
    if "tool_calls" in msg:
      display_text += f"\n[Calls tools: {[tc['function']['name'] for tc in msg['tool_calls']]}]"
    tok = count_tokens(content)
    if reasoning:
      tok += count_tokens(reasoning)
    console.print(f" {idx + 1}. [bold]{role}[/bold]: {display_text[:80].replace('\n', ' ')}... ({tok} tokens)")
  return True


def cmd_undo(session: Any, arg: str) -> bool:
  try:
    count = int(arg.strip()) if arg.strip() else 1
  except ValueError:
    console.print("[bold red]Error: Undo count must be an integer.[/bold red]")
    return True

  if count < 1:
    console.print("[bold red]Error: Undo count must be at least 1.[/bold red]")
    return True

  undone_turns = 0
  for _ in range(count):
    popped_assistant_tool = 0
    while session.messages and session.messages[-1].get("role") in ("tool", "assistant"):
      session.messages.pop()
      popped_assistant_tool += 1
    if session.messages and session.messages[-1].get("role") == "user":
      user_msg = session.messages.pop()
      content = user_msg.get("content") or ""
      console.print(f"[bold green]Undone turn {undone_turns + 1}:[/bold green] Popped {popped_assistant_tool} assistant/tool messages and user prompt: '[yellow]{content}[/yellow]'")
      undone_turns += 1
    else:
      if popped_assistant_tool > 0:
        console.print(f"[bold green]Undone turn {undone_turns + 1}:[/bold green] Popped {popped_assistant_tool} assistant/tool messages (no user prompt found).")
        undone_turns += 1
      else:
        break

  if undone_turns == 0:
    console.print("[bold yellow]History is empty or has no messages to undo.[/bold yellow]")
  return True


def cmd_pop(session: Any, arg: str) -> bool:
  if not arg.strip():
    console.print("[bold red]Error: Usage: /pop <index>[/bold red]")
    return True

  try:
    index = int(arg.strip())
  except ValueError:
    console.print("[bold red]Error: Message index must be an integer.[/bold red]")
    return True

  total = len(session.messages)
  if index < 1 or index > total:
    console.print(f"[bold red]Error: Message index must be between 1 and {total}.[/bold red]")
    return True

  pop_start = index - 1
  popped_messages = session.messages[pop_start:]
  session.messages = session.messages[:pop_start]
  console.print(f"[bold green]Truncated history.[/bold green] Popped {len(popped_messages)} messages from index {index} onwards.")
  return True


COMMANDS: Dict[str, Callable[[Any, str], bool]] = {
  "/exit": cmd_exit,
  "/quit": cmd_exit,
  "/clear": cmd_clear,
  "/reset": cmd_clear,
  "/compress": cmd_compress,
  "/help": cmd_help,
  "/status": cmd_status,
  "/tool_stats": cmd_tool_stats,
  "/provider": cmd_provider,
  "/model": cmd_model,
  "/sandbox": cmd_sandbox,
  "/context": cmd_context,
  "/loops": cmd_loops,
  "/api_key": cmd_api_key,
  "/multiline": cmd_multiline,
  "/system": cmd_system,
  "/load": cmd_load,
  "/save": cmd_save,
  "/save_session": cmd_save,
  "/load_session": cmd_load_session,
  "/tools": cmd_tools,
  "/history": cmd_history,
  "/undo": cmd_undo,
  "/pop": cmd_pop,
}

