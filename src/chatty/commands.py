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
  arg = arg.strip()
  keep_messages = None
  if arg:
    try:
      keep_messages = int(arg)
      if keep_messages < 0:
        console.print("[bold red]Error: Number of messages to keep must be non-negative.[/bold red]")
        return True
    except ValueError:
      console.print("[bold red]Error: Invalid argument for /compress. Must be an integer representing N messages to keep.[/bold red]")
      return True
  session.compress_context(keep_messages=keep_messages)
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
  arg = arg.strip()
  if not arg:
    console.print(f"Current provider: [bold cyan]{session.provider}[/bold cyan]")
  else:
    session.provider = arg
    session.init_client()
    session.update_available_models_async()
    console.print(f"Switched provider to: [bold green]{session.provider}[/bold green]")
    if arg not in ("ollama", "openrouter"):
      console.print("[bold yellow]Remember:[/bold yellow] Use '/url <api_url>' and '/api_key <key>' to configure your custom endpoint.")
  return True


def cmd_model(session: Any, arg: str) -> bool:
  arg = arg.strip()
  if not arg:
    console.print(f"Current model: [bold cyan]{session.model}[/bold cyan]")
    return True

  # Try to parse as integer (1-based ID)
  try:
    idx = int(arg)
    if 1 <= idx <= len(session.models):
      new_model = session.models[idx - 1]
      session.model = new_model
      console.print(f"Switched model to: [bold green]{new_model}[/bold green] (ID: {idx})")
    else:
      console.print(f"[bold red]Error: Invalid model ID '{arg}'. Available IDs: 1 to {len(session.models)}.[/bold red]")
  except ValueError:
    # Treat as model name
    if arg in session.models:
      session.model = arg
      idx = session.models.index(arg) + 1
      console.print(f"Switched model to: [bold green]{arg}[/bold green] (ID: {idx})")
    else:
      session.models.append(arg)
      session.model = arg
      console.print(f"Added and switched model to: [bold green]{arg}[/bold green] (ID: {len(session.models)})")
  return True


def cmd_oracle(session: Any, arg: str) -> bool:
  arg = arg.strip()
  if not arg:
    oracle = session.get_oracle_model()
    if oracle:
      console.print(f"Current oracle model: [bold cyan]{oracle}[/bold cyan]")
    else:
      console.print("Current oracle model is not configured (No oracle tool is active).")
    return True

  session.oracle_model = arg
  console.print(f"Switched oracle model to: [bold green]{arg}[/bold green]")
  return True


def cmd_models(session: Any, arg: str) -> bool:
  parts = arg.strip().split(maxsplit=1)
  if not parts:
    # List current models
    if not hasattr(session, "models") or not session.models:
      console.print("[bold yellow]No models configured.[/bold yellow]")
      return True
    
    from rich.table import Table
    table = Table(title="Configured Models", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Active", style="green", justify="center")
    table.add_column("Model Name", style="white")
    
    for idx, m in enumerate(session.models):
      is_active = "[bold green]*[/bold green]" if m == session.model else ""
      table.add_row(str(idx + 1), is_active, m)
    
    console.print(table)
    console.print("\n[bold]Usage:[/bold]")
    console.print("  [cyan]/model <ID>[/cyan] - Switch to model by ID")
    console.print("  [cyan]/models add <model_name>[/cyan] - Add a new model")
    console.print("  [cyan]/models remove <ID or model_name>[/cyan] - Remove a model")
    console.print("  [cyan]/models available [--refresh][/cyan] - List available models from active provider")
    console.print("  [cyan]/models search <query>[/cyan] - Search available models from active provider")
    return True

  subcmd = parts[0].lower()
  if subcmd == "add":
    if len(parts) < 2:
      console.print("[bold red]Error: Usage: /models add <model_name>[/bold red]")
      return True
    model_name = parts[1].strip()
    if model_name in session.models:
      console.print(f"[bold yellow]Model '{model_name}' is already in the list.[/bold yellow]")
    else:
      session.models.append(model_name)
      console.print(f"[bold green]Added model:[/bold green] {model_name} (ID: {len(session.models)})")
  elif subcmd in ("remove", "delete", "rm"):
    if len(parts) < 2:
      console.print("[bold red]Error: Usage: /models remove <ID or model_name>[/bold red]")
      return True
    if len(session.models) <= 1:
      console.print("[bold red]Error: Cannot remove the last model. At least one model must remain.[/bold red]")
      return True
    target = parts[1].strip()
    
    # Try to parse target as ID
    removed_model = None
    try:
      idx = int(target)
      if 1 <= idx <= len(session.models):
        removed_model = session.models.pop(idx - 1)
      else:
        console.print(f"[bold red]Error: Invalid model ID '{target}'. Available IDs: 1 to {len(session.models)}.[/bold red]")
        return True
    except ValueError:
      # Treat target as model name
      if target in session.models:
        session.models.remove(target)
        removed_model = target
      else:
        console.print(f"[bold red]Error: Model '{target}' not found in list.[/bold red]")
        return True
    
    if removed_model:
      console.print(f"[bold green]Removed model:[/bold green] {removed_model}")
      # If we removed the active model, switch to another one
      if session.model == removed_model:
        session.model = session.models[0]
        console.print(f"Active model switched to: [bold green]{session.model}[/bold green]")
  elif subcmd == "available":
    force_refresh = False
    if len(parts) > 1:
      force_refresh = "--refresh" in parts[1] or "-r" in parts[1]
    
    if force_refresh or not getattr(session, "available_models", None):
      console.print("[yellow]Refreshing available models list...[/yellow]")
      from chatty.llm import fetch_available_models
      session.available_models = fetch_available_models(session, force_refresh=True)
      
    if not getattr(session, "available_models", None):
      console.print("[bold red]No available models found or failed to query provider.[/bold red]")
      return True
      
    from rich.table import Table
    table = Table(title=f"Available Models from {session.provider.upper()}", show_header=True, header_style="bold magenta")
    
    if session.provider == "openrouter":
      table.add_column("Model ID", style="cyan")
      table.add_column("Name", style="white")
      table.add_column("Context Length", style="green", justify="right")
      table.add_column("Input (per 1M)", style="yellow", justify="right")
      table.add_column("Output (per 1M)", style="yellow", justify="right")
      
      # Top 15 most popular models from the fetched registry
      for m in session.available_models[:15]:
        table.add_row(
          m.get("id", ""), 
          m.get("name", ""), 
          f"{m['context']:,}" if m.get("context") else "Unknown",
          f"${m.get('pricing_input', 0):.2f}",
          f"${m.get('pricing_output', 0):.2f}"
        )
      console.print(table)
      console.print("\n[dim]Only displaying the top 15 popular models. Search for other models using '/models search <query>'.[/dim]")
      
    elif session.provider == "ollama":
      table.add_column("Model Tag", style="cyan")
      table.add_column("Name", style="white")
      table.add_column("Size", style="green", justify="right")
      table.add_column("Quantization", style="yellow")
      
      for m in session.available_models:
        size_gb = m.get("size", 0) / (1024**3)
        quant = m.get("details", {}).get("quantization_level", "Unknown")
        table.add_row(m.get("id", ""), m.get("name", ""), f"{size_gb:.2f} GB", quant)
      console.print(table)
      
  elif subcmd == "search":
    if len(parts) < 2:
      console.print("[bold red]Error: Usage: /models search <query>[/bold red]")
      return True
      
    query = parts[1].strip().lower()
    results = [m for m in getattr(session, "available_models", []) if query in m.get("id", "").lower() or query in m.get("name", "").lower()]
    
    if not results:
      console.print(f"[yellow]No models matching '{query}' found.[/yellow]")
      return True
      
    from rich.table import Table
    table = Table(title=f"Search Results for '{query}'", show_header=True, header_style="bold magenta")
    
    if session.provider == "openrouter":
      table.add_column("Model ID", style="cyan")
      table.add_column("Name", style="white")
      table.add_column("Context Length", style="green", justify="right")
      table.add_column("Input (per 1M)", style="yellow", justify="right")
      table.add_column("Output (per 1M)", style="yellow", justify="right")
      
      for m in results[:25]: # Limit to top 25 results to avoid terminal spam
        table.add_row(
          m.get("id", ""), 
          m.get("name", ""), 
          f"{m['context']:,}" if m.get("context") else "Unknown",
          f"${m.get('pricing_input', 0):.2f}",
          f"${m.get('pricing_output', 0):.2f}"
        )
      console.print(table)
      if len(results) > 25:
        console.print(f"[dim]... and {len(results) - 25} more results. Refine your query to narrow down.[/dim]")
    elif session.provider == "ollama":
      table.add_column("Model Tag", style="cyan")
      table.add_column("Name", style="white")
      table.add_column("Size", style="green", justify="right")
      table.add_column("Quantization", style="yellow")
      
      for m in results:
        size_gb = m.get("size", 0) / (1024**3)
        quant = m.get("details", {}).get("quantization_level", "Unknown")
        table.add_row(m.get("id", ""), m.get("name", ""), f"{size_gb:.2f} GB", quant)
      console.print(table)
  else:
    console.print(f"[bold red]Unknown models command '{subcmd}'. Use '/models' to list, '/models add <name>', '/models remove <id/name>', '/models available', or '/models search <query>'.[/bold red]")
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
    session.update_available_models_async(force_refresh=True)
    console.print("[bold green]API key updated successfully.[/bold green]")
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
    try:
      session.save_session(arg.strip())
      console.print(f"[bold green]Session saved successfully to {arg.strip()}[/bold green]")
    except Exception as e:
      console.print(f"[bold red]Error saving session: {str(e)}[/bold red]")
  return True


def cmd_load_session(session: Any, arg: str) -> bool:
  if not arg:
    console.print("[bold red]Error: Usage: /load_session <file_path>[/bold red]")
  else:
    try:
      session.load_session(arg.strip())
      console.print(f"[bold green]Session loaded successfully from {arg.strip()}[/bold green]")
    except Exception as e:
      console.print(f"[bold red]Error loading session: {str(e)}[/bold red]")
  return True


def cmd_tools(session: Any, arg: str) -> bool:
  session.show_tools()
  return True


def cmd_skill(session: Any, arg: str) -> bool:
  arg = arg.strip()
  if not arg:
    available = sorted(session.ondemand_skills.keys())
    active = sorted(session.explicit_skills)
    console.print("[bold cyan]On-demand Skills Status:[/bold cyan]")
    console.print(f"  Available on-demand skills: {', '.join(available) if available else 'None'}")
    console.print(f"  Explicitly loaded: {', '.join(active) if active else 'None'}")
    return True

  names = arg.split()
  if len(names) == 1 and names[0].lower() == "clear":
    session.explicit_skills.clear()
    console.print("[bold green]All explicitly loaded on-demand skills cleared.[/bold green]")
    return True

  for name in names:
    if name in session.ondemand_skills:
      if name not in session.explicit_skills:
        session.explicit_skills.append(name)
      console.print(f"Skill [bold cyan]{name}[/bold cyan] loaded on-demand.")
    else:
      console.print(f"[bold red]Error: On-demand skill '{name}' not found.[/bold red]")
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
    tok = session.count_tokens_estimate(content)
    if reasoning:
      tok += session.count_tokens_estimate(reasoning)
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


def cmd_whitelist(session: Any, arg: str) -> bool:
  arg = arg.strip()
  if not arg:
    # No argument: display the whitelist table
    session.show_whitelist()
    return True
    
  parts = arg.split(maxsplit=2)
  subcmd = parts[0].lower()
  
  if subcmd == "clear":
    session.clear_whitelist_paths()
    console.print("[bold green]Whitelisted paths cleared successfully.[/bold green]")
    
  elif subcmd == "add":
    if len(parts) < 2:
      console.print("[bold red]Usage: /whitelist add <path> [ro|rw][/bold red]")
      return True
    path = parts[1]
    mode = parts[2].lower() if len(parts) >= 3 else "rw"
    
    if mode not in ("ro", "rw"):
      console.print("[bold red]Invalid permission mode. Choose 'ro' (Read-Only) or 'rw' (Read-Write).[/bold red]")
      return True
      
    abs_path = session.add_whitelist_path(path, mode)
    if mode == "ro":
      console.print(f"[bold green]Added Read-Only path:[/bold green] {abs_path}")
    else:
      console.print(f"[bold green]Added Read-Write path:[/bold green] {abs_path}")
      
  elif subcmd == "remove":
    if len(parts) < 2:
      console.print("[bold red]Usage: /whitelist remove <path>[/bold red]")
      return True
    path = parts[1]
    abs_path, removed = session.remove_whitelist_path(path)
      
    if removed:
      console.print(f"[bold green]Removed path from whitelist:[/bold green] {abs_path}")
    else:
      console.print(f"[bold red]Path not found in whitelist:[/bold red] {abs_path}")
      
  else:
    console.print(f"[bold red]Unknown whitelist command '{subcmd}'. Use 'clear', 'add <path> [ro|rw]', or 'remove <path>'.[/bold red]")
    
  return True


def cmd_config(session: Any, arg: str) -> bool:
  from rich.table import Table
  allowed_keys = {
    "api_delay": float,
    "api_timeout": float,
    "max_loops": int,
    "context_size": int,
    "max_thinking_chars": int,
    "max_thinking_leeway_chars": int,
    "history_keep_messages": int,
    "prompt_caching": bool,
    "max_read_chars": int,
    "max_grep_results": int,
    "max_command_chars": int,
    "max_url_chars": int,
    "max_dir_items": int,
  }
  
  arg = arg.strip()
  if not arg:
    table = Table(title="Configuration Settings", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Type", style="dim white")
    for key, val_type in sorted(allowed_keys.items()):
      val = getattr(session, key, None)
      table.add_row(key, str(val), val_type.__name__)
    console.print(table)
    return True
    
  if "=" in arg:
    parts = arg.split("=", 1)
    key = parts[0].strip().lower()
    val_str = parts[1].strip()
    
    if key not in allowed_keys:
      console.print(f"[bold red]Error: '{key}' is not a configurable parameter.[/bold red]")
      console.print(f"Allowed parameters: {', '.join(sorted(allowed_keys.keys()))}")
      return True
      
    val_type = allowed_keys[key]
    try:
      if val_type is bool:
        if val_str.lower() in ("true", "yes", "1", "on"):
          parsed_val = True
        elif val_str.lower() in ("false", "no", "0", "off"):
          parsed_val = False
        else:
          raise ValueError("Must be a boolean value (true/false, yes/no, 1/0, on/off)")
      else:
        parsed_val = val_type(val_str)
        
      setattr(session, key, parsed_val)
      console.print(f"[bold green]Configuration updated:[/bold green] {key} = [cyan]{parsed_val}[/cyan]")
    except ValueError as e:
      console.print(f"[bold red]Error parsing value for '{key}': {str(e)}[/bold red]")
  else:
    key = arg.strip().lower()
    if key not in allowed_keys:
      console.print(f"[bold red]Error: '{key}' is not a configurable parameter.[/bold red]")
      console.print(f"Allowed parameters: {', '.join(sorted(allowed_keys.keys()))}")
      return True
    val = getattr(session, key, None)
    console.print(f"{key} = [bold green]{val}[/bold green] (type: {allowed_keys[key].__name__})")
    
  return True


def cmd_backups(session: Any, arg: str) -> bool:
  arg = arg.strip()
  from chatty.backup import get_backups_dir, list_backups
  
  backups_dir = get_backups_dir(session.sandbox)
  if not os.path.exists(backups_dir):
    console.print("[bold yellow]No backups directory found. No backups have been created yet.[/bold yellow]")
    return True
    
  if not arg:
    console.print("[bold cyan]Files with available backups:[/bold cyan]")
    found_any = False
    for root, dirs, files in os.walk(backups_dir):
      bak_files = [f for f in files if f.endswith(".bak")]
      if bak_files:
        rel_file_path = os.path.relpath(root, backups_dir)
        console.print(f"  - [green]{rel_file_path}[/green] ({len(bak_files)} backup(s) available)")
        found_any = True
    if not found_any:
      console.print("  No backups found.")
    console.print("\nUse [cyan]/backups <file_path>[/cyan] to view details of a specific file's backups.")
    return True
    
  backups = list_backups(session.sandbox, arg)
  if not backups:
    console.print(f"[bold yellow]No backups found for file '{arg}'.[/bold yellow]")
    return True
    
  console.print(f"[bold cyan]Backups for file '{arg}' (newest first):[/bold cyan]")
  for idx, (ts, time_str) in enumerate(backups, 1):
    console.print(f"  {idx}. [yellow]{time_str}[/yellow] (timestamp: {ts})")
  console.print(f"\nUse [cyan]/restore {arg} [index_or_timestamp][/cyan] to restore a specific backup.")
  return True


def cmd_restore(session: Any, arg: str) -> bool:
  arg = arg.strip()
  if not arg:
    console.print("[bold red]Error: Usage: /restore <file_path> [index_or_timestamp][/bold red]")
    return True
    
  parts = arg.split(maxsplit=1)
  file_path = parts[0]
  target = parts[1].strip() if len(parts) > 1 else None
  
  from chatty.backup import list_backups, restore_backup
  
  backups = list_backups(session.sandbox, file_path)
  if not backups:
    console.print(f"[bold red]Error: No backups found for '{file_path}'.[/bold red]")
    return True
    
  timestamp = None
  if target:
    try:
      idx = int(target)
      if 1 <= idx <= len(backups):
        timestamp = backups[idx - 1][0]
      else:
        timestamp = idx
    except ValueError:
      console.print(f"[bold red]Error: Invalid index or timestamp '{target}'.[/bold red]")
      return True
      
  res = restore_backup(session.sandbox, file_path, timestamp)
  if res.startswith("Error"):
    console.print(f"[bold red]{res}[/bold red]")
  else:
    console.print(f"[bold green]{res}[/bold green]")
  return True


def cmd_copy(session: Any, arg: str) -> bool:
  """Extracts code blocks from the last assistant message and copies to clipboard."""
  assistant_msgs = [m for m in session.messages if m.get("role") == "assistant" and m.get("content")]
  if not assistant_msgs:
    console.print("[bold red]No assistant response found in this session yet.[/bold red]")
    return True

  last_content = assistant_msgs[-1]["content"]

  import re
  blocks = re.findall(r"```(\w*)\r?\n(.*?)\r?\n```", last_content, re.DOTALL)
  if not blocks:
    console.print("[bold red]No code blocks found in the last assistant response.[/bold red]")
    return True

  arg = arg.strip()
  if arg:
    try:
      idx = int(arg)
      if not (1 <= idx <= len(blocks)):
        console.print(f"[bold red]Error: Invalid code block index {idx}. Choose between 1 and {len(blocks)}.[/bold red]")
        return True
      selected_block = blocks[idx - 1]
      selected_index = idx
    except ValueError:
      console.print("[bold red]Error: Block index must be a number.[/bold red]")
      return True
  else:
    if len(blocks) == 1:
      selected_block = blocks[0]
      selected_index = 1
    else:
      # Multiple blocks, show list
      console.print("[bold yellow]Multiple code blocks found in the last response:[/bold yellow]")
      for i, (lang, content) in enumerate(blocks, 1):
        line_count = len(content.splitlines())
        lang_str = lang if lang else "text"
        preview = content.splitlines()[0] if content else ""
        if len(preview) > 50:
          preview = preview[:47] + "..."
        console.print(f"  [cyan]{i}.[/cyan] {lang_str} ({line_count} lines) -> [dim]{preview}[/dim]")
      console.print("Usage: /copy <block_index>")
      return True

  code_content = selected_block[1]
  from chatty.utils import copy_to_clipboard
  if copy_to_clipboard(code_content):
    console.print(f"[bold green]Copied code block {selected_index} to clipboard.[/bold green]")
  else:
    console.print("[bold red]Failed to copy to clipboard. Please install a clipboard utility (e.g. xclip, xsel, or wl-copy).[/bold red]")
  return True


def cmd_write(session: Any, arg: str) -> bool:
  """Writes a code block from the last assistant message to a file."""
  arg = arg.strip()
  if not arg:
    console.print("[bold red]Error: Usage: /write <file_path> [block_index][/bold red]")
    return True

  parts = arg.split()
  block_index = None

  if len(parts) > 1 and parts[-1].isdigit():
    block_index = int(parts[-1])
    file_path = " ".join(parts[:-1])
  else:
    file_path = " ".join(parts)

  assistant_msgs = [m for m in session.messages if m.get("role") == "assistant" and m.get("content")]
  if not assistant_msgs:
    console.print("[bold red]No assistant response found in this session yet.[/bold red]")
    return True

  last_content = assistant_msgs[-1]["content"]

  import re
  blocks = re.findall(r"```(\w*)\r?\n(.*?)\r?\n```", last_content, re.DOTALL)
  if not blocks:
    console.print("[bold red]No code blocks found in the last assistant response.[/bold red]")
    return True

  if block_index is not None:
    if not (1 <= block_index <= len(blocks)):
      console.print(f"[bold red]Error: Invalid code block index {block_index}. Choose between 1 and {len(blocks)}.[/bold red]")
      return True
    selected_block = blocks[block_index - 1]
    selected_index = block_index
  else:
    if len(blocks) == 1:
      selected_block = blocks[0]
      selected_index = 1
    else:
      # Multiple blocks, show list
      console.print("[bold yellow]Multiple code blocks found. Please specify which block to write:[/bold yellow]")
      for i, (lang, content) in enumerate(blocks, 1):
        line_count = len(content.splitlines())
        lang_str = lang if lang else "text"
        preview = content.splitlines()[0] if content else ""
        if len(preview) > 50:
          preview = preview[:47] + "..."
        console.print(f"  [cyan]{i}.[/cyan] {lang_str} ({line_count} lines) -> [dim]{preview}[/dim]")
      console.print(f"Usage: /write {file_path} <block_index>")
      return True

  code_content = selected_block[1]
  import os

  file_path = os.path.expanduser(file_path)
  if not os.path.isabs(file_path):
    file_path = os.path.join(session.sandbox, file_path)

  try:
    dir_name = os.path.dirname(file_path)
    if dir_name:
      os.makedirs(dir_name, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(code_content)
    console.print(f"[bold green]Saved code block {selected_index} to '{file_path}'.[/bold green]")
  except Exception as e:
    console.print(f"[bold red]Error saving code block: {str(e)}[/bold red]")
  return True


def cmd_show(session: Any, arg: str) -> bool:
  """Reads a Markdown file and renders it beautifully inside Chatty using rich."""
  file_path = arg.strip()
  if not file_path:
    console.print("[bold red]Error: Usage: /show <file_path>[/bold red]")
    return True

  import os

  file_path = os.path.expanduser(file_path)
  if not os.path.isabs(file_path):
    file_path = os.path.join(session.sandbox, file_path)
  file_path = os.path.realpath(file_path)

  if not os.path.exists(file_path):
    console.print(f"[bold red]Error: File '{file_path}' does not exist.[/bold red]")
    return True

  if os.path.isdir(file_path):
    console.print(f"[bold red]Error: '{file_path}' is a directory.[/bold red]")
    return True

  try:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
      content = f.read()

    _, ext = os.path.splitext(file_path.lower())
    if ext in [".md", ".markdown"]:
      from rich.markdown import Markdown

      console.print(Markdown(content))
    else:
      from rich.syntax import Syntax

      lexer = Syntax.guess_lexer(file_path)
      syntax = Syntax(content, lexer=lexer, line_numbers=True, theme="monokai")
      console.print(syntax)
  except Exception as e:
    console.print(f"[bold red]Error reading file: {str(e)}[/bold red]")
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
  "/models": cmd_models,
  "/oracle": cmd_oracle,
  "/sandbox": cmd_sandbox,
  "/context": cmd_context,
  "/loops": cmd_loops,
  "/api_key": cmd_api_key,
  "/system": cmd_system,
  "/load": cmd_load,
  "/save": cmd_save,
  "/save_session": cmd_save,
  "/load_session": cmd_load_session,
  "/tools": cmd_tools,
  "/skill": cmd_skill,
  "/history": cmd_history,
  "/undo": cmd_undo,
  "/pop": cmd_pop,
  "/whitelist": cmd_whitelist,
  "/permissions": cmd_whitelist,
  "/config": cmd_config,
  "/backups": cmd_backups,
  "/restore": cmd_restore,
  "/copy": cmd_copy,
  "/clip": cmd_copy,
  "/write": cmd_write,
  "/save_code": cmd_write,
  "/show": cmd_show,
}


