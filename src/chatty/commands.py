import json
import os
from typing import Dict, Callable, Any, List, Optional

from rich.console import Console
from rich.panel import Panel
from chatty.utils import load_system_prompt_from_file

console = Console()


def cmd_exit(session: Any, arg: str) -> bool:
  session.cleanup_scratch(prompt=True)
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
    from chatty.providers import PROVIDER_KEYS
    if arg not in PROVIDER_KEYS:
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


def cmd_discovery_model(session: Any, arg: str) -> bool:
  arg = arg.strip()
  if not arg:
    disc_model = session.get_discovery_model()
    if getattr(session.config, "discovery_model", None):
      console.print(f"Current discovery model: [bold cyan]{disc_model}[/bold cyan]")
    else:
      console.print(f"Discovery model is not explicitly configured (defaulting to active model: [bold cyan]{disc_model}[/bold cyan]).")
    return True

  session.discovery_model = arg
  console.print(f"Switched discovery model to: [bold green]{arg}[/bold green]")
  return True


def filter_models(available_models: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
  """Filters and sorts available models based on user query conditions.

  Supports combining multiple conditions with AND logic:
  - Cost/Price: cost<0.1, price<=0.5, cost>0.01, cost=0, out_cost<0.5
  - Context: context>=1M, context>=128k, ctx>32k, context<=2M (k and m multipliers)
  - Size: size<5g, size<=10gb, size>1g (supports g, m, k units)
  - Category / Flags: cat:vision, is:vision, cat:free, is:free
  - Sorting: sort:cost, sort:context, sort:newest, sort:size
  - Free text: keywords matching model id or name (e.g. qwen, coder)
  """
  if not available_models:
    return []

  import re
  import shlex

  # Normalize spaces around comparison operators: e.g. "cost < 0.1" -> "cost<0.1"
  normalized_query = re.sub(
    r'(?i)\b(cost|price|in_cost|out_cost|output_cost|pricing_output|context|ctx|size)\s*([<>=!]+)\s*',
    r'\1\2',
    query
  )
  # Replace commas with spaces
  normalized_query = normalized_query.replace(",", " ")

  try:
    tokens = shlex.split(normalized_query)
  except ValueError:
    tokens = normalized_query.split()

  predicates = []
  text_filters = []
  sort_by = None

  for token in tokens:
    token_lower = token.lower()

    # Skip ellipsis, punctuation, and conjunctions
    if token in ("...", "..", ".") or token_lower in ("and", "&&"):
      continue

    # 1. Input Cost/Price Filter (e.g. cost<0.1, price<=0.5, cost>0.01, cost=0)
    cost_match = re.match(r'^(?:cost|price|in_cost)(<=|>=|<|>|==|=|!=)\$?(\d+(?:\.\d+)?)$', token_lower)
    if cost_match:
      op = cost_match.group(1)
      val = float(cost_match.group(2))
      if op == "<":
        predicates.append(lambda m, v=val: float(m.get("pricing_input") or 0) < v)
      elif op == "<=":
        predicates.append(lambda m, v=val: float(m.get("pricing_input") or 0) <= v)
      elif op == ">":
        predicates.append(lambda m, v=val: float(m.get("pricing_input") or 0) > v)
      elif op == ">=":
        predicates.append(lambda m, v=val: float(m.get("pricing_input") or 0) >= v)
      elif op in ("=", "=="):
        predicates.append(lambda m, v=val: abs(float(m.get("pricing_input") or 0) - v) < 1e-9)
      elif op == "!=":
        predicates.append(lambda m, v=val: abs(float(m.get("pricing_input") or 0) - v) >= 1e-9)
      continue

    # Output cost filter (e.g. out_cost<0.5)
    out_cost_match = re.match(r'^(?:out_cost|output_cost|pricing_output)(<=|>=|<|>|==|=|!=)\$?(\d+(?:\.\d+)?)$', token_lower)
    if out_cost_match:
      op = out_cost_match.group(1)
      val = float(out_cost_match.group(2))
      if op == "<":
        predicates.append(lambda m, v=val: float(m.get("pricing_output") or 0) < v)
      elif op == "<=":
        predicates.append(lambda m, v=val: float(m.get("pricing_output") or 0) <= v)
      elif op == ">":
        predicates.append(lambda m, v=val: float(m.get("pricing_output") or 0) > v)
      elif op == ">=":
        predicates.append(lambda m, v=val: float(m.get("pricing_output") or 0) >= v)
      elif op in ("=", "=="):
        predicates.append(lambda m, v=val: abs(float(m.get("pricing_output") or 0) - v) < 1e-9)
      elif op == "!=":
        predicates.append(lambda m, v=val: abs(float(m.get("pricing_output") or 0) - v) >= 1e-9)
      continue

    # 2. Context Length Filter (e.g. context>=1M, ctx>=32k, context<128k)
    ctx_match = re.match(r'^(?:context|ctx)(<=|>=|<|>|==|=|!=)(\d+(?:\.\d+)?)([kmb]?)$', token_lower)
    if ctx_match:
      op = ctx_match.group(1)
      num = float(ctx_match.group(2))
      unit = ctx_match.group(3)
      if unit == "m":
        target_ctx = int(num * 1_000_000)
      elif unit == "k":
        target_ctx = int(num * 1_000)
      else:
        target_ctx = int(num)

      def check_ctx(m, o=op, target=target_ctx):
        c = m.get("context")
        if c is None:
          return False
        c = int(c)
        if o == "<":
          return c < target
        elif o == "<=":
          return c <= target
        elif o == ">":
          return c > target
        elif o == ">=":
          return c >= target
        elif o in ("=", "=="):
          return c == target
        elif o == "!=":
          return c != target
        return True

      predicates.append(check_ctx)
      continue

    # 3. Size Filter (e.g. size<5g, size<=10gb, size>1g)
    size_match = re.match(r'^size(<=|>=|<|>|==|=|!=)(\d+(?:\.\d+)?)([gmk]?b?)$', token_lower)
    if size_match:
      op = size_match.group(1)
      num = float(size_match.group(2))
      unit = size_match.group(3)
      if "g" in unit:
        target_bytes = int(num * (1024**3))
      elif "m" in unit:
        target_bytes = int(num * (1024**2))
      elif "k" in unit:
        target_bytes = int(num * 1024)
      else:
        target_bytes = int(num * (1024**3)) if num < 1024 else int(num)

      def check_size(m, o=op, target=target_bytes):
        s = m.get("size")
        if s is None:
          return False
        s = int(s)
        if o == "<":
          return s < target
        elif o == "<=":
          return s <= target
        elif o == ">":
          return s > target
        elif o == ">=":
          return s >= target
        elif o in ("=", "=="):
          return s == target
        elif o == "!=":
          return s != target
        return True

      predicates.append(check_size)
      continue

    # 4. Category / Flag Filter (e.g. cat:vision, is:vision, cat:free, is:free)
    if token_lower.startswith(("cat:", "is:")):
      cat_val = token_lower.split(":", 1)[1]
      if cat_val in ("vision", "image", "multimodal"):
        def check_vision(m):
          arch = m.get("architecture")
          if isinstance(arch, dict):
            mods = arch.get("input_modalities") or []
            if "image" in mods or "video" in mods:
              return True
          m_id = str(m.get("id") or "").lower()
          m_name = str(m.get("name") or "").lower()
          return "vision" in m_id or "vision" in m_name
        predicates.append(check_vision)
      elif cat_val == "free":
        predicates.append(lambda m: float(m.get("pricing_input") or 0) == 0 and float(m.get("pricing_output") or 0) == 0)
      continue

    # 5. Sorting Option (e.g. sort:cost, sort:context, sort:newest, sort:size)
    if token_lower.startswith("sort:"):
      sort_val = token_lower[5:]
      if sort_val in ("cost", "price"):
        sort_by = "cost"
      elif sort_val in ("context", "ctx"):
        sort_by = "context"
      elif sort_val in ("newest", "date"):
        sort_by = "newest"
      elif sort_val == "size":
        sort_by = "size"
      continue

    # 6. Freestanding text filter
    text_filters.append(token_lower)

  results = []
  for m in available_models:
    m_id = str(m.get("id") or "").lower()
    m_name = str(m.get("name") or "").lower()

    if text_filters and not all(f in m_id or f in m_name for f in text_filters):
      continue

    if not all(pred(m) for pred in predicates):
      continue

    results.append(m)

  # Perform sorting
  if sort_by == "cost":
    results.sort(key=lambda x: float(x.get("pricing_input") or 0))
  elif sort_by == "context":
    results.sort(key=lambda x: int(x.get("context") or 0), reverse=True)
  elif sort_by == "newest":
    results.sort(key=lambda x: float(x.get("created") or 0), reverse=True)
  elif sort_by == "size":
    results.sort(key=lambda x: int(x.get("size") or 0), reverse=True)

  return results


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
    console.print("  [cyan]/models info <ID or model_name>[/cyan] - Show detailed information about a model")
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
    
    if session.provider == "openrouter" or any("pricing_input" in m for m in session.available_models):
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
      
    elif session.provider == "ollama" or any("size" in m for m in session.available_models):
      table.add_column("Model Tag", style="cyan")
      table.add_column("Name", style="white")
      table.add_column("Size", style="green", justify="right")
      table.add_column("Quantization", style="yellow")
      
      for m in session.available_models:
        size_gb = m.get("size", 0) / (1024**3)
        quant = m.get("details", {}).get("quantization_level", "Unknown")
        table.add_row(m.get("id", ""), m.get("name", ""), f"{size_gb:.2f} GB", quant)
      console.print(table)
    else:
      table.add_column("Model ID", style="cyan")
      table.add_column("Name", style="white")
      for m in session.available_models:
        table.add_row(m.get("id", ""), m.get("name", ""))
      console.print(table)
      
  elif subcmd == "search":
    if len(parts) < 2:
      console.print("[bold red]Error: Usage: /models search <query>[/bold red]")
      return True

    query = parts[1].strip()
    results = filter_models(getattr(session, "available_models", []), query)

    if not results:
      console.print(f"[yellow]No models matching '{query}' found.[/yellow]")
      return True

    from rich.table import Table
    table = Table(title=f"Search Results for '{query}'", show_header=True, header_style="bold magenta")

    if session.provider == "openrouter" or any("pricing_input" in m for m in results):
      table.add_column("Model ID", style="cyan")
      table.add_column("Name", style="white")
      table.add_column("Context Length", style="green", justify="right")
      table.add_column("Input (per 1M)", style="yellow", justify="right")
      table.add_column("Output (per 1M)", style="yellow", justify="right")

      for m in results[:25]:
        p_in = m.get("pricing_input")
        p_out = m.get("pricing_output")
        table.add_row(
          m.get("id", ""),
          m.get("name", ""),
          f"{m['context']:,}" if m.get("context") else "Unknown",
          f"${p_in:.2f}" if p_in is not None else "$0.00",
          f"${p_out:.2f}" if p_out is not None else "$0.00"
        )
      console.print(table)
      if len(results) > 25:
        console.print(f"[dim]... and {len(results) - 25} more results. Refine your query to narrow down.[/dim]")
    elif session.provider == "ollama" or any("size" in m for m in results):
      table.add_column("Model Tag", style="cyan")
      table.add_column("Name", style="white")
      table.add_column("Size", style="green", justify="right")
      table.add_column("Quantization", style="yellow")

      for m in results:
        size_bytes = m.get("size") or 0
        size_gb = size_bytes / (1024**3)
        details = m.get("details") or {}
        quant = details.get("quantization_level", "Unknown") if isinstance(details, dict) else "Unknown"
        table.add_row(m.get("id", ""), m.get("name", ""), f"{size_gb:.2f} GB", quant)
      console.print(table)
    else:
      table.add_column("Model ID", style="cyan")
      table.add_column("Name", style="white")
      for m in results:
        table.add_row(m.get("id", ""), m.get("name", ""))
      console.print(table)
  elif subcmd == "info":
    if len(parts) < 2:
      console.print("[bold red]Error: Usage: /models info <ID or model_name>[/bold red]")
      return True
    target = parts[1].strip()
    resolved_name = target
    try:
      idx = int(target)
      if 1 <= idx <= len(session.models):
        resolved_name = session.models[idx - 1]
    except ValueError:
      pass
    if not getattr(session, "available_models", None):
      console.print("[yellow]Loading available models list...[/yellow]")
      from chatty.llm import fetch_available_models
      session.available_models = fetch_available_models(session)
    if not getattr(session, "available_models", None):
      console.print("[bold red]Error: No available models list found.[/bold red]")
      return True
    match = None
    for m in session.available_models:
      if m.get("id") == resolved_name:
        match = m
        break
    if not match:
      for m in session.available_models:
        if resolved_name.lower() in m.get("id", "").lower() or resolved_name.lower() in m.get("name", "").lower():
          match = m
          break
    if not match:
      console.print(f"[bold red]Error: Model '{target}' not found in available models list.[/bold red]")
      return True
    from rich.table import Table
    from rich.panel import Panel
    from datetime import datetime
    import re
    def extract_params(m):
      details = m.get("details")
      if details and isinstance(details, dict) and details.get("parameter_size"):
        return details["parameter_size"]
      text = (m.get('name', '') + ' ' + m.get('id', '') + ' ' + m.get('description', '')).lower()
      m1 = re.search(r'\b(\d+(?:\.\d+)?)\s*b\s*-\s*parameter', text)
      if m1: return m1.group(1) + 'B'
      m2 = re.search(r'\b(\d+(?:\.\d+)?)\s*b\s*active', text)
      if m2: return m2.group(1) + 'B'
      m3 = re.search(r'\b(\d+(?:\.\d+)?)\s*billion\b', text)
      if m3: return m3.group(1) + 'B'
      m4 = re.findall(r'\b(\d+(?:\.\d+)?)\s*b\b', text)
      for val in m4:
        if val not in ('4', '3', '2', '1'):
          return val + 'B'
      return "Unknown"
    table = Table(show_header=False, box=None, expand=True)
    table.add_column("Key", style="cyan", width=22)
    table.add_column("Value", style="white")
    table.add_row("Model ID", match.get("id", "Unknown"))
    table.add_row("Model Name", match.get("name", "Unknown"))
    if "pricing_input" in match or "context" in match or session.provider == "openrouter":
      created_ts = match.get("created")
      created_str = "Unknown"
      if created_ts:
        try:
          created_str = datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d")
        except Exception:
          pass
      table.add_row("Created Date", created_str)
      table.add_row("Knowledge Cutoff", match.get("knowledge_cutoff") or "Unknown")
      ctx = match.get("context")
      table.add_row("Context Length", f"{ctx:,} tokens" if ctx else "Unknown")
      in_cost = match.get("pricing_input", 0)
      out_cost = match.get("pricing_output", 0)
      table.add_row("Pricing (Input/1M)", f"${in_cost:.2f}")
      table.add_row("Pricing (Output/1M)", f"${out_cost:.2f}")
      if match.get("hugging_face_id"):
        table.add_row("Hugging Face ID", match.get("hugging_face_id"))
      arch = match.get("architecture")
      if arch and isinstance(arch, dict):
        table.add_row("Architecture Modality", arch.get("modality") or "Unknown")
        table.add_row("Tokenizer", arch.get("tokenizer") or "Unknown")
        if arch.get("instruct_type"):
          table.add_row("Instruct Type", arch.get("instruct_type"))
      table.add_row("Estimated Parameters", extract_params(match))
      desc = match.get("description")
    elif "size" in match or "details" in match or session.provider == "ollama":
      size_bytes = match.get("size", 0)
      size_gb = size_bytes / (1024**3)
      table.add_row("Size", f"{size_gb:.2f} GB ({size_bytes:,} bytes)")
      details = match.get("details", {})
      if details and isinstance(details, dict):
        table.add_row("Format", details.get("format") or "Unknown")
        table.add_row("Family", details.get("family") or "Unknown")
        if details.get("families"):
          table.add_row("Families", ", ".join(details["families"]))
        table.add_row("Parameter Size", details.get("parameter_size") or "Unknown")
        table.add_row("Quantization Level", details.get("quantization_level") or "Unknown")
      desc = None
    else:
      desc = None
    from rich.console import Group
    elements = [table]
    if desc:
      elements.append(Panel(desc, title="Description", border_style="dim white", expand=True))
    console.print(Panel(Group(*elements), title=f"Model Information: {match.get('name', 'Unknown')}", border_style="magenta", expand=True))
  else:
    console.print(f"[bold red]Unknown models command '{subcmd}'. Use '/models' to list, '/models add <name>', '/models remove <id/name>', '/models available', '/models search <query>', or '/models info <id/name>'.[/bold red]")
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
  if not session.messages:
    console.print("[bold yellow]Conversation history is empty.[/bold yellow]")
    return True

  arg = arg.strip()
  if arg:
    try:
      idx = int(arg)
    except ValueError:
      console.print("[bold red]Error: History index must be an integer.[/bold red]")
      return True

    total = len(session.messages)
    if idx < 0:
      msg_idx = total + idx + 1
    else:
      msg_idx = idx

    if not (1 <= msg_idx <= total):
      console.print(f"[bold red]Error: Invalid history index '{arg}'. Available indices: 1 to {total} (or negative indexes -1 to -{total}).[/bold red]")
      return True

    msg = session.messages[msg_idx - 1]
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")

    console.print(f"[bold cyan]History Entry {msg_idx} of {total}[/bold cyan]")
    
    metadata_lines = []
    metadata_lines.append(f"  [bold]Role:[/bold] {role}")
    if "name" in msg:
      metadata_lines.append(f"  [bold]Name:[/bold] {msg['name']}")
    if "tool_call_id" in msg:
      metadata_lines.append(f"  [bold]Tool Call ID:[/bold] {msg['tool_call_id']}")

    tok = session.count_tokens_estimate(content)
    if reasoning:
      tok += session.count_tokens_estimate(reasoning)
    metadata_lines.append(f"  [bold]Estimated Tokens:[/bold] {tok}")
    
    console.print("\n".join(metadata_lines))
    console.print()

    if reasoning:
      console.print(Panel(reasoning.strip(), title="🧠 Thinking Process", border_style="yellow"))
      console.print()

    if content.strip():
      from rich.markdown import Markdown
      if role == "assistant":
        border_style = "green"
        title = "🤖 Assistant Response"
      elif role == "user":
        border_style = "blue"
        title = "👤 User Message"
      elif role == "system":
        border_style = "magenta"
        title = "⚙️ System Instructions"
      elif role == "tool":
        border_style = "cyan"
        title = f"🛠️ Tool Output ({msg.get('name', 'unknown')})"
      else:
        border_style = "white"
        title = f"Message ({role})"
        
      console.print(Panel(Markdown(content.strip()), title=title, border_style=border_style))
      console.print()

    if "tool_calls" in msg and msg["tool_calls"]:
      from rich.table import Table
      
      table = Table(show_header=True, header_style="bold magenta", box=None, expand=True)
      table.add_column("Tool Name", style="green", width=25)
      table.add_column("Arguments", style="white")
      
      for tc in msg["tool_calls"]:
        func = tc.get("function", {})
        table.add_row(func.get("name", ""), func.get("arguments", ""))
      
      console.print(Panel(table, title="🛠️ Tool Calls", border_style="yellow"))
      console.print()

    return True

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
    "repo_map": bool,
    "repo_map_tokens": int,
    "discovery_model": str,
    "discovery_loops": int,
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
  """Extracts code blocks or the full markdown response from the last assistant message and copies to clipboard."""
  assistant_msgs = [m for m in session.messages if m.get("role") == "assistant" and m.get("content")]
  if not assistant_msgs:
    console.print("[bold red]No assistant response found in this session yet.[/bold red]")
    return True

  last_content = assistant_msgs[-1]["content"]
  arg = arg.strip().lower()

  import re
  blocks = re.findall(r"```(\w*)\r?\n(.*?)\r?\n```", last_content, re.DOTALL)

  if arg in ("all", "full", "raw", "response", "0"):
    from chatty.utils import copy_to_clipboard
    if copy_to_clipboard(last_content):
      console.print("[bold green]Copied full assistant response to clipboard.[/bold green]")
    else:
      console.print("[bold red]Failed to copy to clipboard. Please install a clipboard utility (e.g. xclip, xsel, or wl-copy).[/bold red]")
    return True

  if not blocks:
    console.print("[bold red]No code blocks found in the last assistant response.[/bold red]")
    console.print("[dim]Tip: Use '/copy all' to copy the entire assistant response.[/dim]")
    return True

  if arg:
    try:
      idx = int(arg)
      if not (1 <= idx <= len(blocks)):
        console.print(f"[bold red]Error: Invalid code block index {idx}. Choose between 1 and {len(blocks)} (or 'all').[/bold red]")
        return True
      selected_block = blocks[idx - 1]
      selected_index = idx
    except ValueError:
      console.print("[bold red]Error: Block index must be a number or 'all'.[/bold red]")
      return True
  else:
    if len(blocks) == 1:
      selected_block = blocks[0]
      selected_index = 1
    else:
      # Multiple blocks, show list
      console.print("[bold yellow]Multiple code blocks found in the last response:[/bold yellow]")
      total_lines = len(last_content.splitlines())
      console.print(f"  [cyan]0.[/cyan] [bold]all[/bold] ({total_lines} lines) -> [dim]Full assistant response[/dim]")
      for i, (lang, content) in enumerate(blocks, 1):
        line_count = len(content.splitlines())
        lang_str = lang if lang else "text"
        preview = content.splitlines()[0] if content else ""
        if len(preview) > 50:
          preview = preview[:47] + "..."
        console.print(f"  [cyan]{i}.[/cyan] {lang_str} ({line_count} lines) -> [dim]{preview}[/dim]")
      console.print("Usage: /copy <block_index> (or /copy all)")
      return True

  code_content = selected_block[1]
  from chatty.utils import copy_to_clipboard
  if copy_to_clipboard(code_content):
    console.print(f"[bold green]Copied code block {selected_index} to clipboard.[/bold green]")
  else:
    console.print("[bold red]Failed to copy to clipboard. Please install a clipboard utility (e.g. xclip, xsel, or wl-copy).[/bold red]")
  return True


def cmd_write(session: Any, arg: str) -> bool:
  """Writes a code block or the full assistant response to a file."""
  arg = arg.strip()
  if not arg:
    console.print("[bold red]Error: Usage: /write <file_path> [block_index|all][/bold red]")
    return True

  parts = arg.split()
  target = None

  if len(parts) > 1:
    last_part = parts[-1].lower()
    if last_part in ("all", "full", "raw", "response", "0"):
      target = "all"
      file_path = " ".join(parts[:-1])
    elif last_part.isdigit():
      target = int(last_part)
      file_path = " ".join(parts[:-1])
    else:
      file_path = " ".join(parts)
  else:
    file_path = " ".join(parts)

  assistant_msgs = [m for m in session.messages if m.get("role") == "assistant" and m.get("content")]
  if not assistant_msgs:
    console.print("[bold red]No assistant response found in this session yet.[/bold red]")
    return True

  last_content = assistant_msgs[-1]["content"]

  import re
  blocks = re.findall(r"```(\w*)\r?\n(.*?)\r?\n```", last_content, re.DOTALL)

  if target == "all":
    save_content = last_content
    save_desc = "full assistant response"
  elif not blocks:
    save_content = last_content
    save_desc = "full assistant response (no code blocks found)"
  elif isinstance(target, int):
    if target == 0:
      save_content = last_content
      save_desc = "full assistant response"
    elif 1 <= target <= len(blocks):
      save_content = blocks[target - 1][1]
      save_desc = f"code block {target}"
    else:
      console.print(f"[bold red]Error: Invalid code block index {target}. Choose between 1 and {len(blocks)} (or 'all').[/bold red]")
      return True
  else:
    if len(blocks) == 1:
      save_content = blocks[0][1]
      save_desc = "code block 1"
    else:
      # Multiple blocks, show list
      console.print("[bold yellow]Multiple code blocks found. Please specify which block to write:[/bold yellow]")
      total_lines = len(last_content.splitlines())
      console.print(f"  [cyan]0.[/cyan] [bold]all[/bold] ({total_lines} lines) -> [dim]Full assistant response[/dim]")
      for i, (lang, content) in enumerate(blocks, 1):
        line_count = len(content.splitlines())
        lang_str = lang if lang else "text"
        preview = content.splitlines()[0] if content else ""
        if len(preview) > 50:
          preview = preview[:47] + "..."
        console.print(f"  [cyan]{i}.[/cyan] {lang_str} ({line_count} lines) -> [dim]{preview}[/dim]")
      console.print(f"Usage: /write {file_path} <block_index> (or /write {file_path} all)")
      return True

  import os

  file_path = os.path.expanduser(file_path)
  if not os.path.isabs(file_path):
    file_path = os.path.join(session.sandbox, file_path)

  try:
    dir_name = os.path.dirname(file_path)
    if dir_name:
      os.makedirs(dir_name, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(save_content)
    console.print(f"[bold green]Saved {save_desc} to '{file_path}'.[/bold green]")
  except Exception as e:
    console.print(f"[bold red]Error saving code block: {str(e)}[/bold red]")
  return True


def cmd_save_response(session: Any, arg: str) -> bool:
  """Saves the entire last assistant response (markdown) to a file."""
  arg = arg.strip()
  if not arg:
    console.print("[bold red]Error: Usage: /save_response <file_path>[/bold red]")
    return True
  return cmd_write(session, f"{arg} all")


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


def cmd_find_symbol(session: Any, arg: str) -> bool:
  arg = arg.strip()
  if not arg:
    console.print("[bold red]Error: Please specify a symbol name to search for.[/bold red]")
    return True
  from chatty.tools.code_intel import SymbolExtractor
  lsp_client = getattr(session, "lsp_client", None)
  extractor = SymbolExtractor(session.sandbox, lsp_client)
  matches = extractor.find_symbol(arg)
  if not matches:
    console.print(f"[yellow]No matches found for symbol '{arg}'.[/yellow]")
    return True
  console.print(f"[bold green]Found {len(matches)} match(es) for symbol '{arg}':[/bold green]")
  for match in matches:
    parent_part = f" (in Class {match['parent']})" if match.get("parent") else ""
    console.print(f"  - [cyan]{match['path']}:{match['line']}[/cyan] ({match['type']}{parent_part})")
  return True


def cmd_repo_map(session: Any, arg: str) -> bool:
  arg = arg.strip().lower()
  refresh = arg in ("refresh", "-r", "--refresh", "reload")
  repo_map = session.get_repo_map(refresh=refresh)
  if not repo_map:
    console.print("[yellow]Repository map is empty or no source files found.[/yellow]")
    return True
  from rich.panel import Panel
  console.print(Panel(repo_map, title="[bold cyan]Repository Map (Tree-Sitter / PageRank)[/bold cyan]", border_style="cyan"))
  return True


def cmd_discover(session: Any, arg: str) -> bool:
  arg = arg.strip()
  if not arg:
    console.print("[bold red]Error: Please specify a task or topic to discover context for.[/bold red]")
    console.print("Usage: /discover <task description>")
    return True
  from chatty.discovery import run_discovery
  dossier = run_discovery(session, arg)
  if not dossier:
    console.print("[yellow]Discovery agent finished without returning context.[/yellow]")
    return True
  from rich.panel import Panel
  from rich.markdown import Markdown
  console.print(Panel(Markdown(dossier), title=f"[bold cyan]🔍 Discovery Dossier: {arg}[/bold cyan]", border_style="cyan"))

  session.messages.append({
    "role": "user",
    "content": f"[Context gathered by Discovery Agent for task: '{arg}']:\n\n{dossier}"
  })
  session.messages.append({
    "role": "assistant",
    "content": f"Understood. I have reviewed the discovered context for '{arg}'. How would you like to proceed?"
  })
  console.print("[bold green]Dossier loaded into active conversation context.[/bold green]")
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
  "/discover": cmd_discover,
  "/discovery_model": cmd_discovery_model,
  "/discover_model": cmd_discovery_model,
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
  "/find_symbol": cmd_find_symbol,
  "/repo_map": cmd_repo_map,
  "/map": cmd_repo_map,
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
  "/save_response": cmd_save_response,
  "/save_reply": cmd_save_response,
  "/show": cmd_show,
}


