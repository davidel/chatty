import json
import logging
import os
import re
import select
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape

from chatty.tools import TOOLS_SCHEMA, execute_tool
from chatty.safety import active_session_var
from chatty.ui import optional_live, LiveScreenLayout
from chatty.llm import ThinkingBudgetExceeded, is_safe_to_interrupt, detect_repetition_loop

logger = logging.getLogger("chatty")
console = Console()

REPO_SCOUT_TOOL_NAMES: Set[str] = {
  "read_file",
  "search_grep",
  "locate_files",
  "get_outline",
  "find_symbol",
  "get_file_info",
  "fetch_url",
}
DISCOVERY_TOOL_NAMES: Set[str] = REPO_SCOUT_TOOL_NAMES

WEB_RESEARCH_TOOL_NAMES: Set[str] = {
  "search_web",
  "fetch_url",
}


def extract_session_context(session: Any) -> str:
  """Extracts compact context from the parent session messages."""
  if not getattr(session, "messages", None):
    return ""

  referenced_files: List[str] = []
  recent_user_requests: List[str] = []

  for msg in session.messages[-15:]:
    role = msg.get("role")
    if role == "user":
      content = msg.get("content", "")
      if isinstance(content, str) and content.strip():
        # Avoid huge prompt dumps
        summary = content.strip().split("\n")[0][:120]
        recent_user_requests.append(summary)

    # Check for tool call arguments
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
      func = tc.get("function", {})
      args_str = func.get("arguments", "")
      try:
        args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
        for k in ("path", "file_path", "target_file"):
          val = args.get(k)
          if val and isinstance(val, str) and val not in referenced_files:
            referenced_files.append(val)
      except Exception:
        pass

  parts = []
  if referenced_files:
    top_files = referenced_files[-8:]
    parts.append("Recently referenced files in active session:\n" + "\n".join(f"- `{f}`" for f in top_files))

  if recent_user_requests:
    last_req = recent_user_requests[-1]
    parts.append(f"Recent user prompt context: \"{last_req}\"")

  if not parts:
    return ""

  return "## Existing Session Context\n" + "\n\n".join(parts)


def build_discovery_system_prompt(session: Any, task: str) -> str:
  """Builds the specialized reconnaissance system prompt."""
  context_section = extract_session_context(session)

  repo_map_section = ""
  if hasattr(session, "get_repo_map") and getattr(session.config, "repo_map", True):
    try:
      rmap = session.get_repo_map()
      if rmap:
        repo_map_section = f"## Workspace Repository Map\n```\n{rmap}\n```\n"
    except Exception as e:
      logger.debug(f"Error fetching repo map for discovery: {e}")

  prompt = (
    "You are an expert Workspace Discovery & Reconnaissance Agent.\n"
    "Your SOLE MISSION is to investigate the codebase and compile a clear, precise, and actionable context dossier "
    "for the following task, without attempting to solve or implement the task yourself.\n\n"
    f"Task to investigate:\n{task}\n\n"
    "CRITICAL CONSTRAINTS & RULES:\n"
    "1. DO NOT write code, propose patches, or attempt to implement the solution.\n"
    "2. DO NOT write, edit, delete, or create any files. You only have read-only inspection tools.\n"
    "3. Use your tools proactively (locate_files, search_grep, find_symbol, get_outline, read_file, get_file_info, fetch_url) "
    "to explore the codebase.\n"
    "4. Thoroughly investigate:\n"
    "   - Exact target files and line ranges that are relevant to this task.\n"
    "   - Key definitions (classes, functions, types, constants).\n"
    "   - Call graphs, callers, callees, and dependencies.\n"
    "   - Existing tests and test fixtures covering this functionality.\n"
    "   - Potential constraints, edge cases, or architectural patterns.\n"
    "5. When you have gathered all necessary information, provide your FINAL RESPONSE as a clean, structured Markdown dossier "
    "in the following format (do NOT call further tools once you provide this):\n\n"
    "### Target Files & Line Ranges\n"
    "- `path/to/file`: lines X-Y (brief reason why this code is relevant)\n\n"
    "### Key Symbols & Definitions\n"
    "- `SymbolName` (in `path/to/file:line`): role and purpose\n\n"
    "### Dependencies & Call Sites\n"
    "- Inter-module interactions, callers, and callees\n\n"
    "### Relevant Tests\n"
    "- `tests/test_something.py`: existing test coverage\n\n"
    "### Architectural Notes & Constraints\n"
    "- Important conventions, pitfalls, or design decisions to keep in mind\n"
  )

  if context_section:
    prompt += f"\n{context_section}\n"
  if repo_map_section:
    prompt += f"\n{repo_map_section}\n"

  return prompt


def build_web_research_system_prompt(query: str) -> str:
  """Builds the specialized technical web research system prompt."""
  return (
    "You are an expert Technical Web Research & Documentation Scout.\n"
    "Your SOLE MISSION is to investigate online sources, official documentation, issues, and technical articles "
    "to provide a comprehensive, accurate, and actionable answer to the following technical query, "
    "without attempting to modify local files.\n\n"
    f"Query to research:\n{query}\n\n"
    "CRITICAL CONSTRAINTS & RULES:\n"
    "1. Use 'search_web' to locate high-quality, authoritative technical sources (official documentation, "
    "GitHub issues/releases, Stack Overflow, package repositories).\n"
    "2. Use 'fetch_url' to inspect detailed pages, API specifications, and code samples.\n"
    "3. PRESERVE TECHNICAL FIDELITY: Maintain exact method names, code snippets, type annotations, "
    "configuration options, and error messages verbatim. Do not paraphrase code or guess syntax.\n"
    "4. Note specific library versions, deprecations, breaking changes, and minimum requirements.\n"
    "5. When you have gathered all necessary information, provide your FINAL RESPONSE as a clean, "
    "structured Markdown briefing (do NOT call further tools once you provide this):\n\n"
    "### Summary & Direct Answer\n"
    "- Clear, direct answer to the query\n\n"
    "### Code Examples & API Signatures\n"
    "```language\n"
    "# Verbatim working code examples or syntax\n"
    "```\n\n"
    "### Important Nuances & Edge Cases\n"
    "- Version compatibility, pitfalls, deprecations, or required flags\n\n"
    "### References & Sources\n"
    "- [Source Title](URL): what was verified here\n"
  )


def _run_scout_loop(
  session: Any,
  title: str,
  system_prompt: str,
  user_prompt: str,
  allowed_tools: Set[str],
  max_loops: Optional[int] = None,
  log_label: str = "Scout agent",
  query: str = ""
) -> str:
  """Shared loop engine for scout agents."""
  discovery_model = session.get_discovery_model()
  loops_limit = max_loops if max_loops is not None else getattr(session.config, "discovery_loops", 50)

  logger.info(f"Starting {log_label} (model={discovery_model}, max_loops={loops_limit})")

  def _format_panel_content(status_text: str) -> Text:
    if query:
      trimmed_query = query.strip()
      if "\n" in trimmed_query:
        return Text.from_markup(
          f"[bold]Query:[/bold]\n[yellow]{escape(trimmed_query)}[/yellow]\n\n{status_text}"
        )
      return Text.from_markup(
        f"[bold]Query:[/bold] [yellow]{escape(trimmed_query)}[/yellow]\n\n{status_text}"
      )
    return Text.from_markup(status_text)

  # Filter tools for this scout mode
  scout_tools = [
    t for t in TOOLS_SCHEMA
    if t.get("type") == "function" and t.get("function", {}).get("name") in allowed_tools
  ]

  scout_messages: List[Dict[str, Any]] = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ]

  final_dossier = ""
  panels = [{
    "title": title,
    "content": _format_panel_content(f"Starting {log_label.lower()} with [bold cyan]{escape(discovery_model)}[/bold cyan]..."),
    "border_style": "cyan"
  }]

  max_retries = 3

  for loop_idx in range(loops_limit):
    actual_model, extra_body = session._resolve_model_and_provider(discovery_model)
    session._throttle_request()

    discovery_timeout = getattr(session.config, "discovery_timeout", None) or getattr(session.config, "api_timeout", 60.0)
    kwargs: Dict[str, Any] = {
      "model": actual_model,
      "messages": scout_messages,
      "tools": scout_tools,
      "stream": True,
      "timeout": discovery_timeout,
    }
    if extra_body:
      kwargs["extra_body"] = extra_body

    content_accumulated = ""
    tool_calls_accumulated: List[Dict[str, Any]] = []
    reasoning_accumulated = ""
    last_thinking_display_time = time.time()
    current_max_thinking = getattr(session.config, "max_thinking_chars", 12000)
    whitelist_thinking_this_turn = False
    usage_metadata = None
    api_succeeded = False

    panels[0]["content"] = _format_panel_content(
      f"Investigation step [bold yellow]{loop_idx + 1}/{loops_limit}[/bold yellow] (model: {escape(discovery_model)})..."
    )

    for attempt in range(1, max_retries + 1):
      content_accumulated = ""
      tool_calls_accumulated = []
      reasoning_accumulated = ""
      last_thinking_display_time = time.time()
      try:
        with optional_live(LiveScreenLayout(panels, None), console=console, enabled=not session.headless, refresh_per_second=12, transient=True) as live:
          session._active_live = live
          try:
            stream = session._create_completion(**kwargs)
            for chunk in stream:
              if hasattr(chunk, "usage") and chunk.usage:
                usage_metadata = chunk.usage
              elif hasattr(chunk, "model_extra") and chunk.model_extra and "usage" in chunk.model_extra:
                usage_metadata = chunk.model_extra["usage"]

              if not chunk.choices:
                continue

              choice = chunk.choices[0]
              delta = choice.delta

              val = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
              if val is None and hasattr(delta, "model_extra") and delta.model_extra:
                val = delta.model_extra.get("reasoning") or delta.model_extra.get("reasoning_content")
              if val is None and isinstance(delta, dict):
                val = delta.get("reasoning") or delta.get("reasoning_content")

              if val:
                if isinstance(val, str):
                  if reasoning_accumulated and val.startswith(reasoning_accumulated):
                    reasoning_accumulated = val
                  else:
                    reasoning_accumulated += val

              now = time.time()
              if reasoning_accumulated and not content_accumulated and not tool_calls_accumulated:
                if now - last_thinking_display_time >= 5.0:
                  last_thinking_display_time = now
                  t_count = session.count_tokens_estimate(reasoning_accumulated)
                  panels[0]["content"] = _format_panel_content(
                    f"Investigation step [bold yellow]{loop_idx + 1}/{loops_limit}[/bold yellow] (model: {escape(discovery_model)}) "
                    f"[dim]• Thinking: ~{t_count:,} tokens ({len(reasoning_accumulated):,} chars)...[/dim]"
                  )
                  if live is not None:
                    live.update(LiveScreenLayout(panels, None))

              max_thinking_chars = getattr(session.config, "max_thinking_chars", 12000)
              leeway = getattr(session.config, "max_thinking_leeway_chars", 2000)

              exceeded_budget = len(reasoning_accumulated) > current_max_thinking
              exceeded_hard_limit = len(reasoning_accumulated) > (current_max_thinking + leeway)

              should_abort = False
              if exceeded_hard_limit:
                should_abort = True
              elif exceeded_budget:
                if is_safe_to_interrupt(reasoning_accumulated):
                  should_abort = True

              if should_abort:
                if whitelist_thinking_this_turn:
                  should_abort = False
                elif not session.headless:
                  with session._pause_live():
                    session._print(f"\n[bold yellow]⚠️  {log_label} exceeded internal thinking budget.[/bold yellow]")
                    t_count = session.count_tokens_estimate(reasoning_accumulated)
                    session._print(
                      f"Accumulated thinking: {len(reasoning_accumulated)} characters "
                      f"(~{t_count:,} tokens, limit: {current_max_thinking} chars)."
                    )
                    session._print("What would you like to do?")
                    session._print(
                      "  [bold red]\\[s][/bold red]top (abort thinking stream) / "
                      "  [bold green]\\[l][/bold green]et this go (increase budget by another 12000 chars) / "
                      "  [bold cyan]\\[w][/bold cyan]hitelist for the current turn: ",
                      end=""
                    )
                    try:
                      if sys.stdin.isatty():
                        ready, _, _ = select.select([sys.stdin], [], [], 30.0)
                        if ready:
                          resp = sys.stdin.readline().strip().lower()
                        else:
                          session._print("\n[bold red]⚠️  Timeout: No response received within 30 seconds. Aborting thinking stream to save tokens.[/bold red]")
                          resp = "s"
                      else:
                        resp = input().strip().lower()
                    except (KeyboardInterrupt, EOFError):
                      resp = "s"
                    except Exception as err:
                      logger.warning(f"Error during timed input select: {err}. Falling back to standard input.")
                      try:
                        resp = input().strip().lower()
                      except (KeyboardInterrupt, EOFError):
                        resp = "s"

                    if resp in ("l", "let"):
                      current_max_thinking = len(reasoning_accumulated) + max_thinking_chars
                      should_abort = False
                      logger.info(f"User allowed thinking to continue in {log_label}. New budget: {current_max_thinking} chars.")
                    elif resp in ("w", "whitelist"):
                      whitelist_thinking_this_turn = True
                      should_abort = False
                      logger.info(f"User whitelisted thinking for the current turn in {log_label}.")
                    else:
                      should_abort = True
                      logger.info(f"User decided to stop thinking in {log_label}. Aborting stream.")

              is_repeating = detect_repetition_loop(reasoning_accumulated)
              exceeded_absolute_ceiling = len(reasoning_accumulated) > 48000

              if (is_repeating or exceeded_absolute_ceiling) and not content_accumulated and not delta.content and not tool_calls_accumulated and not delta.tool_calls:
                logger.warning(
                  f"Aborting stream: {log_label} got stuck in a repetitive loop or exceeded absolute thinking ceiling "
                  f"(len={len(reasoning_accumulated)}, repeating={is_repeating})."
                )
                if not session.headless:
                  with session._pause_live():
                    if is_repeating:
                      session._print(f"\n[bold red]⚠️  {log_label} got stuck in an infinite thinking loop. Aborting attempt to self-correct...[/bold red]")
                    else:
                      session._print(f"\n[bold red]⚠️  {log_label} exceeded absolute thinking limit (48k chars). Aborting attempt to self-correct...[/bold red]")
                if hasattr(stream, "close"):
                  try:
                    stream.close()
                  except Exception:
                    pass
                raise ThinkingBudgetExceeded(
                  f"{log_label} got stuck in an infinite thinking loop." if is_repeating else f"{log_label} exceeded absolute thinking limit.",
                  reasoning_accumulated
                )

              if (should_abort
                  and not content_accumulated
                  and not delta.content
                  and not tool_calls_accumulated
                  and not delta.tool_calls):
                if hasattr(stream, "close"):
                  try:
                    stream.close()
                  except Exception:
                    pass
                raise ThinkingBudgetExceeded(
                  f"{log_label} exceeded maximum internal thinking budget.",
                  reasoning_accumulated
                )

              if delta.content:
                content_accumulated += delta.content

              if delta.tool_calls:
                for tc in delta.tool_calls:
                  idx = tc.index
                  while len(tool_calls_accumulated) <= idx:
                    tool_calls_accumulated.append({
                      "id": None,
                      "type": "function",
                      "function": {"name": "", "arguments": ""}
                    })
                  item = tool_calls_accumulated[idx]
                  if tc.id:
                    item["id"] = tc.id
                  if tc.function:
                    if tc.function.name:
                      item["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                      item["function"]["arguments"] += tc.function.arguments

            api_succeeded = True
            break
          finally:
            session._active_live = None
      except ThinkingBudgetExceeded as e:
        logger.warning(f"{log_label} thinking budget exceeded on attempt {attempt}")
        if attempt < max_retries:
          if not session.headless:
            session._print(f"[bold yellow]⚠️  {log_label} exceeded thinking budget. Retrying with self-correction nudge (attempt {attempt}/{max_retries})...[/bold yellow]")
          scout_messages.append({
            "role": "user",
            "content": "I aborted your previous attempt because you exceeded the thinking budget. Please proceed directly to issuing tool calls or compiling your final context dossier without further internal reasoning."
          })
          time.sleep(1)
          continue
        else:
          if not session.headless:
            session._print(f"[bold red]Error: {log_label} repeatedly exceeded internal thinking budget without producing output.[/bold red]")
          break
      except Exception as e:
        logger.warning(f"{log_label} API attempt {attempt} failed: {e}")
        if attempt < max_retries:
          time.sleep(2 ** attempt)
        else:
          logger.exception(f"{log_label} API call failed permanently.")
          if not session.headless:
            console.print(f"[bold red]Error in {log_label}:[/bold red] {e}")
          return final_dossier or f"Error: {log_label} encountered an error: {e}"

    if not api_succeeded:
      break

    # Track token usage
    p_tok = getattr(usage_metadata, "prompt_tokens", None) if usage_metadata else None
    c_tok = getattr(usage_metadata, "completion_tokens", None) if usage_metadata else None
    if p_tok is None:
      p_tok = session._calculate_tokens_for_messages(scout_messages)
    if c_tok is None:
      c_tok = session.count_tokens_estimate(content_accumulated)

    if discovery_model not in session.model_usage:
      session.model_usage[discovery_model] = {"prompt_tokens": 0, "completion_tokens": 0}
    session.model_usage[discovery_model]["prompt_tokens"] += p_tok
    session.model_usage[discovery_model]["completion_tokens"] += c_tok

    # Fallback to text parsed tool calls if needed
    if not tool_calls_accumulated and content_accumulated:
      parsed_calls = session.extract_tool_calls_from_text(content_accumulated)
      if parsed_calls:
        tool_calls_accumulated = parsed_calls
        content_accumulated = ""

    # If no tool calls were made, the agent finished its investigation!
    if not tool_calls_accumulated:
      final_dossier = content_accumulated.strip()
      break

    # Format assistant message with tool calls
    assistant_msg: Dict[str, Any] = {
      "role": "assistant",
      "content": content_accumulated or None,
      "tool_calls": tool_calls_accumulated
    }
    scout_messages.append(assistant_msg)

    # Execute tools
    for tc in tool_calls_accumulated:
      if not tc.get("id"):
        tc["id"] = f"call_{uuid.uuid4().hex[:12]}"
      t_name = tc.get("function", {}).get("name", "")
      t_args_raw = tc.get("function", {}).get("arguments", "")

      try:
        args_parsed = json.loads(t_args_raw) if isinstance(t_args_raw, str) else (t_args_raw or {})
      except Exception:
        from chatty.utils import repair_json
        try:
          args_parsed = json.loads(repair_json(t_args_raw))
        except Exception as e:
          args_parsed = {}

      if t_name not in allowed_tools:
        t_result = (
          f"Error: Tool '{t_name}' is not permitted in this scout mode. "
          f"You only have access to: {', '.join(sorted(allowed_tools))}."
        )
      else:
        token = active_session_var.set(session)
        try:
          logger.info(f"{log_label} tool execution: {t_name} with {args_parsed}")
          t_result = execute_tool(t_name, args_parsed, session)
        except Exception as e:
          t_result = f"Error executing {t_name}: {str(e)}"
        finally:
          active_session_var.reset(token)

      scout_messages.append({
        "role": "tool",
        "tool_call_id": tc["id"],
        "name": t_name,
        "content": t_result
      })

  # If loop limit was reached without final output, request synthesis
  if not final_dossier:
    panels[0]["content"] = _format_panel_content(
      f"Investigation limit reached ([bold yellow]{loops_limit}/{loops_limit}[/bold yellow]). "
      f"Synthesizing final briefing (model: {escape(discovery_model)})..."
    )

    actual_model, extra_body = session._resolve_model_and_provider(discovery_model)
    scout_messages.append({
      "role": "user",
      "content": "Investigation turn limit reached. Please synthesize all gathered context and information into the final Markdown briefing now."
    })

    # Compress older tool results in scout_messages if needed to avoid context limit overflow
    total_msgs = len(scout_messages)
    for idx, msg in enumerate(scout_messages):
      if msg.get("role") == "tool" and idx < total_msgs - 10:
        c = msg.get("content") or ""
        max_tool_chars = getattr(session.config, "max_history_tool_chars", 1000)
        if len(c) > max_tool_chars:
          half = max_tool_chars // 2
          msg["content"] = f"{c[:half]}\n\n... [truncated] ...\n\n{c[-half:]}"

    discovery_timeout = getattr(session.config, "discovery_timeout", None) or getattr(session.config, "api_timeout", 60.0)
    kwargs = {
      "model": actual_model,
      "messages": scout_messages,
      "stream": True,
      "timeout": discovery_timeout,
    }
    if extra_body:
      kwargs["extra_body"] = extra_body

    content_accumulated = ""
    reasoning_accumulated = ""
    current_max_thinking = getattr(session.config, "max_thinking_chars", 12000)
    whitelist_thinking_this_turn = False
    last_thinking_display_time = time.time()
    usage_metadata = None
    for attempt in range(1, max_retries + 1):
      session._throttle_request()
      content_accumulated = ""
      reasoning_accumulated = ""
      last_thinking_display_time = time.time()
      try:
        with optional_live(
          LiveScreenLayout(panels, None),
          console=console,
          enabled=not session.headless,
          refresh_per_second=12,
          transient=True
        ) as live:
          session._active_live = live
          try:
            stream = session._create_completion(**kwargs)
            for chunk in stream:
              if hasattr(chunk, "usage") and chunk.usage:
                usage_metadata = chunk.usage
              elif hasattr(chunk, "model_extra") and chunk.model_extra and "usage" in chunk.model_extra:
                usage_metadata = chunk.model_extra["usage"]

              if not chunk.choices:
                continue

              choice = chunk.choices[0]
              delta = choice.delta

              val = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
              if val is None and hasattr(delta, "model_extra") and delta.model_extra:
                val = delta.model_extra.get("reasoning") or delta.model_extra.get("reasoning_content")
              if val is None and isinstance(delta, dict):
                val = delta.get("reasoning") or delta.get("reasoning_content")

              if val:
                if isinstance(val, str):
                  if reasoning_accumulated and val.startswith(reasoning_accumulated):
                    reasoning_accumulated = val
                  else:
                    reasoning_accumulated += val

              now = time.time()
              if reasoning_accumulated and not content_accumulated:
                if now - last_thinking_display_time >= 5.0:
                  last_thinking_display_time = now
                  t_count = session.count_tokens_estimate(reasoning_accumulated)
                  panels[0]["content"] = _format_panel_content(
                    f"Investigation limit reached ([bold yellow]{loops_limit}/{loops_limit}[/bold yellow]). "
                    f"Synthesizing final briefing (model: {escape(discovery_model)}) "
                    f"[dim]• Thinking: ~{t_count:,} tokens ({len(reasoning_accumulated):,} chars)...[/dim]"
                  )
                  if live is not None:
                    live.update(LiveScreenLayout(panels, None))

              max_thinking_chars = getattr(session.config, "max_thinking_chars", 12000)
              leeway = getattr(session.config, "max_thinking_leeway_chars", 2000)

              exceeded_budget = len(reasoning_accumulated) > current_max_thinking
              exceeded_hard_limit = len(reasoning_accumulated) > (current_max_thinking + leeway)

              should_abort = False
              if exceeded_hard_limit:
                should_abort = True
              elif exceeded_budget:
                if is_safe_to_interrupt(reasoning_accumulated):
                  should_abort = True

              if should_abort:
                if whitelist_thinking_this_turn:
                  should_abort = False
                elif not session.headless:
                  with session._pause_live():
                    session._print(f"\n[bold yellow]⚠️  {log_label} synthesis exceeded internal thinking budget.[/bold yellow]")
                    t_count = session.count_tokens_estimate(reasoning_accumulated)
                    session._print(
                      f"Accumulated thinking: {len(reasoning_accumulated)} characters "
                      f"(~{t_count:,} tokens, limit: {current_max_thinking} chars)."
                    )
                    session._print("What would you like to do?")
                    session._print(
                      "  [bold red]\\[s][/bold red]top (abort thinking stream) / "
                      "  [bold green]\\[l][/bold green]et this go (increase budget by another 12000 chars) / "
                      "  [bold cyan]\\[w][/bold cyan]hitelist for the current turn: ",
                      end=""
                    )
                    try:
                      if sys.stdin.isatty():
                        ready, _, _ = select.select([sys.stdin], [], [], 30.0)
                        if ready:
                          resp = sys.stdin.readline().strip().lower()
                        else:
                          session._print("\n[bold red]⚠️  Timeout: No response received within 30 seconds. Aborting thinking stream to save tokens.[/bold red]")
                          resp = "s"
                      else:
                        resp = input().strip().lower()
                    except (KeyboardInterrupt, EOFError):
                      resp = "s"
                    except Exception as err:
                      logger.warning(f"Error during timed input select: {err}. Falling back to standard input.")
                      try:
                        resp = input().strip().lower()
                      except (KeyboardInterrupt, EOFError):
                        resp = "s"

                    if resp in ("l", "let"):
                      current_max_thinking = len(reasoning_accumulated) + max_thinking_chars
                      should_abort = False
                      logger.info(f"User allowed thinking to continue in {log_label} synthesis. New budget: {current_max_thinking} chars.")
                    elif resp in ("w", "whitelist"):
                      whitelist_thinking_this_turn = True
                      should_abort = False
                      logger.info(f"User whitelisted thinking for the current turn in {log_label} synthesis.")
                    else:
                      should_abort = True
                      logger.info(f"User decided to stop thinking in {log_label} synthesis. Aborting stream.")

              is_repeating = detect_repetition_loop(reasoning_accumulated)
              exceeded_absolute_ceiling = len(reasoning_accumulated) > 48000

              if (is_repeating or exceeded_absolute_ceiling) and not content_accumulated and not delta.content:
                logger.warning(
                  f"Aborting synthesis stream: {log_label} got stuck in a repetitive loop or exceeded absolute thinking ceiling "
                  f"(len={len(reasoning_accumulated)}, repeating={is_repeating})."
                )
                if not session.headless:
                  with session._pause_live():
                    if is_repeating:
                      session._print(f"\n[bold red]⚠️  {log_label} synthesis got stuck in an infinite thinking loop. Aborting attempt to self-correct...[/bold red]")
                    else:
                      session._print(f"\n[bold red]⚠️  {log_label} synthesis exceeded absolute thinking limit (48k chars). Aborting attempt to self-correct...[/bold red]")
                if hasattr(stream, "close"):
                  try:
                    stream.close()
                  except Exception:
                    pass
                raise ThinkingBudgetExceeded(
                  f"{log_label} synthesis got stuck in an infinite thinking loop." if is_repeating else f"{log_label} synthesis exceeded absolute thinking limit.",
                  reasoning_accumulated
                )

              if should_abort and not content_accumulated and not delta.content:
                if hasattr(stream, "close"):
                  try:
                    stream.close()
                  except Exception:
                    pass
                raise ThinkingBudgetExceeded(
                  f"{log_label} synthesis exceeded maximum internal thinking budget.",
                  reasoning_accumulated
                )

              if delta.content:
                content_accumulated += delta.content

            if not content_accumulated and reasoning_accumulated:
              content_accumulated = reasoning_accumulated
            final_dossier = content_accumulated.strip()
            break
          finally:
            session._active_live = None
      except ThinkingBudgetExceeded as e:
        logger.warning(f"{log_label} synthesis thinking budget exceeded on attempt {attempt}")
        if attempt < max_retries:
          if not session.headless:
            session._print(f"[bold yellow]⚠️  {log_label} synthesis exceeded thinking budget. Retrying with self-correction nudge (attempt {attempt}/{max_retries})...[/bold yellow]")
          scout_messages.append({
            "role": "user",
            "content": "I aborted your previous synthesis attempt because you exceeded the thinking budget. Please synthesize all gathered context and information into the final Markdown briefing now without further internal reasoning."
          })
          time.sleep(1)
          continue
        else:
          if not session.headless:
            session._print(f"[bold red]Error: {log_label} synthesis repeatedly exceeded internal thinking budget.[/bold red]")
          break
      except Exception as e:
        logger.warning(f"{log_label} synthesis API attempt {attempt} failed: {e}")
        if attempt < max_retries:
          time.sleep(2 ** attempt)
        else:
          logger.exception(f"{log_label} synthesis failed permanently.")
          if not session.headless:
            console.print(f"[bold red]Error in {log_label} synthesis:[/bold red] {e}")

    # Track token usage
    if content_accumulated:
      p_tok = getattr(usage_metadata, "prompt_tokens", None) if usage_metadata else None
      c_tok = getattr(usage_metadata, "completion_tokens", None) if usage_metadata else None
      if p_tok is None:
        p_tok = session._calculate_tokens_for_messages(scout_messages)
      if c_tok is None:
        c_tok = session.count_tokens_estimate(content_accumulated)

      if discovery_model not in session.model_usage:
        session.model_usage[discovery_model] = {"prompt_tokens": 0, "completion_tokens": 0}
      session.model_usage[discovery_model]["prompt_tokens"] += p_tok
      session.model_usage[discovery_model]["completion_tokens"] += c_tok

    # Fallback if synthesis failed or returned empty: assemble findings from tools
    if not final_dossier:
      findings = []
      for m in scout_messages:
        if m.get("role") == "tool":
          t_res = (m.get("content") or "").strip()
          if t_res and not t_res.startswith("Error:"):
            first_lines = [l for l in t_res.splitlines() if l.strip()][:6]
            if first_lines:
              t_n = m.get("name", "tool")
              findings.append(f"**Evidence from `{t_n}`:**\n" + "\n".join(first_lines))
      if findings:
        final_dossier = (
          f"### Investigation Findings (Turn Limit Reached)\n\n"
          f"The {log_label.lower()} reached the maximum allowed turns ({loops_limit}). "
          f"Here is the context gathered during exploration:\n\n"
          + "\n\n".join(findings[-10:])
        )
      else:
        final_dossier = f"Investigation reached the maximum turn limit ({loops_limit} turns) without completing synthesis."

  return final_dossier


def run_discovery(session: Any, task: str, max_loops: Optional[int] = None) -> str:
  """Runs codebase reconnaissance using the discovery agent."""
  system_prompt = build_discovery_system_prompt(session, task)
  user_prompt = f"Investigate the codebase and assemble the reconnaissance dossier for this task:\n{task}"
  return _run_scout_loop(
    session=session,
    title="🔍 Discovery Agent",
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    allowed_tools=REPO_SCOUT_TOOL_NAMES,
    max_loops=max_loops,
    log_label="Discovery agent",
    query=task,
  )


def run_web_research(session: Any, query: str, max_loops: Optional[int] = None) -> str:
  """Runs web research using the scout agent."""
  system_prompt = build_web_research_system_prompt(query)
  user_prompt = f"Research the web and compile a comprehensive technical briefing for this query:\n{query}"
  return _run_scout_loop(
    session=session,
    title="🌐 Web Research Agent",
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    allowed_tools=WEB_RESEARCH_TOOL_NAMES,
    max_loops=max_loops,
    log_label="Web research agent",
    query=query,
  )
