import json
import logging
import os
import re
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

    kwargs: Dict[str, Any] = {
      "model": actual_model,
      "messages": scout_messages,
      "tools": scout_tools,
      "stream": True,
    }
    if extra_body:
      kwargs["extra_body"] = extra_body

    content_accumulated = ""
    tool_calls_accumulated: List[Dict[str, Any]] = []
    usage_metadata = None
    api_succeeded = False

    panels[0]["content"] = _format_panel_content(
      f"Investigation step [bold yellow]{loop_idx + 1}/{loops_limit}[/bold yellow] (model: {escape(discovery_model)})..."
    )

    for attempt in range(1, max_retries + 1):
      try:
        with optional_live(LiveScreenLayout(panels, None), console=console, enabled=not session.headless, refresh_per_second=12, transient=True) as live:
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
    try:
      actual_model, extra_body = session._resolve_model_and_provider(discovery_model)
      scout_messages.append({
        "role": "user",
        "content": "Investigation turn limit reached. Please synthesize all gathered context and information into the final Markdown briefing now."
      })
      kwargs = {
        "model": actual_model,
        "messages": scout_messages,
        "stream": False,
      }
      if extra_body:
        kwargs["extra_body"] = extra_body
      resp = session._create_completion(**kwargs)
      if resp.choices and resp.choices[0].message:
        final_dossier = resp.choices[0].message.content or ""
    except Exception as e:
      logger.warning(f"Error requesting {log_label} synthesis: {e}")

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
