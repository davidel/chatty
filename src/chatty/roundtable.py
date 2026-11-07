import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from chatty.tools import TOOLS_SCHEMA, execute_tool
from chatty.safety import active_session_var
from chatty.ui import optional_live, LiveScreenLayout
from chatty.llm import is_safe_to_interrupt, detect_repetition_loop, truncate_thinking_by_line
from chatty.utils import repair_json
from chatty.models_config import get_saved_models, get_project_chatty_dir

logger = logging.getLogger("chatty")
console = Console()

ROUNDTABLE_VERIFICATION_TOOL_NAMES: Set[str] = {
  "read_file",
  "search_grep",
  "locate_files",
  "get_outline",
  "find_symbol",
  "get_file_info",
  "search_web",
  "fetch_url",
}


class PointStatus(str, Enum):
  PROPOSED = "PROPOSED"
  CONFIRMED = "CONFIRMED"
  REVISED = "REVISED"
  CONTESTED = "CONTESTED"
  SPLIT = "SPLIT"


@dataclass
class RoundTablePoint:
  id: str
  statement: str
  status: str = PointStatus.PROPOSED
  rationale: str = ""
  counterexamples: List[str] = field(default_factory=list)
  tradeoff: str = ""
  history: List[Dict[str, Any]] = field(default_factory=list)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "id": self.id,
      "statement": self.statement,
      "status": self.status,
      "rationale": self.rationale,
      "counterexamples": self.counterexamples,
      "tradeoff": self.tradeoff,
      "history": self.history,
    }


def _extract_json_block(text: str) -> Optional[str]:
  """Extracts JSON block from markdown code fences or raw string."""
  if not text:
    return None
  code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
  if code_match:
    return code_match.group(1).strip()
  # Try bracket-to-bracket match
  first_brace = text.find("{")
  last_brace = text.rfind("}")
  if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
    return text[first_brace:last_brace + 1].strip()
  return None


def _parse_structured_json(text: str) -> Optional[Dict[str, Any]]:
  """Attempts to parse JSON, falling back to json repair."""
  candidate = _extract_json_block(text) or text.strip()
  try:
    data = json.loads(candidate)
    if isinstance(data, dict):
      return data
  except Exception:
    pass
  try:
    repaired = repair_json(candidate)
    data = json.loads(repaired)
    if isinstance(data, dict):
      return data
  except Exception:
    pass
  return None


def _format_points_for_prompt(points: List[RoundTablePoint]) -> str:
  """Formats current points as a clean Markdown list for model prompt context."""
  lines = []
  for p in points:
    stat_val = p.status.value if hasattr(p.status, "value") else str(p.status)
    status_str = f"[{stat_val}]"
    line = f"- **{p.id}** {status_str}: {p.statement}"
    if p.rationale:
      line += f"\n  *Rationale:* {p.rationale}"
    if p.counterexamples:
      line += f"\n  *Counterexamples noted:* {'; '.join(p.counterexamples)}"
    if p.tradeoff:
      line += f"\n  *Trade-off / Dissent:* {p.tradeoff}"
    lines.append(line)
  return "\n".join(lines)


def _format_points_table(points: List[RoundTablePoint]) -> Table:
  """Creates a Rich Table summarizing the current points and statuses."""
  table = Table(title="🏛️ Round Table Consensus Points", expand=True)
  table.add_column("Point", style="bold cyan", width=6)
  table.add_column("Status", width=12)
  table.add_column("Statement", style="white")
  table.add_column("Notes / Rationale", style="dim")

  status_styles = {
    PointStatus.PROPOSED: "[yellow]PROPOSED[/yellow]",
    PointStatus.CONFIRMED: "[bold green]CONFIRMED[/bold green]",
    PointStatus.REVISED: "[bold blue]REVISED[/bold blue]",
    PointStatus.CONTESTED: "[bold red]CONTESTED[/bold red]",
    PointStatus.SPLIT: "[magenta]SPLIT[/magenta]",
    "PROPOSED": "[yellow]PROPOSED[/yellow]",
    "CONFIRMED": "[bold green]CONFIRMED[/bold green]",
    "REVISED": "[bold blue]REVISED[/bold blue]",
    "CONTESTED": "[bold red]CONTESTED[/bold red]",
    "SPLIT": "[magenta]SPLIT[/magenta]",
  }

  for p in points:
    stat_val = p.status.value if hasattr(p.status, "value") else str(p.status)
    styled_status = status_styles.get(p.status, status_styles.get(stat_val, stat_val))
    note = p.tradeoff or (p.counterexamples[-1] if p.counterexamples else p.rationale)
    table.add_row(p.id, styled_status, p.statement, note or "")
  return table


def build_moderator_decomposition_prompt(query: str) -> str:
  return (
    f"USER INQUIRY / TASK:\n{query}\n\n"
    "You are the Moderator of an adversarial technical round table. "
    "Your first task is to break down the user's technical inquiry into a set of discrete, "
    "atomic, and verifiable points (named P1, P2, P3, ...).\n\n"
    "Each point must state a specific assertion, architecture choice, data structure, invariant, "
    "algorithm detail, or error-handling approach that can be independently verified, tested, or refuted.\n\n"
    "Output your analysis strictly in the following JSON format:\n"
    "```json\n"
    "{\n"
    "  \"points\": [\n"
    "    {\n"
    "      \"id\": \"P1\",\n"
    "      \"statement\": \"Precise assertion or technical choice\",\n"
    "      \"rationale\": \"Why this point is proposed\"\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "```"
  )


def build_expert_system_prompt(expert_idx: int, total_experts: int, expert_model: str) -> str:
  return (
    f"You are Expert {expert_idx} of {total_experts} (model: {expert_model}) in an adversarial "
    "verification round table.\n"
    "Your primary duty is to stress-test the proposed technical points (P1, P2, ...) and check "
    "each other's reasoning.\n\n"
    "CRITICAL ANTI-SYCOPHANCY RULES:\n"
    "1. Do not be agreeable for the sake of politeness. Sycophancy and empty praise are strictly rejected.\n"
    "2. Your explicit goal is to identify flaws, edge cases, off-by-one errors, race conditions, "
    "unhandled inputs, algorithmic inefficiencies, or false assumptions.\n"
    "3. You may ONLY mark a point as 'CONFIRMED' if you have actively verified that no counterexample exists "
    "and you state the formal invariant or rationale under 'verification_rationale'.\n"
    "4. If you find any flaw, mark it 'CONTESTED', provide a concrete 'counterexample', and propose a 'suggested_revision'.\n"
    "5. You have access to read-only inspection tools (read_file, search_grep, find_symbol, etc.) "
    "to check the real codebase or external documentation before making claims.\n\n"
    "Output your evaluation strictly in the following JSON format:\n"
    "```json\n"
    "{\n"
    "  \"reviews\": [\n"
    "    {\n"
    "      \"point_id\": \"P1\",\n"
    "      \"verdict\": \"CONFIRMED\",\n"
    "      \"verification_rationale\": \"Explicit invariant or proof of correctness\",\n"
    "      \"counterexample\": \"\"\n"
    "    },\n"
    "    {\n"
    "      \"point_id\": \"P2\",\n"
    "      \"verdict\": \"CONTESTED\",\n"
    "      \"verification_rationale\": \"\",\n"
    "      \"counterexample\": \"Concrete scenario or edge case where P2 fails\",\n"
    "      \"suggested_revision\": \"How P2 should be amended\"\n"
    "    }\n"
    "  ],\n"
    "  \"new_points\": [\n"
    "    \"P4: Optional newly identified essential point that was missing\"\n"
    "  ]\n"
    "}\n"
    "```"
  )


def build_moderator_reconciliation_prompt(
  points: List[RoundTablePoint],
  round_num: int,
  max_rounds: int,
  sequential_critiques: List[Dict[str, Any]]
) -> str:
  points_context = _format_points_for_prompt(points)
  critiques_json = json.dumps(sequential_critiques, indent=2)
  is_final_round = round_num >= max_rounds

  reconciliation_rules = (
    "RECONCILIATION RULES:\n"
    "1. Concrete, demonstrated counterexamples strictly outweigh generic confirmations.\n"
    "2. If an expert provided a valid counterexample, update the point's statement to incorporate "
    "the fix, set status to 'REVISED', and record the revision in 'rationale'.\n"
    "3. If all experts confirmed a point with valid rationales and no unaddressed counterexample remains, "
    "set status to 'CONFIRMED'.\n"
    "4. If a point remains contested with an active counterexample and rounds remain, mark it 'CONTESTED'.\n"
  )

  if is_final_round:
    reconciliation_rules += (
      "5. FINAL ROUND RULE: Because this is round "
      f"{round_num} of {max_rounds}, any points where experts remain in fundamental disagreement "
      "must be marked 'SPLIT'. In 'tradeoff', clearly articulate the competing perspectives (e.g. memory vs latency).\n"
    )

  return (
    f"You are the Moderator of the round table (Round {round_num}/{max_rounds}).\n"
    "Reconcile the sequential expert reviews and update the master points list.\n\n"
    f"CURRENT POINTS:\n{points_context}\n\n"
    f"EXPERT REVIEWS FROM THIS ROUND:\n{critiques_json}\n\n"
    f"{reconciliation_rules}\n"
    "Output the reconciled points strictly in the following JSON format:\n"
    "```json\n"
    "{\n"
    "  \"points\": [\n"
    "    {\n"
    "      \"id\": \"P1\",\n"
    "      \"statement\": \"Updated or confirmed statement\",\n"
    "      \"status\": \"CONFIRMED | REVISED | CONTESTED | SPLIT\",\n"
    "      \"rationale\": \"Why this status was assigned\",\n"
    "      \"counterexamples\": [\"Any recorded counterexamples\"],\n"
    "      \"tradeoff\": \"Document trade-off if SPLIT, else empty\"\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "```"
  )


def build_moderator_synthesis_prompt(
  query: str,
  points: List[RoundTablePoint],
  rounds_run: int
) -> str:
  points_context = _format_points_for_prompt(points)
  return (
    "You are the Moderator delivering the final synthesized consensus from the expert round table.\n\n"
    f"ORIGINAL QUERY: {query}\n"
    f"ROUNDS CONDUCTED: {rounds_run}\n\n"
    f"FINAL REFINED POINTS:\n{points_context}\n\n"
    "Please synthesize a comprehensive, rigorous final technical recommendation.\n"
    "Structure your response with:\n"
    "1. **Executive Summary**: Core decision and key consensus highlights.\n"
    "2. **Point-by-Point Consensus Analysis**: Detail each point (P1, P2, ...), its final status, "
    "invariants verified, and revisions made.\n"
    "3. **Trade-offs & Divergences**: For any points marked SPLIT, thoroughly explain the opposing viewpoints.\n"
    "4. **Actionable Implementation Plan**: Concrete, step-by-step guidance for implementing the solution."
  )


def _call_llm_stream(
  session: Any,
  model_id: str,
  messages: List[Dict[str, Any]],
  tools: Optional[List[Dict[str, Any]]] = None,
  panel_title: str = "Round Table",
  border_style: str = "cyan"
) -> Tuple[str, List[Dict[str, Any]], str]:
  """Calls an LLM with streaming into Rich panels. Returns (content, tool_calls, reasoning)."""
  actual_model, extra_body = session._resolve_model_and_provider(model_id)
  session._throttle_request()

  timeout = getattr(session.config, "roundtable_timeout", 120.0)
  kwargs: Dict[str, Any] = {
    "model": actual_model,
    "messages": messages,
    "stream": True,
    "timeout": timeout,
  }
  if tools:
    kwargs["tools"] = tools
  if extra_body:
    kwargs["extra_body"] = extra_body

  content_accumulated = ""
  tool_calls_accumulated: List[Dict[str, Any]] = []
  reasoning_accumulated = ""
  current_max_thinking = getattr(session.config, "max_thinking_chars", 8000)
  usage_metadata = None

  panels = [{
    "title": panel_title,
    "content": Text.from_markup(f"Connecting to [bold cyan]{escape(model_id)}[/bold cyan]..."),
    "border_style": border_style,
  }]

  with optional_live(LiveScreenLayout(panels, None), console=console, enabled=not session.headless, refresh_per_second=10, transient=True) as live:
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

        # Extract reasoning/thinking
        val = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if val is None and hasattr(delta, "model_extra") and delta.model_extra:
          val = delta.model_extra.get("reasoning") or delta.model_extra.get("reasoning_content")
        if val is None and isinstance(delta, dict):
          val = delta.get("reasoning") or delta.get("reasoning_content")
        if val and isinstance(val, str):
          if reasoning_accumulated and val.startswith(reasoning_accumulated):
            reasoning_accumulated = val
          else:
            reasoning_accumulated += val

        # Extract tool calls
        if hasattr(delta, "tool_calls") and delta.tool_calls:
          for tc in delta.tool_calls:
            idx = getattr(tc, "index", 0)
            while len(tool_calls_accumulated) <= idx:
              tool_calls_accumulated.append({
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
              })
            if hasattr(tc, "id") and tc.id:
              tool_calls_accumulated[idx]["id"] += tc.id
            if hasattr(tc, "function") and tc.function:
              fn = tc.function
              if hasattr(fn, "name") and fn.name:
                tool_calls_accumulated[idx]["function"]["name"] += fn.name
              if hasattr(fn, "arguments") and fn.arguments:
                tool_calls_accumulated[idx]["function"]["arguments"] += fn.arguments

        # Extract content
        if delta.content:
          content_accumulated += delta.content

        # Update Live Screen Panel
        status_lines = []
        if reasoning_accumulated.strip() and not content_accumulated:
          t_count = session.count_tokens_estimate(reasoning_accumulated)
          disp_think = truncate_thinking_by_line(reasoning_accumulated)
          status_lines.append(f"[dim]• Thinking (~{t_count} tok):[/dim]\n[italic dim]{escape(disp_think[-300:])}[/italic dim]")
        if content_accumulated:
          status_lines.append(escape(content_accumulated[-600:]))
        if tool_calls_accumulated:
          tc_names = [tc.get("function", {}).get("name") for tc in tool_calls_accumulated if tc.get("function", {}).get("name")]
          if tc_names:
            status_lines.append(f"[bold yellow]Tools:[/bold yellow] {', '.join(tc_names)}")

        if status_lines:
          panels[0]["content"] = Text.from_markup("\n".join(status_lines))

    except Exception as e:
      logger.exception(f"Error during stream from {model_id}: {e}")
      if not content_accumulated:
        content_accumulated = f"Error consulting {model_id}: {str(e)}"
    finally:
      session._active_live = None

  # Track tokens
  p_tok = getattr(usage_metadata, "prompt_tokens", 0) if usage_metadata else session.count_tokens_estimate(str(messages))
  c_tok = getattr(usage_metadata, "completion_tokens", 0) if usage_metadata else (
    session.count_tokens_estimate(content_accumulated) + session.count_tokens_estimate(reasoning_accumulated)
  )
  if model_id not in session.model_usage:
    session.model_usage[model_id] = {"prompt_tokens": 0, "completion_tokens": 0}
  session.model_usage[model_id]["prompt_tokens"] += p_tok
  session.model_usage[model_id]["completion_tokens"] += c_tok

  return content_accumulated, tool_calls_accumulated, reasoning_accumulated


def _run_expert_turn(
  session: Any,
  expert_idx: int,
  total_experts: int,
  expert_model: str,
  query: str,
  points: List[RoundTablePoint],
  round_num: int,
  prior_expert_critiques: List[Dict[str, Any]]
) -> Dict[str, Any]:
  """Runs a single expert turn with up to 2 tool loops, returning structured review dict."""
  allowed_tools = ROUNDTABLE_VERIFICATION_TOOL_NAMES
  verification_tools = [
    t for t in TOOLS_SCHEMA
    if t.get("type") == "function" and t.get("function", {}).get("name") in allowed_tools
  ]

  sys_prompt = build_expert_system_prompt(expert_idx, total_experts, expert_model)
  points_context = _format_points_for_prompt(points)

  user_prompt = (
    f"TASK / QUESTION:\n{query}\n\n"
    f"CURRENT CANDIDATE POINTS (ROUND {round_num}):\n{points_context}\n\n"
  )

  if prior_expert_critiques:
    critiques_summary = json.dumps(prior_expert_critiques, indent=2)
    user_prompt += (
      "CRITIQUES FROM PEERS EARLIER IN THIS ROUND:\n"
      f"{critiques_summary}\n\n"
      "Review the candidate points. Evaluate whether earlier peer objections are valid, "
      "find new edge cases or counterexamples, and verify remaining claims."
    )
  else:
    user_prompt += (
      "You are the first expert to review these candidate points this round. "
      "Identify flaws, edge cases, and verify correctness."
    )

  messages: List[Dict[str, Any]] = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": user_prompt},
  ]

  panel_title = f"🤖 Expert {expert_idx}/{total_experts} ({expert_model}) [Round {round_num}]"
  max_tool_loops = 2
  final_content = ""
  total_reasoning = ""

  for loop_idx in range(max_tool_loops + 1):
    tools = verification_tools if loop_idx < max_tool_loops else None
    content, tool_calls, reasoning = _call_llm_stream(
      session=session,
      model_id=expert_model,
      messages=messages,
      tools=tools,
      panel_title=panel_title,
      border_style="yellow",
    )
    if reasoning:
      total_reasoning += ("\n" + reasoning if total_reasoning else reasoning)

    if not tool_calls:
      final_content = content
      break

    # Format assistant message with tool calls (pruning internal reasoning from context)
    messages.append({
      "role": "assistant",
      "content": content or None,
      "tool_calls": tool_calls,
    })

    # Execute tools
    for tc in tool_calls:
      if not tc.get("id"):
        tc["id"] = f"call_{uuid.uuid4().hex[:12]}"
      t_name = tc.get("function", {}).get("name", "")
      t_args_raw = tc.get("function", {}).get("arguments", "")

      try:
        args_parsed = json.loads(t_args_raw) if isinstance(t_args_raw, str) else (t_args_raw or {})
      except Exception:
        try:
          args_parsed = json.loads(repair_json(t_args_raw))
        except Exception:
          args_parsed = {}

      if t_name not in allowed_tools:
        t_result = f"Error: Tool '{t_name}' not permitted in round table."
      else:
        token = active_session_var.set(session)
        try:
          t_result = execute_tool(t_name, args_parsed, session)
        except Exception as e:
          t_result = f"Error executing {t_name}: {str(e)}"
        finally:
          active_session_var.reset(token)

      messages.append({
        "role": "tool",
        "tool_call_id": tc["id"],
        "name": t_name,
        "content": str(t_result),
      })

  # Parse structured JSON review from the final content
  parsed_data = _parse_structured_json(final_content)
  review_record: Dict[str, Any] = {
    "expert_idx": expert_idx,
    "model": expert_model,
    "round": round_num,
    "raw_output": final_content,
    "reviews": parsed_data.get("reviews", []) if parsed_data else [],
    "new_points": parsed_data.get("new_points", []) if parsed_data else [],
    "reasoning_summary": truncate_thinking_by_line(total_reasoning) if total_reasoning else "",
  }
  return review_record


def _format_audit_log(
  query: str,
  panel_models: List[str],
  moderator_model: str,
  total_rounds: int,
  points: List[RoundTablePoint],
  transcript: List[Dict[str, Any]],
  final_synthesis: str
) -> str:
  """Builds a comprehensive markdown audit log of the entire deliberation."""
  timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
  models_str = ", ".join(panel_models)

  lines = [
    "# 🏛️ Round Table Deliberation Audit Log",
    f"- **Timestamp:** {timestamp}",
    f"- **Topic / Query:** {query}",
    f"- **Panel Members ({len(panel_models)}):** {models_str}",
    f"- **Moderator:** {moderator_model}",
    f"- **Total Rounds:** {total_rounds}",
    "",
    "## Final Status Table",
    "| Point | Status | Statement | Rationale / Trade-off |",
    "|---|---|---|---|",
  ]

  for p in points:
    note = p.tradeoff or (p.counterexamples[-1] if p.counterexamples else p.rationale)
    note_clean = note.replace("\n", " ").strip()
    stmt_clean = p.statement.replace("\n", " ").strip()
    lines.append(f"| **{p.id}** | `{p.status}` | {stmt_clean} | {note_clean} |")

  lines.extend(["", "## Round-by-Round Deliberation Transcript", ""])

  for entry in transcript:
    role = entry.get("role", "")
    rd = entry.get("round", 1)
    if role == "moderator_decomposition":
      lines.append(f"### Initial Decomposition by Moderator ({moderator_model})")
      lines.append(entry.get("raw", ""))
      lines.append("")
    elif role == "expert":
      m = entry.get("model", "")
      idx = entry.get("expert_idx", 1)
      lines.append(f"### Round {rd} — Expert {idx}: `{m}`")
      if entry.get("reasoning_summary"):
        lines.append(f"> **Reasoning Scratchpad:**\n> {entry.get('reasoning_summary')}\n")
      lines.append(entry.get("raw_output", ""))
      lines.append("")
    elif role == "moderator_reconciliation":
      lines.append(f"### Round {rd} — Moderator Reconciliation ({moderator_model})")
      lines.append(entry.get("raw", ""))
      lines.append("")

  lines.extend([
    "## Final Synthesized Recommendation",
    final_synthesis,
    "",
  ])
  return "\n".join(lines)


def run_roundtable(
  session: Any,
  query: str,
  models: Optional[List[str]] = None,
  moderator: Optional[str] = None,
  max_rounds: int = 3
) -> str:
  """Runs the full round-table deliberation workflow."""
  expanded_query = session._expand_query_files(query) if hasattr(session, "_expand_query_files") else query
  saved = get_saved_models(getattr(session, "sandbox", None))

  # Resolve panel models
  panel_models = models or getattr(session, "roundtable_models", None) or saved.get("roundtable_models")
  if not panel_models:
    # Fallback to active model repeated or default
    active_m = session.model or "default"
    panel_models = [active_m]

  # Resolve moderator model
  moderator_model = moderator or getattr(session, "roundtable_moderator", None) or saved.get("roundtable_moderator") or session.model

  console.print(Panel(
    Text.from_markup(
      f"[bold cyan]Topic:[/bold cyan] {escape(query)}\n"
      f"[bold green]Panel ({len(panel_models)}):[/bold green] {', '.join(panel_models)}\n"
      f"[bold purple]Moderator:[/bold purple] {moderator_model}\n"
      f"[bold yellow]Max Rounds:[/bold yellow] {max_rounds}"
    ),
    title="🏛️ Summoning the Round Table",
    border_style="cyan"
  ))

  transcript: List[Dict[str, Any]] = []

  # STEP 1: Moderator decomposes query into initial candidate points (P1, P2, ...)
  decomp_prompt = build_moderator_decomposition_prompt(expanded_query)
  decomp_messages = [
    {"role": "system", "content": "You are a senior technical architect acting as a round-table moderator."},
    {"role": "user", "content": decomp_prompt},
  ]
  decomp_content, _, _ = _call_llm_stream(
    session=session,
    model_id=moderator_model,
    messages=decomp_messages,
    panel_title="👑 Moderator: Decomposing Problem into Points",
    border_style="purple",
  )
  transcript.append({
    "role": "moderator_decomposition",
    "raw": decomp_content,
  })

  parsed_decomp = _parse_structured_json(decomp_content)
  points: List[RoundTablePoint] = []
  if parsed_decomp and "points" in parsed_decomp:
    for idx, raw_pt in enumerate(parsed_decomp["points"]):
      pid = raw_pt.get("id") or f"P{idx + 1}"
      stmt = raw_pt.get("statement") or str(raw_pt)
      rat = raw_pt.get("rationale") or ""
      points.append(RoundTablePoint(id=pid, statement=stmt, rationale=rat, status=PointStatus.PROPOSED))
  else:
    # Fallback if decomposition JSON failed
    points = [
      RoundTablePoint(id="P1", statement=decomp_content.strip()[:300], status=PointStatus.PROPOSED)
    ]

  console.print(_format_points_table(points))

  # STEP 2: Multi-Round Sequential Deliberation Loop
  rounds_completed = 0
  for round_num in range(1, max_rounds + 1):
    rounds_completed = round_num
    console.print(f"\n[bold yellow]─── Round {round_num}/{max_rounds} ───[/bold yellow]")

    round_critiques: List[Dict[str, Any]] = []

    # Sequential pass across all experts
    for exp_idx, exp_model in enumerate(panel_models, 1):
      review = _run_expert_turn(
        session=session,
        expert_idx=exp_idx,
        total_experts=len(panel_models),
        expert_model=exp_model,
        query=expanded_query,
        points=points,
        round_num=round_num,
        prior_expert_critiques=round_critiques,
      )
      round_critiques.append(review)
      transcript.append({
        "role": "expert",
        "round": round_num,
        "expert_idx": exp_idx,
        "model": exp_model,
        "raw_output": review["raw_output"],
        "reasoning_summary": review.get("reasoning_summary", ""),
      })

    # Moderator Reconciliation Turn
    reconcile_prompt = build_moderator_reconciliation_prompt(
      points=points,
      round_num=round_num,
      max_rounds=max_rounds,
      sequential_critiques=round_critiques,
    )
    reconcile_messages = [
      {"role": "system", "content": "You are the round-table moderator reconciling expert evaluations."},
      {"role": "user", "content": reconcile_prompt},
    ]
    reconcile_content, _, _ = _call_llm_stream(
      session=session,
      model_id=moderator_model,
      messages=reconcile_messages,
      panel_title=f"👑 Moderator: Reconciling Round {round_num}",
      border_style="purple",
    )
    transcript.append({
      "role": "moderator_reconciliation",
      "round": round_num,
      "raw": reconcile_content,
    })

    parsed_reconcile = _parse_structured_json(reconcile_content)
    if parsed_reconcile and "points" in parsed_reconcile:
      updated_points: List[RoundTablePoint] = []
      for pt_data in parsed_reconcile["points"]:
        pid = pt_data.get("id", f"P{len(updated_points)+1}")
        stmt = pt_data.get("statement", "")
        stat = pt_data.get("status", PointStatus.PROPOSED)
        rat = pt_data.get("rationale", "")
        ce = pt_data.get("counterexamples", [])
        to = pt_data.get("tradeoff", "")
        updated_points.append(RoundTablePoint(
          id=pid,
          statement=stmt,
          status=stat,
          rationale=rat,
          counterexamples=ce if isinstance(ce, list) else [str(ce)],
          tradeoff=to,
        ))
      if updated_points:
        points = updated_points

    console.print(_format_points_table(points))

    # Check for early consensus: if all points are CONFIRMED
    all_confirmed = all(p.status == PointStatus.CONFIRMED for p in points)
    if all_confirmed:
      console.print("[bold green]✨ Full consensus achieved across all points![/bold green]")
      break

  # STEP 3: Final Moderator Synthesis
  synthesis_prompt = build_moderator_synthesis_prompt(expanded_query, points, rounds_completed)
  synth_messages = [
    {"role": "system", "content": "You are the round-table moderator compiling the final consensus synthesis."},
    {"role": "user", "content": synthesis_prompt},
  ]
  final_synthesis, _, _ = _call_llm_stream(
    session=session,
    model_id=moderator_model,
    messages=synth_messages,
    panel_title="👑 Moderator: Final Consensus Synthesis",
    border_style="green",
  )

  # STEP 4: Persist Last Deliberation Audit Log
  audit_md = _format_audit_log(
    query=query,
    panel_models=panel_models,
    moderator_model=moderator_model,
    total_rounds=rounds_completed,
    points=points,
    transcript=transcript,
    final_synthesis=final_synthesis,
  )

  chatty_dir = get_project_chatty_dir(getattr(session, "sandbox", None), create=True)
  if chatty_dir:
    log_path = os.path.join(chatty_dir, "roundtable_last.md")
    try:
      with open(log_path, "w", encoding="utf-8") as f:
        f.write(audit_md)
      logger.info(f"Saved round table audit log to {log_path}")
    except Exception as e:
      logger.error(f"Failed to write roundtable audit log: {e}")

  return final_synthesis
