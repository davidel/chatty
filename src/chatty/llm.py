import os
import time
import json
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

import openai
from chatty.tools import execute_tool, get_available_formatters, TOOLS_SCHEMA
from chatty.utils import count_tokens, truncate_output, sanitize_tool_output, repair_json
from chatty.ui import optional_live, LazyMarkdown
from chatty.safety import active_session_var

from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.markup import escape

logger = logging.getLogger("chatty")
console = Console()


def _invalidate_token_cache(self):
  self._cached_history_tokens = None


def _calculate_tokens_for_messages(self, messages: List[Dict[str, Any]]) -> int:
  total_tokens = 0
  if messages:
    sys_msg = messages[0]
    total_tokens += count_tokens(sys_msg.get("content") or "")
    for msg in messages[1:]:
      content = msg.get("content") or ""
      if msg.get("tool_calls"):
        content += json.dumps(msg["tool_calls"])
      if msg.get("tool_call_id"):
        content += msg["tool_call_id"]
      total_tokens += count_tokens(content) + 12
  return total_tokens


def init_client(self):
  """Initializes or updates the OpenAI client based on active settings."""
  if self.provider == "ollama":
    base = self.url or "http://localhost:11434/v1"
    self.client = openai.OpenAI(
      base_url=base,
      api_key="ollama"  # placeholder key
    )
  else:  # openrouter
    base = self.url or "https://openrouter.ai/api/v1"
    key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
      self._print(
        "[bold red]Warning:[/bold red] OpenRouter API key is not configured. "
        "Use [cyan]/api_key <key>[/cyan] or set the [cyan]OPENROUTER_API_KEY[/cyan] environment variable."
      )
      key = "missing_api_key"
    self.client = openai.OpenAI(
      base_url=base,
      api_key=key,
      default_headers={
        "HTTP-Referer": "https://github.com/davidel/chatty",
        "X-Title": "Chatty"
      }
    )


def _throttle_request(self):
  """Ensures at least 1.5 seconds have elapsed since the last API request or response."""
  now = time.time()
  elapsed = now - self.last_api_call_time
  min_delay = 1.5
  if elapsed < min_delay:
    sleep_needed = min_delay - elapsed
    logger.info(f"Pacing API requests: sleeping for {sleep_needed:.2f}s...")
    time.sleep(sleep_needed)
  self.last_api_call_time = time.time()


def _format_api_error(self, e: Exception) -> str:
  """Formats an API exception, appending rate-limit headers if available."""
  err_msg = str(e)
  response = getattr(e, "response", None)
  if response is not None:
    headers = getattr(response, "headers", None)
    if headers:
      rate_limit_info = []
      for header_name in ["x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset", "retry-after"]:
        val = headers.get(header_name)
        if val is not None:
          rate_limit_info.append(f"{header_name}: {val}")
      if rate_limit_info:
        err_msg += f" | Headers: ({', '.join(rate_limit_info)})"
  return err_msg


def _is_retryable_exception(self, e: Exception) -> bool:
  """Checks if an API exception is transient or rate-limit related and should be retried."""
  err_msg = str(e).lower()
  if isinstance(e, openai.APIConnectionError):
    return True
  if isinstance(e, openai.RateLimitError):
    return True
  if isinstance(e, openai.APIStatusError):
    status_code = getattr(e, "status_code", None)
    if status_code is not None:
      if status_code >= 500 or status_code == 429:
        return True
      if status_code == 400:
        indicators = [
          "high-frequency", "non-compliant", "rate limit", "rate-limit",
          "rate_limited", "rate-limited", "too many requests", "comply with the platform",
          "usage agreement", "appeal, contact", "provider returned error",
          "risk_control", "risk control"
        ]
        if any(ind in err_msg for ind in indicators):
          return True
  if isinstance(e, openai.APIError):
    indicators = [
      "high-frequency", "non-compliant", "rate limit", "rate-limit",
      "rate_limited", "rate-limited", "too many requests", "comply with the platform",
      "usage agreement", "appeal, contact", "provider returned error",
      "risk_control", "risk control"
    ]
    if any(ind in err_msg for ind in indicators):
      return True
  return False


def _resolve_model_and_provider(self, model_name: str) -> Tuple[str, Optional[Dict[str, Any]]]:
  """Resolves model name and provider preferences (if colon syntax is present)."""
  if not model_name or ":" not in model_name:
    return model_name, None
  parts = model_name.rsplit(":", 1)
  base_model = parts[0]
  suffix = parts[1]
  if suffix in ("free", "nitro", "floor"):
    return model_name, None
  extra_body = {
    "provider": {
      "order": [suffix],
      "allow_fallbacks": False
    }
  }
  return base_model, extra_body


def get_oracle_model(self) -> str:
  """Returns the configured oracle model, or determines a default based on provider."""
  oracle_model = getattr(self.config, "oracle_model", None)
  if oracle_model:
    return oracle_model
  if self.provider == "openrouter":
    return "google/gemini-2.5-pro"
  other_models = [m for m in self.models if m != self.model]
  if other_models:
    return other_models[0]
  return self.model


def consult_oracle(self, query: str) -> str:
  """Consults an oracle model for suggestions/assistance."""
  oracle_model = self.get_oracle_model()
  messages = [
    {
      "role": "system",
      "content": (
        "You are an AI oracle. You are assisting another AI agent that is currently "
        "stuck or needs advice on a difficult logic, programming, or reasoning step. "
        "Provide clear, concise, and highly accurate suggestions, code, or solutions "
        "to help the agent proceed."
      )
    },
    {
      "role": "user",
      "content": query
    }
  ]
  logger.info(f"Consulting oracle (model={oracle_model}) with query: {query}")
  content_accumulated = ""
  panel = Panel("Connecting to Oracle LLM...", title="🔮 Oracle", border_style="purple")
  max_retries = 3
  for attempt in range(1, max_retries + 1):
    try:
      with optional_live(Group(panel), console=console, enabled=not self.headless, refresh_per_second=12, transient=True) as live:
        actual_model, extra_body = self._resolve_model_and_provider(oracle_model)
        self._throttle_request()
        kwargs = {
          "model": actual_model,
          "messages": messages,
          "stream": True
        }
        if extra_body:
          kwargs["extra_body"] = extra_body
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
          if not chunk.choices:
            continue
          choice = chunk.choices[0]
          delta = choice.delta
          if delta.content:
            content_accumulated += delta.content
            panel = Panel(LazyMarkdown(content_accumulated), title="🔮 Oracle", border_style="purple")
            live.update(Group(panel))
      self.last_api_call_time = time.time()
      if content_accumulated:
        self._print(Panel(Markdown(content_accumulated), title="🔮 Oracle", border_style="purple"))
      else:
        self._print("[bold red]Oracle returned an empty response.[/bold red]")
      break
    except Exception as e:
      self.last_api_call_time = time.time()
      logger.exception("Error during oracle consultation")
      if attempt < max_retries and self._is_retryable_exception(e):
        backoff_time = 2 ** attempt
        formatted_err = self._format_api_error(e)
        self._print(f"[bold yellow]⚠️  Error calling Oracle API: {formatted_err}. Retrying in {backoff_time}s (attempt {attempt}/{max_retries})...[/bold yellow]")
        time.sleep(backoff_time)
        content_accumulated = ""
        continue
      else:
        formatted_err = self._format_api_error(e)
        err_msg = f"Error during oracle consultation: {formatted_err}"
        self._print(f"[bold red]{err_msg}[/bold red]")
        return err_msg
  return content_accumulated or "Error: Oracle returned an empty response."


def prune_history(self, log: bool = True) -> List[Dict[str, Any]]:
  """Prunes conversation history to respect the configured context size, compressing older tool outputs."""
  sys_prompt = self.get_active_system_prompt()
  system_msg = {"role": "system", "content": sys_prompt}
  if self.prompt_caching:
    system_msg["cache_control"] = {"type": "ephemeral"}
  sys_tokens = count_tokens(sys_prompt)
  
  if sys_tokens >= self.context_size:
    return [system_msg]
    
  processed_messages = []
  total_msgs = len(self.messages)
  
  for idx, msg in enumerate(self.messages):
    cloned_msg = dict(msg)
    # Compress tool outputs that are not part of the active window
    if cloned_msg.get("role") == "tool" and idx < total_msgs - self.history_keep_messages:
      content = cloned_msg.get("content") or ""
      if len(content) > self.max_history_tool_chars:
        half = self.max_history_tool_chars // 2
        truncated_len = len(content) - self.max_history_tool_chars
        cloned_msg["content"] = (
          f"{content[:half]}\n\n"
          f"... [TRUNCATED {truncated_len} CHARACTERS OF HISTORICAL TOOL OUTPUT] ...\n\n"
          f"{content[-half:]}"
        )
    if cloned_msg.get("role") == "assistant" and self.provider != "openrouter":
      for field in ["reasoning", "reasoning_content", "reasoning_details", "thought_signature"]:
        cloned_msg.pop(field, None)
    processed_messages.append(cloned_msg)
    
  pruned = []
  accumulated_tokens = sys_tokens
  
  # Process from newest to oldest
  for msg in reversed(processed_messages):
    content = msg.get("content") or ""
    # Estimate tool call tokens
    if msg.get("tool_calls"):
      content += json.dumps(msg["tool_calls"])
    if msg.get("tool_call_id"):
      content += msg["tool_call_id"]
      
    msg_tokens = count_tokens(content) + 12  # add safety overhead per message structure
    
    if accumulated_tokens + msg_tokens > self.context_size:
      break
      
    pruned.insert(0, msg)
    accumulated_tokens += msg_tokens
    
  if log:
    logger.info(f"Pruning history: kept {len(pruned)} out of {total_msgs} messages (accumulated tokens: {accumulated_tokens})")
    pruned_count = total_msgs - len(pruned)
    if pruned_count > 0:
      logger.warning(f"Context window limit reached. Pruned {pruned_count} messages from history.")
      self._print(
        f"\n[bold yellow]⚠️  Context Warning: {pruned_count} older message(s) were pruned from history "
        f"to fit the context size limit ({self.context_size} tokens).[/bold yellow]"
      )
      # If the very first user prompt is no longer in the pruned message history
      if self.messages and self.messages[0] not in pruned:
        self._print(
          "[bold red]⚠️  Critical: Your initial prompt/instructions have been pruned from context! "
          "The AI may lose track of the overall goal. Consider running '/compress' to reload a summary recap.[/bold red]\n"
        )
  # Filter orphaned tool messages
  defined_ids = set()
  for msg in pruned:
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
      for tc in msg["tool_calls"]:
        defined_ids.add(tc.get("id"))
        
  final_pruned = []
  for msg in pruned:
    if msg.get("role") == "tool":
      t_id = msg.get("tool_call_id")
      if t_id not in defined_ids:
        continue
    final_pruned.append(msg)
    
  if self.prompt_caching and final_pruned:
    final_pruned = [dict(msg) for msg in final_pruned]
    final_pruned[-1]["cache_control"] = {"type": "ephemeral"}
    if len(final_pruned) >= 2:
      final_pruned[-2]["cache_control"] = {"type": "ephemeral"}
      
  return [system_msg] + final_pruned


def _log_llm_request(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> None:
  """Logs detailed information about the LLM request in DEBUG mode."""
  if not logger.isEnabledFor(logging.DEBUG):
    return

  # Mask headers
  headers = getattr(self.client, "default_headers", {})
  masked_headers = {}
  for k, v in headers.items():
    if k.lower() in ("authorization", "api-key", "x-api-key"):
      if isinstance(v, str):
        if len(v) > 12:
          masked_headers[k] = v[:8] + "..." + v[-4:]
        else:
          masked_headers[k] = "..."
      else:
        masked_headers[k] = "..."
    else:
      masked_headers[k] = v

  # Mask API key
  api_key = getattr(self.client, "api_key", None)
  masked_key = "None"
  if api_key:
    if len(api_key) > 12:
      masked_key = api_key[:8] + "..." + api_key[-4:]
    else:
      masked_key = "..."

  logger.debug("=== LLM Request Details ===")
  logger.debug(f"Provider: {self.provider}")
  logger.debug(f"Model: {self.model}")
  logger.debug(f"Base URL: {getattr(self.client, 'base_url', 'Unknown')}")
  logger.debug(f"API Key: {masked_key}")
  logger.debug(f"Default Headers: {masked_headers}")
  logger.debug(f"Timeout: {getattr(self.client, 'timeout', 'Default')}")
  logger.debug(f"Max Retries: {getattr(self.client, 'max_retries', 'Default')}")
  logger.debug(f"Request Messages ({len(messages)}):")
  try:
    logger.debug(json.dumps(messages, indent=2, default=str))
  except Exception as e:
    logger.debug(f"Error serializing request messages: {e}")
    logger.debug(str(messages))

  if tools:
    logger.debug(f"Request Tools ({len(tools)}):")
    try:
      logger.debug(json.dumps(tools, indent=2, default=str))
    except Exception as e:
      logger.debug(f"Error serializing request tools: {e}")
      logger.debug(str(tools))
  else:
    logger.debug("Request Tools: None")
  logger.debug("===========================")


def _log_llm_response_summary(
  self,
  content_accumulated: str,
  tool_calls_accumulated: List[Dict[str, Any]],
  extra_fields_accumulated: Dict[str, Any],
  finish_reason: Optional[str] = None,
  usage: Optional[Any] = None,
  response_model: Optional[str] = None,
  system_fingerprint: Optional[str] = None,
  chunk_id: Optional[str] = None
) -> None:
  """Logs detailed summary of the LLM response in DEBUG mode."""
  if not logger.isEnabledFor(logging.DEBUG):
    return

  logger.debug("=== LLM Response Details ===")
  logger.debug(f"Chunk ID: {chunk_id}")
  logger.debug(f"Response Model: {response_model}")
  logger.debug(f"System Fingerprint: {system_fingerprint}")
  logger.debug(f"Finish Reason: {finish_reason}")

  if usage:
    if hasattr(usage, "prompt_tokens"):
      logger.debug(f"Usage: prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}, total_tokens={usage.total_tokens}")
    elif isinstance(usage, dict):
      logger.debug(f"Usage: prompt_tokens={usage.get('prompt_tokens')}, completion_tokens={usage.get('completion_tokens')}, total_tokens={usage.get('total_tokens')}")
    else:
      logger.debug(f"Usage: {usage}")
  else:
    logger.debug("Usage: Not provided/available")

  for field, val in extra_fields_accumulated.items():
    if val is not None:
      logger.debug(f"Extra Field ({field}): {val}")

  if content_accumulated:
    logger.debug(f"Assistant response content ({len(content_accumulated)} chars):")
    logger.debug(content_accumulated)
  else:
    logger.debug("Assistant response content: None")

  if tool_calls_accumulated:
    logger.debug(f"Assistant response tool calls ({len(tool_calls_accumulated)}):")
    try:
      logger.debug(json.dumps(tool_calls_accumulated, indent=2, default=str))
    except Exception as e:
      logger.debug(f"Error serializing tool calls: {e}")
      logger.debug(str(tool_calls_accumulated))
  else:
    logger.debug("Assistant response tool calls: None")
  logger.debug("============================")


def is_safe_to_interrupt(text: str) -> bool:
  """Checks if the accumulated text is at a safe sentence/markdown boundary to interrupt."""
  # If we are inside an unfinished code block, do not interrupt
  if text.count("```") % 2 != 0:
    return False

  # If we are inside inline code, do not interrupt
  if text.count("`") % 2 != 0:
    return False

  # Check basic bracket/parenthesis balancing to avoid breaking in the middle of a math expression or reference
  if text.count("(") != text.count(")"):
    return False
  if text.count("[") != text.count("]"):
    return False
  if text.count("{") != text.count("}"):
    return False

  stripped = text.rstrip()
  if not stripped:
    return False

  # If we just finished a code block, that is a clean boundary
  if stripped.endswith("```"):
    return True

  # Check if the last character of the stripped text is punctuation,
  # or punctuation followed by closing quotes/parentheses
  if re.search(r"[.!?]['\")\]]*$", stripped):
    return True

  # Or ends with a newline, which is a paragraph or line break
  if text.endswith("\n") or text.endswith("\r"):
    return True

  return False


class ThinkingBudgetExceeded(RuntimeError):
  """Raised when the LLM's internal thinking stream exceeds the configured limit."""
  def __init__(self, message: str, accumulated_thinking: str):
    super().__init__(message)
    self.accumulated_thinking = accumulated_thinking


def run_llm_cycle(self):
  """Executes a full inference cycle, resolving tool calls recursively."""
  self.load_skills()
  max_tool_loops = self.max_loops
  loop_count = 0
  logger.info(f"Starting LLM cycle. Max sequential tool loops: {max_tool_loops}")
  whitelist_thinking_this_turn = False
  
  while loop_count < max_tool_loops:
    self.current_loop = loop_count + 1
    max_retries = 3
    api_succeeded = False
    finish_reason = None
    
    last_aborted_thinking = None
    for attempt in range(1, max_retries + 1):
      # Prepare message payloads based on limit settings
      active_messages = self.prune_history()
      
      # If this is a retry and the last message in active_messages is a tool message,
      # append a temporary user message nudge to bypass trailing tool message chat template issues
      if attempt > 1 and active_messages and active_messages[-1].get("role") == "tool":
        active_messages.append({
          "role": "user",
          "content": "Please continue the task using the tool output above."
        })
      
      # If the previous attempt was aborted due to an internal thinking loop, feed it back
      if last_aborted_thinking:
        active_messages.append({
          "role": "user",
          "content": (
            "⚠️ The system aborted your previous attempt because you got stuck in an internal thinking loop. "
            "Review your thinking path below, identify the repetition/deadlock, and use a completely different approach or respond directly:\n\n"
            f"--- ABORTED THINKING PATH ---\n{last_aborted_thinking}\n------------------------------"
          )
        })
        last_aborted_thinking = None
      
      # Cache token count for the status bar during streaming
      self._cached_history_tokens = self._calculate_tokens_for_messages(active_messages)

      # Start LLM stream call
      tool_calls_accumulated = []
      content_accumulated = ""
      extra_fields_accumulated = {
        "reasoning": None,
        "reasoning_content": None,
        "reasoning_details": None,
        "thought_signature": None
      }
      
      logger.info(f"Loop {loop_count + 1}/{max_tool_loops} (Attempt {attempt}/{max_retries}): Sending request to LLM (model={self.model}) with {len(active_messages)} messages")
      try:
        # Live rendering console helper
        panel = Panel("Connecting to LLM...", title="Assistant", border_style="green")
        with optional_live(Group(panel, self.get_rich_status_bar()), 
                           console=console, enabled=not self.headless, refresh_per_second=12, transient=True) as live:
          self._active_live = live
          # Log request details in DEBUG mode
          self._log_llm_request(active_messages, self.get_tools())
          
          # Resolve model and provider
          actual_model, extra_body = self._resolve_model_and_provider(self.model)
          
          self._throttle_request()
          # Try calling with stream_options={"include_usage": True}
          try:
            kwargs = {
              "model": actual_model,
              "messages": active_messages,
              "tools": self.get_tools(),
              "stream": True,
              "stream_options": {"include_usage": True}
            }
            if extra_body:
              kwargs["extra_body"] = extra_body
            stream = self.client.chat.completions.create(**kwargs)
          except Exception as e:
            if self._is_retryable_exception(e):
              raise
            logger.debug(f"Failed to call API with stream_options: {e}. Retrying without stream_options.")
            kwargs = {
              "model": actual_model,
              "messages": active_messages,
              "tools": self.get_tools(),
              "stream": True
            }
            if extra_body:
              kwargs["extra_body"] = extra_body
            stream = self.client.chat.completions.create(**kwargs)
          
          first_metadata_chunk = True
          first_chunk = True
          finish_reason = None
          usage_metadata = None
          chunk_id = None
          resp_model = None
          sys_fp = None
          last_extra_fields = {
            "reasoning": None,
            "reasoning_content": None,
            "reasoning_details": None,
            "thought_signature": None
          }
          current_max_thinking = getattr(self.config, "max_thinking_chars", 12000)
          for chunk in stream:
            # Capture usage if present
            if hasattr(chunk, "usage") and chunk.usage:
              usage_metadata = chunk.usage
            elif hasattr(chunk, "model_extra") and chunk.model_extra and "usage" in chunk.model_extra:
              usage_metadata = chunk.model_extra["usage"]

            # Log metadata on first chunk
            if first_metadata_chunk:
              chunk_id = getattr(chunk, "id", None)
              resp_model = getattr(chunk, "model", None)
              sys_fp = getattr(chunk, "system_fingerprint", None)
              logger.debug(
                f"LLM response started. Chunk ID: {chunk_id}, Model: {resp_model}, System Fingerprint: {sys_fp}"
              )
              first_metadata_chunk = False

            if not chunk.choices:
              continue
            choice = chunk.choices[0]
            delta = choice.delta
            if hasattr(choice, "finish_reason") and choice.finish_reason:
              finish_reason = choice.finish_reason
            
            # Extract any OpenRouter extra fields for reasoning/thought
            extra_fields = ["reasoning", "reasoning_content", "reasoning_details", "thought_signature"]
            has_new_reasoning = False
            for field in extra_fields:
              val = getattr(delta, field, None)
              if val is None and hasattr(delta, "model_extra") and delta.model_extra:
                val = delta.model_extra.get(field)
              if val is None and isinstance(delta, dict):
                val = delta.get(field)
                
              if val is not None:
                has_new_reasoning = True
                
                # Determine if we should append or replace to prevent duplicate metadata / emoji bloat
                should_append = False
                if field in ("reasoning", "reasoning_content"):
                  if isinstance(val, str) and isinstance(extra_fields_accumulated[field], str):
                    is_duplicate = (val == last_extra_fields[field])
                    is_metadata_like = len(val) > 1 or any(ord(c) > 0x2000 for c in val)
                    if not (is_duplicate and is_metadata_like):
                      should_append = True
                
                if should_append:
                  extra_fields_accumulated[field] += val
                else:
                  extra_fields_accumulated[field] = val
                  
                last_extra_fields[field] = val
            
            # Process streaming content
            if delta.content:
              content_accumulated += delta.content

            # Process streaming reasoning/thought
            reasoning_accumulated = (extra_fields_accumulated.get("reasoning_content") or 
                                     extra_fields_accumulated.get("reasoning") or "")
            
            # Guard against endless internal thinking loops
            max_thinking_chars = getattr(self.config, "max_thinking_chars", 12000)
            leeway = getattr(self.config, "max_thinking_leeway_chars", 2000)
            
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
              elif not self.headless:
                with self._pause_live():
                  self._print("\n[bold yellow]⚠️  LLM exceeded internal thinking budget.[/bold yellow]")
                  self._print(f"Accumulated thinking: {len(reasoning_accumulated)} characters (limit: {current_max_thinking} chars).")
                  self._print("What would you like to do?")
                  self._print(
                    "  [bold red]\\[s][/bold red]top (abort thinking stream) / "
                    "  [bold green]\\[l][/bold green]et this go (increase budget by another 12000 chars) / "
                    "  [bold cyan]\\[w][/bold cyan]hitelist for the current turn: ",
                    end=""
                  )
                  try:
                    response = input().strip().lower()
                  except (KeyboardInterrupt, EOFError):
                    response = "s"
                  
                  if response in ("l", "let"):
                    current_max_thinking = len(reasoning_accumulated) + max_thinking_chars
                    if hasattr(self, "config") and self.config is not None:
                      self.config.max_thinking_chars = current_max_thinking
                    should_abort = False
                    logger.info(f"User allowed thinking to continue. New budget: {current_max_thinking} chars.")
                  elif response in ("w", "whitelist"):
                    whitelist_thinking_this_turn = True
                    should_abort = False
                    logger.info("User whitelisted thinking for the current turn.")
                  else:
                    should_abort = True
                    logger.info("User decided to stop thinking. Aborting stream.")

            if (should_abort 
                and not content_accumulated 
                and not delta.content 
                and not tool_calls_accumulated 
                and not delta.tool_calls):
              logger.warning(f"Aborting stream: LLM exceeded maximum internal thinking budget of {current_max_thinking} chars (accumulated: {len(reasoning_accumulated)}).")
              if hasattr(stream, "close"):
                try:
                  stream.close()
                except Exception:
                  pass
              raise ThinkingBudgetExceeded(
                "LLM exceeded maximum internal thinking budget.",
                reasoning_accumulated
              )
            
            if delta.content or has_new_reasoning:
              first_chunk = False
              renderables = []
              if reasoning_accumulated.strip():
                renderables.append(Panel(LazyMarkdown(reasoning_accumulated), title="Thinking", border_style="yellow"))
              if content_accumulated:
                renderables.append(Panel(LazyMarkdown(content_accumulated), title="Assistant", border_style="green"))
              
              if renderables:
                panel = Group(*renderables)
                live.update(Group(panel, self.get_rich_status_bar()))
                
            # Process streaming tool calls
            if delta.tool_calls:
              first_chunk = False
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
                      
                # Render loading indicator
                panel = Panel(f"Accumulating tool arguments... ({len(tool_calls_accumulated)} call(s))", 
                               title="System", border_style="yellow")
                live.update(Group(panel, self.get_rich_status_bar()))
          # Remove status bar before exiting Live context
          live.update(panel)
        
        # Reconstruct and print the final panels permanently to console
        final_panels = []
        if reasoning_accumulated.strip():
          final_panels.append(Panel(Markdown(reasoning_accumulated), title="Thinking", border_style="yellow"))
        if content_accumulated:
          final_panels.append(Panel(Markdown(content_accumulated), title="Assistant", border_style="green"))
        
        if final_panels:
          self._print(Group(*final_panels))
        
        if finish_reason == "length":
          logger.warning("LLM response was truncated due to output token limit (finish_reason='length').")
          self._print("\n[bold yellow]⚠️  Warning: The AI's response was truncated because it reached the maximum output token limit.[/bold yellow]\n")
        
        logger.info(f"LLM call succeeded. Content size: {len(content_accumulated)} chars, Tool calls count: {len(tool_calls_accumulated)}")
        self._log_llm_response_summary(
          content_accumulated=content_accumulated,
          tool_calls_accumulated=tool_calls_accumulated,
          extra_fields_accumulated=extra_fields_accumulated,
          finish_reason=finish_reason,
          usage=usage_metadata,
          response_model=resp_model,
          system_fingerprint=sys_fp,
          chunk_id=chunk_id
        )
        self.last_api_call_time = time.time()
      except ThinkingBudgetExceeded as e:
        self.last_api_call_time = time.time()
        logger.warning(f"Thinking budget exceeded on attempt {attempt}")
        last_aborted_thinking = e.accumulated_thinking
        if attempt < max_retries:
          self._print(f"[bold yellow]⚠️  LLM exceeded internal thinking budget. Retrying with self-correction nudge (attempt {attempt}/{max_retries})...[/bold yellow]")
          time.sleep(1)
          continue
        else:
          self._print("[bold red]Error: LLM repeatedly exceeded internal thinking budget without producing output.[/bold red]")
          break
      except Exception as e:
        self.last_api_call_time = time.time()
        logger.exception("Error calling LLM API")
        formatted_err = self._format_api_error(e)
        if attempt < max_retries and self._is_retryable_exception(e):
          backoff_time = 2 ** attempt
          self._print(f"[bold yellow]⚠️  Error calling API: {formatted_err}. Retrying in {backoff_time}s (attempt {attempt}/{max_retries})...[/bold yellow]")
          time.sleep(backoff_time)
          continue
        else:
          self._print(f"[bold red]Error calling API:[/bold red] {formatted_err}")
          break
      finally:
        self._active_live = None
          
      # If we didn't receive structured tool calls, try to extract them from text content
      if not tool_calls_accumulated and content_accumulated:
        parsed_calls = self.extract_tool_calls_from_text(content_accumulated)
        if parsed_calls:
          tool_calls_accumulated = parsed_calls
          content_accumulated = ""
          
      # Check if response was empty (no content, no tool calls)
      is_empty_response = (not tool_calls_accumulated) and (not content_accumulated or not content_accumulated.strip())
      
      if not is_empty_response:
        api_succeeded = True
        break
        
      if attempt < max_retries:
        logger.info(f"LLM returned an empty response on loop {self.current_loop} (attempt {attempt}/{max_retries}). Retrying in 2s...")
        self._print(f"[bold yellow]⚠️  LLM returned an empty response. Retrying in 2s (attempt {attempt}/{max_retries})...[/bold yellow]")
        time.sleep(2)
      else:
        logger.info(f"LLM returned multiple empty responses on loop {self.current_loop}. Breaking cycle.")
        self._print("[bold red]❌  LLM returned multiple empty responses. Breaking cycle.[/bold red]")
        
    if not api_succeeded:
      break
            
    # Ensure every accumulated tool call has a unique ID and valid JSON arguments
    for tc in tool_calls_accumulated:
      if not tc.get("id") or tc.get("id") == "call_text_parsed":
        tc["id"] = f"call_{uuid.uuid4().hex[:12]}"
      
      func_obj = tc.get("function")
      if isinstance(func_obj, dict):
        t_args_raw = func_obj.get("arguments")
        if not isinstance(t_args_raw, str):
          func_obj["arguments"] = json.dumps(t_args_raw) if t_args_raw is not None else "{}"
        else:
          try:
            json.loads(t_args_raw)
          except Exception:
            try:
              repaired = repair_json(t_args_raw)
              json.loads(repaired)
              func_obj["arguments"] = repaired
            except Exception:
              func_obj["arguments"] = "{}"
            
    # Construct assistant message record
    assistant_msg = {"role": "assistant"}
    if content_accumulated:
      assistant_msg["content"] = content_accumulated
    else:
      assistant_msg["content"] = None
        
    if tool_calls_accumulated:
      assistant_msg["tool_calls"] = []
      for tc in tool_calls_accumulated:
        assistant_msg["tool_calls"].append({
          "id": tc["id"],
          "type": "function",
          "function": {
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"]
          }
        })
            
    for field, val in extra_fields_accumulated.items():
      if val is not None:
        assistant_msg[field] = val
          
    self.messages.append(assistant_msg)
    
    # If the response was truncated due to output token limit, automatically continue
    if finish_reason == "length":
      logger.warning("LLM response was truncated (finish_reason='length'). Automatically continuing...")
      self._print("[bold yellow]🔄  AI response was truncated because it reached the maximum output token limit. Automatically continuing...[/bold yellow]")
      loop_count += 1
      continue
    
    # If no tools called, we're finished with this turn
    if not tool_calls_accumulated:
      break
        
    # Otherwise, execute requested tools sequentially
    for tc in tool_calls_accumulated:
      t_id = tc["id"]
      t_name = tc["function"]["name"]
      t_args_raw = tc["function"]["arguments"]
      
      try:
        args_parsed = json.loads(t_args_raw) if t_args_raw else {}
      except Exception:
        try:
          args_parsed = json.loads(repair_json(t_args_raw)) if t_args_raw else {}
        except Exception as e:
          args_parsed = {}
          t_result = f"Error: Arguments failed JSON parsing: {str(e)}"
      else:
        # Execute tool
        token = active_session_var.set(self)
        try:
          if t_name == "ask_question":
            logger.info(f"Executing tool {t_name} (id={t_id}) with arguments: {args_parsed}")
            t_result = execute_tool(t_name, args_parsed, self)
          else:
            exec_panel = Panel(
              f"Name: [cyan]{t_name}[/cyan]\nArguments: [yellow]{escape(json.dumps(args_parsed, indent=2))}[/yellow]",
              title="🔧 Executing Tool",
              border_style="yellow"
            )
            with optional_live(Group(exec_panel, self.get_rich_status_bar()), console=console, enabled=not self.headless, refresh_per_second=12) as live:
              self._active_live = live
              try:
                logger.info(f"Executing tool {t_name} (id={t_id}) with arguments: {args_parsed}")
                t_result = execute_tool(t_name, args_parsed, self)
              finally:
                self._active_live = None
              # Remove status bar before exiting Live context
              live.update(exec_panel)
        finally:
          active_session_var.reset(token)
          self.temp_allowed_ro_paths.clear()
          self.temp_allowed_rw_paths.clear()
            
      # Print result summary nicely
      self._print(Panel(
        Text(t_result),
        title="🔧 Tool Result",
        border_style="dim yellow"
      ))
      
      truncated = "TRUNCATED" in t_result or "truncated" in t_result.lower() or "WARNING" in t_result
      logger.info(f"Tool {t_name} (id={t_id}) completed. Result size: {len(t_result)} characters (truncated: {truncated})")
      logger.debug(f"Tool {t_name} (id={t_id}) result content: {t_result}")
      
      # Record result for context
      # We wrap tool output in a JSON object to prevent issues with gateways/models
      # trying to parse raw code/braces as invalid JSON.
      t_result_sanitized = sanitize_tool_output(t_result)
      wrapped_content = json.dumps({"output": t_result_sanitized})

      self.messages.append({
        "role": "tool",
        "tool_call_id": t_id,
        "name": t_name,
        "content": wrapped_content
      })
        
    loop_count += 1
    
  if loop_count >= max_tool_loops:
    self._print("[bold red]Reached maximum sequential tool loop executions. Breaking cycle.[/bold red]")
  self.current_loop = 0
