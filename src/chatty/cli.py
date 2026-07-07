#!/usr/bin/env python3
import argparse
import logging
import sys
from rich.console import Console

from chatty.logging_setup import setup_logging
from chatty.utils import get_ollama_models, load_system_prompt_from_file
from chatty.session import ChatbotSession

logger = logging.getLogger("chatty")
console = Console()


def main():
  parser = argparse.ArgumentParser(
    description="AI Chatbot CLI with advanced sandboxed text and file system interaction."
  )
  parser.add_argument(
    "--provider", "-p",
    choices=["ollama", "openrouter"],
    default="ollama",
    help="The LLM backend provider to use (default: ollama)."
  )
  parser.add_argument(
    "--model", "-m",
    action="append",
    help="Model identifier(s) to use. Can be specified multiple times or as comma-separated values. The first model becomes the active model."
  )
  parser.add_argument(
    "--oracle-model",
    help="Model identifier to use as the oracle. Default determines based on provider."
  )
  parser.add_argument(
    "--context-size", "-c",
    type=int,
    default=8192,
    help="Target context window length constraint in tokens (default: 8192)."
  )
  parser.add_argument(
    "--sandbox", "-s",
    default="./sandbox",
    help="Path to the sandboxed file system directory. Writes are strictly restricted here (default: ./sandbox)."
  )
  parser.add_argument(
    "--skills-path", "-k",
    action="append",
    default=[],
    help="Custom directories to search for skills. Can be specified multiple times."
  )
  parser.add_argument(
    "--whitelist", "-w",
    action="append",
    default=[],
    help="Add an out-of-sandbox path to the initial whitelist. Can end with :ro or :rw to set mode (defaults to ro). Can be specified multiple times."
  )
  parser.add_argument(
    "--static-skills",
    action="store_true",
    default=None,
    help="Load all available skills statically into the system prompt to maximize prompt caching (defaults to True for OpenRouter, False for Ollama)."
  )
  parser.add_argument(
    "--prompt-caching",
    action="store_true",
    default=False,
    help="Explicitly enable prompt caching for compatible models (adds cache_control tagging, default: False)."
  )
  parser.add_argument(
    "--max-loops", "-l",
    type=int,
    default=20,
    help="Maximum sequential tool execution loops allowed in a single turn (default: 20)."
  )
  parser.add_argument(
    "--config-prompt", "-f",
    help="Path to a YAML or text configuration file containing the custom system prompt."
  )
  parser.add_argument(
    "--prompt-mode", "-d",
    choices=["replace", "integrate"],
    default="replace",
    help="How to apply the custom system prompt file (replace default prompt, or integrate/append to it)."
  )
  parser.add_argument(
    "--api-key", "-a",
    help="OpenRouter API key. Overrides the OPENROUTER_API_KEY environment variable."
  )
  parser.add_argument(
    "--url", "-u",
    help="API Base URL override (defaults to Ollama local endpoint or OpenRouter base URL)."
  )
  parser.add_argument(
    "--max-read-chars",
    type=int,
    default=40000,
    help="Max characters to read from a file during full read tool execution (default: 40000)."
  )
  parser.add_argument(
    "--max-grep-results",
    type=int,
    default=100,
    help="Max results returned by regex search tool (default: 100)."
  )
  parser.add_argument(
    "--max-command-chars",
    type=int,
    default=16000,
    help="Max characters returned from standard output/error of a shell command (default: 16000)."
  )
  parser.add_argument(
    "--max-history-tool-chars",
    type=int,
    default=1000,
    help="Max characters to keep in historical tool outputs before compression (default: 1000)."
  )
  parser.add_argument(
    "--history-keep-messages",
    type=int,
    default=4,
    help="Number of recent messages to keep fully uncompressed (default: 4)."
  )
  parser.add_argument(
    "--max-url-chars",
    type=int,
    default=24000,
    help="Max characters returned from fetched URLs (default: 24000)."
  )
  parser.add_argument(
    "--max-dir-items",
    type=int,
    default=200,
    help="Max items listed by the directory list tool (default: 200)."
  )
  parser.add_argument(
    "--log-file",
    default="chatty.log",
    help="Path to the file where operations will be logged (default: chatty.log). Set to empty string to disable logging."
  )
  parser.add_argument(
    "--log-level",
    default="info",
    choices=["debug", "info", "warning", "error"],
    help="Logging level (default: info)."
  )
  parser.add_argument(
    "--headless",
    action="store_true",
    default=False,
    help="Run the chatbot in headless mode (no console printing or terminal interactive loop)."
  )
  
  args = parser.parse_args()
  
  # Initialize logging
  if args.log_file:
    setup_logging(args.log_file, args.log_level)
    logger.info("==========================================")
    logger.info(f"Logging initialized to '{args.log_file}' (level: {args.log_level}).")
  
  # Load system prompt from file if specified
  custom_system_prompt = None
  if args.config_prompt:
    try:
      custom_system_prompt = load_system_prompt_from_file(args.config_prompt)
      if not args.headless:
        console.print(f"[bold blue]Info:[/bold blue] Loaded custom system prompt from '{args.config_prompt}' (mode: {args.prompt_mode}).")
    except Exception as e:
      if not args.headless:
        console.print(f"[bold red]Error loading prompt configuration:[/bold red] {e}")
      sys.exit(1)
          
  # Resolve default models
  models = []
  if args.model:
    for m in args.model:
      for part in m.split(','):
        part = part.strip()
        if part:
          models.append(part)

  if not models:
    if args.provider == "ollama":
      # Attempt to auto-detect model from local Ollama tags
      ollama_url = args.url or "http://localhost:11434/v1"
      local_models = get_ollama_models(ollama_url)
      if local_models:
        # Pick the first matching model
        models = [local_models[0]]
        if not args.headless:
          console.print(f"[bold blue]Info:[/bold blue] Auto-detected local Ollama model: [bold green]{models[0]}[/bold green]")
      else:
        models = ["qwen2.5-coder:7b"]
        if not args.headless:
          console.print(f"[bold blue]Info:[/bold blue] No local Ollama models detected. Fallback default: [bold green]{models[0]}[/bold green]")
    else:
      models = ["google/gemini-2.5-flash"]
      if not args.headless:
        console.print(f"[bold blue]Info:[/bold blue] OpenRouter provider selected. Default model: [bold green]{models[0]}[/bold green]")
          
  model = models[0]
  
  # Initialize and execute chat session
  chat_session = ChatbotSession(
    provider=args.provider,
    model=model,
    models=models,
    oracle_model=args.oracle_model,
    context_size=args.context_size,
    sandbox=args.sandbox,
    api_key=args.api_key,
    url=args.url,
    max_loops=args.max_loops,
    system_prompt_override=custom_system_prompt,
    prompt_mode=args.prompt_mode,
    skills_paths=args.skills_path,
    max_read_chars=args.max_read_chars,
    max_grep_results=args.max_grep_results,
    max_command_chars=args.max_command_chars,
    max_history_tool_chars=args.max_history_tool_chars,
    history_keep_messages=args.history_keep_messages,
    max_url_chars=args.max_url_chars,
    max_dir_items=args.max_dir_items,
    static_skills=args.static_skills,
    prompt_caching=args.prompt_caching,
    headless=args.headless,
    whitelist=args.whitelist
  )
  
  if not args.headless:
    chat_session.start_loop()


if __name__ == "__main__":
  main()
