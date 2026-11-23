#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from rich.console import Console

from chatty.logging_setup import setup_logging
from chatty.utils import get_ollama_models, load_system_prompt_from_file
from chatty.session import ChatbotSession
from chatty.llm import get_default_openrouter_model
from chatty.config_specs import register_config_arguments

logger = logging.getLogger("chatty")
console = Console()


def main():
  parser = argparse.ArgumentParser(
    description="AI Chatbot CLI with advanced sandboxed text and file system interaction."
  )
  parser.add_argument(
    "--provider", "-p",
    help="The LLM backend provider to use. Defaults to the domain of the --url if --url is provided, or 'ollama'."
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
    "--roundtable-models", "-rm",
    help="Model identifier(s) for the round table expert panel (comma-separated)."
  )
  parser.add_argument(
    "--roundtable-moderator",
    help="Model identifier to use as the round table moderator (defaults to active model)."
  )
  parser.add_argument(
    "--sandbox", "-s",
    help="Path to the sandboxed file system directory. Writes are strictly restricted here (default: ./sandbox)."
  )
  parser.add_argument(
    "--skills-path", "-k",
    action="append",
    default=[],
    help="Custom directories to search for skills. Can be specified multiple times."
  )
  parser.add_argument(
    "--ondemand-skills-path", "-o",
    action="append",
    default=[],
    help="Custom directories to search for on-demand skills. Can be specified multiple times."
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
    "--max-history-tool-chars",
    type=int,
    default=1000,
    help="Max characters to keep in historical tool outputs before compression (default: 1000)."
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
  parser.add_argument(
    "--mcp-config",
    help="Path to an MCP (Model Context Protocol) JSON configuration file."
  )
  parser.add_argument(
    "--mcp",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable or disable Model Context Protocol (MCP) server integration (default: True)."
  )
  parser.add_argument(
    "--roundtable-timeout",
    type=float,
    default=120.0,
    help="Timeout in seconds for round table expert API requests and streams (default: 120.0)."
  )
  parser.add_argument(
    "--roundtable-max-tool-calls",
    type=int,
    default=10,
    help="Default maximum tool loops per round table expert turn (default: 10)."
  )
  register_config_arguments(parser)
  parser.add_argument(
    "--backend", "-B",
    choices=["auto", "docker", "bwrap", "landlock", "none"],
    default="auto",
    help="Execution backend for commands: 'auto' (default: docker if configured, else bwrap/landlock if on Linux, else local), 'docker' (companion container), 'bwrap' (Bubblewrap sandbox), 'landlock' (Linux Landlock LSM), or 'none' (direct execution)."
  )
  parser.add_argument(
    "--docker-image",
    help="Docker image tag to use for the companion container (default: 'chatty-sandbox:<user>' or 'chatty-sandbox:latest')."
  )
  parser.add_argument(
    "--docker-base-image",
    help="Base image to use when building the companion container (default: 'debian:stable-slim')."
  )
  parser.add_argument(
    "--dockerfile",
    help="Path to a custom Dockerfile to build and use for the companion container."
  )
  parser.add_argument(
    "--docker-build",
    action="store_true",
    default=False,
    help="Force building or rebuilding the Docker companion image on startup."
  )
  parser.add_argument(
    "--docker-host",
    help="Custom Docker host endpoint (e.g. 'tcp://127.0.0.1:2375' or 'unix:///run/user/1000/docker.sock'). Overrides DOCKER_HOST."
  )
  parser.add_argument(
    "--docker-reset",
    action="store_true",
    default=False,
    help="Reset (recreate) the Docker companion container for the current workspace."
  )
  parser.add_argument(
    "--docker-ephemeral",
    action="store_true",
    default=False,
    help="Run the companion container as ephemeral (destroy on session exit)."
  )
  parser.add_argument(
    "--docker-auth-proxy",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable credential proxy for Docker companion container to safely inject tokens (HF_TOKEN, AWS, GCS, R2, etc.) without revealing them to the LLM."
  )
  parser.add_argument(
    "--docker-auth-proxy-rw-urls",
    "--rw-urls",
    dest="docker_auth_proxy_rw_urls",
    default=os.environ.get("CHATTY_RW_URLS"),
    help="Semicolon-separated list of cloud storage URLs/buckets permitted for write/delete operations (e.g. 's3://scratch;r2://artifacts/*'). Defaults to CHATTY_RW_URLS env var."
  )
  parser.add_argument(
    "--bwrap-overlay",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable OverlayFS on /usr for the Bubblewrap sandbox (default: True)."
  )
  parser.add_argument(
    "--bwrap-reset",
    action="store_true",
    default=False,
    help="Reset (delete and recreate) the Bubblewrap sandbox writable state (<sandbox>/.chatty/sbox/)."
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
          
  sandbox_specified = args.sandbox is not None
  sandbox_dir = args.sandbox if sandbox_specified else "./sandbox"

  from chatty.models_config import get_saved_models, save_models_for_tasks
  saved_models = get_saved_models(sandbox=sandbox_dir if sandbox_specified else None)
  saved_main = saved_models.get("main")

  from urllib.parse import urlparse

  # Determine provider
  provider = args.provider
  if not provider:
    if args.url:
      parsed = urlparse(args.url)
      domain = parsed.netloc or parsed.path
      if ":" in domain:
        domain = domain.split(":")[0]
      if not domain:
        parser.error("Could not parse domain name from --url.")
      provider = domain
    else:
      provider = "ollama"

  # Validate custom provider constraints
  from chatty.providers import PROVIDER_KEYS, get_provider
  is_standard_provider = provider in PROVIDER_KEYS
  if not is_standard_provider:
    if not args.url:
      parser.error(f"Provider '{provider}' requires an API URL. Pass it via --url / -u.")
    if not args.model and not saved_main:
      parser.error(f"Provider '{provider}' requires a model. Pass it via --model / -m.")

  # Resolve default models
  models = []
  if args.model:
    for m in args.model:
      for part in m.split(','):
        part = part.strip()
        if part:
          models.append(part)
  elif saved_main:
    models = [saved_main]
    if not args.headless:
      console.print(f"[bold blue]Info:[/bold blue] Loaded saved main model from .chatty: [bold green]{models[0]}[/bold green]")

  if not models:
    prov_inst = get_provider(provider)
    if provider == "ollama":
      # Attempt to auto-detect model from local Ollama tags
      ollama_url = args.url or prov_inst.get_default_url()
      local_models = get_ollama_models(ollama_url)
      if local_models:
        models = [local_models[0]]
        if not args.headless:
          console.print(f"[bold blue]Info:[/bold blue] Auto-detected local Ollama model: [bold green]{models[0]}[/bold green]")
      else:
        models = [prov_inst.get_default_model(profile=args.profile)]
        if not args.headless:
          console.print(f"[bold blue]Info:[/bold blue] No local Ollama models detected. Fallback default: [bold green]{models[0]}[/bold green]")
    else:
      default_m = prov_inst.get_default_model(args.api_key, profile=args.profile)
      if default_m:
        models = [default_m]
        if not args.headless:
          console.print(f"[bold blue]Info:[/bold blue] {provider.capitalize()} provider selected. Default model: [bold green]{models[0]}[/bold green]")
      else:
        parser.error(f"Provider '{provider}' requires a model. Pass it via --model / -m.")
          
  model = models[0]

  oracle_model = args.oracle_model
  if not oracle_model and "oracle" in saved_models:
    oracle_model = saved_models["oracle"]
    if not args.headless:
      console.print(f"[bold blue]Info:[/bold blue] Loaded saved oracle model from .chatty: [bold green]{oracle_model}[/bold green]")

  discovery_model = args.discovery_model
  if not discovery_model and "discovery" in saved_models:
    discovery_model = saved_models["discovery"]
    if not args.headless:
      console.print(f"[bold blue]Info:[/bold blue] Loaded saved discovery model from .chatty: [bold green]{discovery_model}[/bold green]")

  roundtable_models = None
  if args.roundtable_models:
    roundtable_models = [m.strip() for m in args.roundtable_models.split(",") if m.strip()]
  elif "roundtable_models" in saved_models:
    roundtable_models = saved_models["roundtable_models"]
    if not args.headless:
      console.print(f"[bold blue]Info:[/bold blue] Loaded saved round table models from .chatty: [bold green]{', '.join(roundtable_models)}[/bold green]")

  roundtable_moderator = args.roundtable_moderator
  if not roundtable_moderator and "roundtable_moderator" in saved_models:
    roundtable_moderator = saved_models["roundtable_moderator"]
    if not args.headless:
      console.print(f"[bold blue]Info:[/bold blue] Loaded saved round table moderator from .chatty: [bold green]{roundtable_moderator}[/bold green]")

  save_models_for_tasks({
    "main": model,
    "oracle": oracle_model,
    "discovery": discovery_model,
    "roundtable_models": roundtable_models,
    "roundtable_moderator": roundtable_moderator
  }, sandbox=sandbox_dir if sandbox_specified else None)
  
  # Initialize and execute chat session
  with ChatbotSession(
    provider=provider,
    model=model,
    models=models,
    oracle_model=oracle_model,
    context_size=args.context_size,
    sandbox=sandbox_dir,
    sandbox_specified=sandbox_specified,
    api_key=args.api_key,
    url=args.url,
    max_loops=args.max_loops,
    system_prompt_override=custom_system_prompt,
    prompt_mode=args.prompt_mode,
    skills_paths=args.skills_path,
    ondemand_skills_paths=args.ondemand_skills_path,
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
    auto_approve=args.auto_approve,
    god_mode=args.god_mode,
    whitelist=args.whitelist,
    max_thinking_chars=args.max_thinking_chars,
    max_thinking_leeway_chars=args.max_thinking_leeway_chars,
    api_delay=args.api_delay,
    api_timeout=args.api_timeout,
    oracle_timeout=args.oracle_timeout,
    discovery_timeout=args.discovery_timeout,
    profile=args.profile,
    repo_map=args.repo_map,
    repo_map_tokens=args.repo_map_tokens,
    discovery_model=discovery_model,
    discovery_loops=args.discovery_loops,
    roundtable_models=roundtable_models,
    roundtable_moderator=roundtable_moderator,
    roundtable_timeout=args.roundtable_timeout,
    roundtable_max_tool_calls=args.roundtable_max_tool_calls,
    backend=args.backend,
    docker_image=args.docker_image,
    docker_base_image=args.docker_base_image,
    dockerfile=args.dockerfile,
    docker_build=args.docker_build,
    docker_host=args.docker_host,
    docker_reset=args.docker_reset,
    docker_persistent=not args.docker_ephemeral,
    docker_auth_proxy=args.docker_auth_proxy,
    docker_auth_proxy_rw_urls=args.docker_auth_proxy_rw_urls,
    bwrap_overlay=args.bwrap_overlay,
    bwrap_reset=args.bwrap_reset,
    mcp_config=args.mcp_config,
    enable_mcp=args.mcp
  ) as chat_session:
    if not args.headless:
      chat_session.start_loop()


if __name__ == "__main__":
  main()
