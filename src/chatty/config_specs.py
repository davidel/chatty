import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ConfigSpec:
  name: str
  type: type
  default: Any
  description: str
  flags: Tuple[str, ...]
  action: Optional[Any] = None
  choices: Optional[Tuple[str, ...]] = None

  def parse_value(self, val_str: str) -> Any:
    if self.type is bool:
      lower = val_str.strip().lower()
      if lower in ("true", "yes", "1", "on"):
        return True
      elif lower in ("false", "no", "0", "off"):
        return False
      else:
        raise ValueError("Must be a boolean value (true/false, yes/no, 1/0, on/off)")
    if self.choices is not None and val_str not in self.choices:
      raise ValueError(f"Must be one of: {', '.join(self.choices)}")
    return self.type(val_str)

  def register_arg(self, parser_or_group: Any) -> None:
    kwargs: Dict[str, Any] = {
      "default": self.default,
      "help": self.description,
    }
    if self.action is not None:
      kwargs["action"] = self.action
    else:
      kwargs["type"] = self.type
    if self.choices is not None:
      kwargs["choices"] = list(self.choices)
    parser_or_group.add_argument(*self.flags, **kwargs)


CONFIG_SPECS: Dict[str, ConfigSpec] = {
  "api_delay": ConfigSpec(
    name="api_delay",
    type=float,
    default=2.5,
    description="Minimum delay in seconds between consecutive API requests to avoid rate limits (default: 2.5).",
    flags=("--api-delay",),
  ),
  "api_timeout": ConfigSpec(
    name="api_timeout",
    type=float,
    default=60.0,
    description="Timeout in seconds for API requests and streams (default: 60.0).",
    flags=("--api-timeout",),
  ),
  "auto_approve": ConfigSpec(
    name="auto_approve",
    type=bool,
    default=False,
    action="store_true",
    description="Automatically approve file changes without interactive prompt.",
    flags=("--auto-approve",),
  ),
  "god_mode": ConfigSpec(
    name="god_mode",
    type=bool,
    default=False,
    action="store_true",
    description="Automatically allow Python script execution with filesystem operations if running inside a safe sandbox (docker, bwrap, landlock).",
    flags=("--god-mode",),
  ),
  "context_size": ConfigSpec(
    name="context_size",
    type=int,
    default=8192,
    description="Target context window length constraint in tokens (default: 8192).",
    flags=("--context-size", "-c"),
  ),
  "discovery_loops": ConfigSpec(
    name="discovery_loops",
    type=int,
    default=50,
    description="Maximum tool loops allowed for the discovery agent (default: 50).",
    flags=("--discovery-loops",),
  ),
  "discovery_model": ConfigSpec(
    name="discovery_model",
    type=str,
    default=None,
    description="Model identifier to use for the discovery/scout agent (defaults to active model).",
    flags=("--discovery-model",),
  ),
  "discovery_timeout": ConfigSpec(
    name="discovery_timeout",
    type=float,
    default=None,
    description="Timeout in seconds for discovery agent API requests and streams (default: inherits --api-timeout).",
    flags=("--discovery-timeout",),
  ),
  "history_keep_messages": ConfigSpec(
    name="history_keep_messages",
    type=int,
    default=4,
    description="Number of recent messages to keep fully uncompressed during context compaction (default: 4).",
    flags=("--history-keep-messages",),
  ),
  "max_command_chars": ConfigSpec(
    name="max_command_chars",
    type=int,
    default=16000,
    description="Max characters returned from standard output/error of a shell command (default: 16000).",
    flags=("--max-command-chars",),
  ),
  "max_dir_items": ConfigSpec(
    name="max_dir_items",
    type=int,
    default=200,
    description="Max items listed by the directory list tool (default: 200).",
    flags=("--max-dir-items",),
  ),
  "max_grep_results": ConfigSpec(
    name="max_grep_results",
    type=int,
    default=100,
    description="Max results returned by regex search tool (default: 100).",
    flags=("--max-grep-results",),
  ),
  "max_loops": ConfigSpec(
    name="max_loops",
    type=int,
    default=20,
    description="Maximum sequential tool execution loops allowed in a single turn (default: 20).",
    flags=("--max-loops", "-l"),
  ),
  "max_read_chars": ConfigSpec(
    name="max_read_chars",
    type=int,
    default=40000,
    description="Max characters to read from a file during full read tool execution (default: 40000).",
    flags=("--max-read-chars",),
  ),
  "max_thinking_chars": ConfigSpec(
    name="max_thinking_chars",
    type=int,
    default=12000,
    description="Maximum internal thinking characters before prompting the user (default: 12000).",
    flags=("--max-thinking-chars",),
  ),
  "max_thinking_leeway_chars": ConfigSpec(
    name="max_thinking_leeway_chars",
    type=int,
    default=2000,
    description="Leeway in characters beyond the maximum before hard-aborting or prompting (default: 2000).",
    flags=("--max-thinking-leeway-chars",),
  ),
  "max_url_chars": ConfigSpec(
    name="max_url_chars",
    type=int,
    default=24000,
    description="Max characters returned from fetched URLs (default: 24000).",
    flags=("--max-url-chars",),
  ),
  "oracle_timeout": ConfigSpec(
    name="oracle_timeout",
    type=float,
    default=180.0,
    description="Timeout in seconds for oracle API requests and streams (default: 180.0).",
    flags=("--oracle-timeout",),
  ),
  "profile": ConfigSpec(
    name="profile",
    type=str,
    default="coder",
    choices=("coder", "general"),
    description="Agent role profile ('coder' for software engineering, 'general' for general assistant; default: 'coder').",
    flags=("--profile",),
  ),
  "prompt_caching": ConfigSpec(
    name="prompt_caching",
    type=bool,
    default=False,
    action="store_true",
    description="Explicitly enable prompt caching for compatible models (adds cache_control tagging, default: False).",
    flags=("--prompt-caching",),
  ),
  "repo_map": ConfigSpec(
    name="repo_map",
    type=bool,
    default=None,
    action=argparse.BooleanOptionalAction,
    description="Include a Tree-Sitter repository map in the system prompt (default: True for coder, False for general).",
    flags=("--repo-map",),
  ),
  "repo_map_tokens": ConfigSpec(
    name="repo_map_tokens",
    type=int,
    default=1024,
    description="Target token budget for repository map (default: 1024).",
    flags=("--repo-map-tokens",),
  ),
}


def register_config_arguments(parser: argparse.ArgumentParser) -> None:
  group = parser.add_argument_group("Runtime Configuration Options (tunable via /config)")
  for spec in CONFIG_SPECS.values():
    spec.register_arg(group)
