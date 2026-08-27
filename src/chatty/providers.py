import os
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.parse

logger = logging.getLogger("chatty")


class LiteLLMClientStub:
  def __init__(self, base_url, api_key, default_headers=None, timeout=None, max_retries=None):
    self.base_url = base_url
    self.api_key = api_key
    self.default_headers = default_headers or {}
    self.timeout = timeout
    self.max_retries = max_retries


PROVIDER_KEYS = {
  "openrouter": "OPENROUTER_API_KEY",
  "ollama": None,
  "openai": "OPENAI_API_KEY",
  "anthropic": "ANTHROPIC_API_KEY",
  "gemini": "GEMINI_API_KEY",
  "groq": "GROQ_API_KEY",
}


def resolve_api_key(provider_name: str, custom_key: Optional[str] = None) -> str:
  if custom_key:
    return custom_key
  env_var = PROVIDER_KEYS.get(provider_name)
  if env_var:
    return os.environ.get(env_var) or ""
  return os.environ.get("CUSTOM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""


def get_cache_filepath():
  env_path = os.environ.get("CHATTY_CACHE_PATH")
  if env_path:
    cache_dir = os.path.abspath(env_path)
  else:
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, ".cache", "chatty")
  os.makedirs(cache_dir, exist_ok=True)
  return os.path.join(cache_dir, "openrouter_models_cache.json")


def load_cached_models() -> Optional[List[Dict[str, Any]]]:
  cache_path = get_cache_filepath()
  if os.path.exists(cache_path):
    try:
      with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if time.time() - data.get("timestamp", 0) < 86400:
          return data.get("models")
    except Exception:
      pass
  return None


def save_cached_models(models: List[Dict[str, Any]]):
  cache_path = get_cache_filepath()
  try:
    with open(cache_path, "w", encoding="utf-8") as f:
      json.dump({"timestamp": time.time(), "models": models}, f)
  except Exception:
    pass


class BaseProvider(ABC):
  @property
  @abstractmethod
  def name(self) -> str:
    pass

  @abstractmethod
  def get_default_url(self) -> Optional[str]:
    pass

  @abstractmethod
  def get_default_model(self, api_key: Optional[str] = None) -> Optional[str]:
    pass

  @abstractmethod
  def init_client(self, url: Optional[str], api_key: Optional[str]) -> LiteLLMClientStub:
    pass

  @abstractmethod
  def fetch_models(self, url: Optional[str], api_key: Optional[str], force_refresh: bool = False) -> List[Dict[str, Any]]:
    pass

  def get_litellm_model_prefix(self, model: str) -> str:
    return f"{self.name}/{model}"

  def clean_assistant_message(self, message: Dict[str, Any]) -> None:
    message.pop("reasoning", None)
    message.pop("reasoning_content", None)
    message.pop("reasoning_details", None)
    message.pop("thought_signature", None)


class OllamaProvider(BaseProvider):
  @property
  def name(self) -> str:
    return "ollama"

  def get_default_url(self) -> str:
    return "http://localhost:11434/v1"

  def get_default_model(self, api_key: Optional[str] = None) -> str:
    return "qwen2.5-coder:7b"

  def init_client(self, url: Optional[str], api_key: Optional[str]) -> LiteLLMClientStub:
    base = url or self.get_default_url()
    return LiteLLMClientStub(
      base_url=base,
      api_key="ollama"
    )

  def fetch_models(self, url: Optional[str], api_key: Optional[str], force_refresh: bool = False) -> List[Dict[str, Any]]:
    base_url = url or self.get_default_url()
    if base_url.endswith("/v1"):
      api_url = base_url[:-3] + "/api/tags"
    else:
      api_url = base_url.rstrip("/") + "/api/tags"
    try:
      req = urllib.request.Request(api_url)
      with urllib.request.urlopen(req, timeout=5) as response:
        data = json.loads(response.read().decode())
        return [
          {
            "id": m["name"],
            "name": m["name"].split(":")[0],
            "size": m.get("size", 0),
            "details": m.get("details", {})
          }
          for m in data.get("models", [])
        ]
    except Exception as e:
      logger.error(f"Error fetching Ollama models: {e}")
      return []


class OpenRouterProvider(BaseProvider):
  @property
  def name(self) -> str:
    return "openrouter"

  def get_default_url(self) -> str:
    return "https://openrouter.ai/api/v1"

  def get_default_model(self, api_key: Optional[str] = None) -> str:
    try:
      cached = load_cached_models()
      if cached:
        for m in cached:
          if m.get("pricing_input") == 0.0 and m.get("pricing_output") == 0.0:
            return m["id"]
      url = "https://openrouter.ai/api/v1/models"
      headers = {}
      key = resolve_api_key("openrouter", api_key)
      if key and key != "missing_api_key":
        headers["Authorization"] = f"Bearer {key}"
      req = urllib.request.Request(url, headers=headers)
      with urllib.request.urlopen(req, timeout=5) as response:
        data = json.loads(response.read().decode())
        for m in data.get("data", []):
          pricing = m.get("pricing", {})
          if float(pricing.get("prompt", 0)) == 0.0 and float(pricing.get("completion", 0)) == 0.0:
            return m["id"]
    except Exception:
      pass
    return "google/gemini-2.5-flash:free"

  def init_client(self, url: Optional[str], api_key: Optional[str]) -> LiteLLMClientStub:
    base = url or self.get_default_url()
    key = resolve_api_key("openrouter", api_key)
    if not key:
      from rich.console import Console
      Console().print(
        "[bold red]Warning:[/bold red] OpenRouter API key is not configured. "
        "Use [cyan]/api_key <key>[/cyan] or set the [cyan]OPENROUTER_API_KEY[/cyan] environment variable."
      )
      key = "missing_api_key"
    return LiteLLMClientStub(
      base_url=base,
      api_key=key,
      default_headers={
        "HTTP-Referer": "https://github.com/davidel/chatty",
        "X-Title": "Chatty"
      }
    )

  def fetch_models(self, url: Optional[str], api_key: Optional[str], force_refresh: bool = False) -> List[Dict[str, Any]]:
    if not force_refresh:
      cached = load_cached_models()
      if cached is not None:
        return cached
    api_url = "https://openrouter.ai/api/v1/models?sort=most-popular"
    headers = {}
    key = resolve_api_key("openrouter", api_key)
    if key and key != "missing_api_key":
      headers["Authorization"] = f"Bearer {key}"
    try:
      req = urllib.request.Request(api_url, headers=headers)
      with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode())
        models = [
          {
            "id": m["id"],
            "name": m["name"],
            "context": m.get("context_length"),
            "pricing_input": float(m.get("pricing", {}).get("prompt", 0)) * 1e6,
            "pricing_output": float(m.get("pricing", {}).get("completion", 0)) * 1e6,
            "description": m.get("description"),
            "architecture": m.get("architecture"),
            "created": m.get("created"),
            "knowledge_cutoff": m.get("knowledge_cutoff"),
            "hugging_face_id": m.get("hugging_face_id"),
          }
          for m in data.get("data", [])
        ]
        save_cached_models(models)
        return models
    except Exception as e:
      logger.error(f"Error fetching OpenRouter models: {e}")
      return []

  def clean_assistant_message(self, message: Dict[str, Any]) -> None:
    message.pop("reasoning", None)
    message.pop("reasoning_content", None)
    message.pop("reasoning_details", None)


class GenericProvider(BaseProvider):
  def __init__(self, name: str):
    self._name = name

  @property
  def name(self) -> str:
    return self._name

  def get_default_url(self) -> Optional[str]:
    return None

  def get_default_model(self, api_key: Optional[str] = None) -> Optional[str]:
    return None

  def init_client(self, url: Optional[str], api_key: Optional[str]) -> LiteLLMClientStub:
    base = url
    if not base:
      if self.name in PROVIDER_KEYS:
        base = None
      else:
        from rich.console import Console
        Console().print(f"[bold red]Error:[/bold red] Provider '{self.name}' requires an API URL. Use [cyan]/url <url>[/cyan] or configure --url.")
        base = "http://localhost:8000/v1"
    key = resolve_api_key(self.name, api_key)
    return LiteLLMClientStub(
      base_url=base,
      api_key=key
    )

  def fetch_models(self, url: Optional[str], api_key: Optional[str], force_refresh: bool = False) -> List[Dict[str, Any]]:
    return []

  def get_litellm_model_prefix(self, model: str) -> str:
    is_custom_openai = "." in self.name or "localhost" in self.name or "/" in self.name or ":" in self.name
    if is_custom_openai:
      return f"openai/{model}"
    return f"{self.name}/{model}"


PROVIDER_REGISTRY = {
  "openrouter": OpenRouterProvider(),
  "ollama": OllamaProvider(),
}


def get_provider(provider_name: str) -> BaseProvider:
  if provider_name in PROVIDER_REGISTRY:
    return PROVIDER_REGISTRY[provider_name]
  return GenericProvider(provider_name)
