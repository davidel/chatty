import datetime
import json
import logging
import os
import re
import urllib.parse
from typing import List, Dict, Any, Tuple, Optional
import requests
import tiktoken
from rich.console import Console

logger = logging.getLogger("chatty")
console = Console()


def preprocess_shell_string(cmd_str: str) -> str:
  operators = {"|", "&", ";", "<", ">", "(", ")"}
  res = []
  in_single = False
  in_double = False
  escaped = False
  for char in cmd_str:
    if escaped:
      res.append(char)
      escaped = False
      continue
    if char == "\\":
      res.append(char)
      escaped = True
      continue
    if char == "'" and not in_double:
      in_single = not in_single
      res.append(char)
      continue
    if char == '"' and not in_single:
      in_double = not in_double
      res.append(char)
      continue
    if not in_single and not in_double and char in operators:
      res.append(" " + char + " ")
    else:
      res.append(char)
  return "".join(res)


def parse_shell_commands(cmd_str: str) -> list:
  import shlex
  preprocessed = preprocess_shell_string(cmd_str)
  try:
    tokens = shlex.split(preprocessed, posix=True)
  except Exception:
    tokens = preprocessed.strip().split()
      
  binaries = []
  
  control_operators = {"|", "&&", "||", ";", "&", "\n", "(", ")"}
  redirections = {">", "<", ">>", "<<", ">&", "<&"}
  
  state = "START"
  
  iterator = iter(tokens)
  while True:
    try:
      token = next(iterator)
    except StopIteration:
      break
        
    if token in control_operators:
      state = "START"
      continue
        
    if state == "START":
      is_redirect = False
      for red in redirections:
        if token == red or token.endswith(red):
          is_redirect = True
          break
      
      if is_redirect:
        try:
          next(iterator)
        except StopIteration:
          pass
        continue
          
      if "=" in token and not token.startswith("="):
        continue
          
      state = "ARG"
      binaries.append(os.path.basename(token))
    else:
      is_redirect = False
      for red in redirections:
        if token == red or token.endswith(red):
          is_redirect = True
          break
      if is_redirect:
        try:
          next(iterator)
        except StopIteration:
          pass
        continue
          
  return binaries


def record_command_binaries(args, session=None):
  if not args:
    return
  if session is None:
    try:
      from chatty.session import ChatbotSession
      session = getattr(ChatbotSession, "_active_session", None)
    except ImportError:
      pass
  if not session:
    return
  
  binaries = []
  if isinstance(args, list):
    if args:
      first = args[0]
      if isinstance(first, (str, bytes)):
        name = first.decode('utf-8', errors='ignore') if isinstance(first, bytes) else first
        binaries.append(os.path.basename(name))
      else:
        binaries.append(str(first))
  elif isinstance(args, (str, bytes)):
    cmd_str = args.decode('utf-8', errors='ignore') if isinstance(args, bytes) else args
    binaries = parse_shell_commands(cmd_str)
      
  for binary_name in binaries:
    session.external_binaries_count += 1
    session.external_binaries_breakdown[binary_name] = session.external_binaries_breakdown.get(binary_name, 0) + 1


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
  """
  Parses YAML frontmatter from markdown.
  Returns (metadata_dict, body_content).
  """
  metadata = {}
  body = content
  
  if content.startswith("---"):
    parts = content.split("---", 2)
    if len(parts) >= 3:
      yaml_content = parts[1]
      body = parts[2].strip()
      
      for line in yaml_content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
          continue
        if ":" in line:
          key, val = line.split(":", 1)
          key = key.strip()
          val = val.strip().strip('"').strip("'")
          
          if val.startswith('[') and val.endswith(']'):
            import ast
            try:
              val = ast.literal_eval(val)
            except Exception:
              pass
          metadata[key] = val
  return metadata, body


def load_system_prompt_from_file(file_path: str) -> str:
  """Loads custom system prompt from a YAML configuration or raw text file."""
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Configuration file '{file_path}' does not exist.")
  with open(file_path, 'r', encoding='utf-8') as f:
    try:
      import yaml
      data = yaml.safe_load(f)
      if isinstance(data, dict):
        if "system_prompt" in data:
          return str(data["system_prompt"])
        else:
          raise KeyError(f"YAML configuration in '{file_path}' is missing the 'system_prompt' key.")
      return str(data)
    except Exception as e:
      # If YAML parsing fails or not a YAML, read as plain text
      f.seek(0)
      return f.read().strip()


def count_tokens(text: str) -> int:
  """Estimates token length using tiktoken's cl100k_base encoder."""
  try:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text, disallowed_special=()))
  except Exception:
    # fallback estimation if tiktoken fails
    return len(text) // 4


def truncate_output(text: str, max_chars: int = 16000) -> str:
  """Truncates the middle of a string if it exceeds max_chars, leaving head and tail blocks."""
  if len(text) <= max_chars:
    return text
  half = max_chars // 2
  truncated_chars = len(text) - max_chars
  return (
    f"{text[:half]}\n\n"
    f"... [TRUNCATED {truncated_chars} CHARACTERS OF OUTPUT] ...\n\n"
    f"{text[-half:]}"
  )


def get_ollama_models(url: str) -> List[str]:
  """Queries local Ollama tags API endpoint to retrieve downloaded models list."""
  parsed = urllib.parse.urlparse(url)
  base_api_url = f"{parsed.scheme}://{parsed.netloc}/api/tags"
  try:
    response = requests.get(base_api_url, timeout=2)
    if response.status_code == 200:
      models_data = response.json()
      return [m["name"] for m in models_data.get("models", [])]
  except Exception:
    pass
  return []


def print_diff(path: str, old_content: str, new_content: str):
  """Renders a beautiful color-coded diff of file changes to the console."""
  import difflib
  from rich.text import Text
  from rich.panel import Panel
  
  old_lines = old_content.splitlines(keepends=True)
  new_lines = new_content.splitlines(keepends=True)
  
  diff = list(difflib.unified_diff(
    old_lines,
    new_lines,
    fromfile=f"old/{path}",
    tofile=f"new/{path}",
    n=3
  ))
  
  if not diff:
    return
    
  text = Text()
  for line in diff:
    if line.startswith('+') and not line.startswith('+++'):
      text.append(line, style="green")
    elif line.startswith('-') and not line.startswith('---'):
      text.append(line, style="red")
    elif line.startswith('@@'):
      text.append(line, style="cyan")
    elif line.startswith('---') or line.startswith('+++'):
      text.append(line, style="bold white")
    else:
      text.append(line, style="dim white")
      
  console.print(Panel(
    text,
    title=f"📝 File Changes: {os.path.basename(path)}",
    border_style="magenta"
  ))


def tool_fetch_url(url: str, max_chars: int = 24000) -> str:
  """Fetch the text content of a public URL and convert it to clean text (removes HTML tags)."""
  try:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    html = response.text
    
    html_clean = re.sub(r'<(script|style|head|header|footer|nav).*?>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html_clean = re.sub(r'<br\s*/?>', '\n', html_clean, flags=re.IGNORECASE)
    html_clean = re.sub(r'</?(p|div|li|h[1-6]).*?>', '\n', html_clean, flags=re.IGNORECASE)
    text = re.sub(r'<.*?>', '', html_clean, flags=re.DOTALL)
    import html as html_parser
    text = html_parser.unescape(text)
    
    cleaned_lines = []
    for line in text.split('\n'):
      stripped = line.strip()
      if stripped:
        cleaned_lines.append(stripped)
      elif cleaned_lines and cleaned_lines[-1] != "":
        cleaned_lines.append("")
    full_text = "\n".join(cleaned_lines).strip()
    if len(full_text) > max_chars:
      return full_text[:max_chars] + f"\n\n[WARNING: URL content truncated. Total length: {len(full_text)} characters.]"
    return full_text
  except Exception as e:
    return f"Error fetching URL: {str(e)}"
