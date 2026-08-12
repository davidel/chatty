import datetime
import json
import logging
import os
import re
import urllib.parse
from typing import List, Dict, Any, Tuple, Optional
from html.parser import HTMLParser
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
      try:
        import yaml
        parsed = yaml.safe_load(yaml_content)
        if isinstance(parsed, dict):
          metadata = parsed
        else:
          metadata = {}
      except Exception:
        # Fallback to naive parser
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


def truncate_thinking_by_line(text: str, max_lines: Optional[int] = None) -> str:
  """Truncates the thinking text by lines to fit within the terminal window size."""
  if max_lines is None:
    # 2 border lines, 1 warning line, 1 status bar line + 2 extra safety lines
    max_lines = max(3, console.height - 6)

  lines = text.split("\n")
  if len(lines) <= max_lines:
    return text

  warning = "... [thinking output truncated for terminal performance] ..."
  # Keep the last max_lines - 1 lines
  kept_lines = lines[-(max_lines - 1):]
  return warning + "\n" + "\n".join(kept_lines)


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


def _try_ocr_fallback(content_bytes: bytes) -> str:
  """Attempts to extract text using OCR from scanned PDFs, if dependencies are present."""
  import shutil
  if not shutil.which("tesseract") or not shutil.which("pdftoppm"):
    return "[Info: PDF contains no extractable text. Scanned PDF detected, but tesseract or pdftoppm is missing. Skipping OCR.]"

  try:
    import pdf2image
    import pytesseract
  except ImportError:
    return "[Info: PDF contains no extractable text. Scanned PDF detected, but pdf2image or pytesseract is not installed. Skipping OCR.]"

  try:
    images = pdf2image.convert_from_bytes(content_bytes, last_page=5)
    ocr_pages = []
    for idx, img in enumerate(images):
      text = pytesseract.image_to_string(img)
      if text.strip():
        ocr_pages.append(f"--- Page {idx + 1} (OCR) ---\n{text.strip()}")
    if ocr_pages:
      return "\n\n".join(ocr_pages)
    return "[Info: PDF contains no extractable text. OCR was unable to parse any text content.]"
  except Exception as e:
    return f"[Error running OCR fallback: {str(e)}]"


class HTMLTextExtractor(HTMLParser):
  """A standard library HTML parser that extracts text while ignoring formatting/script blocks."""

  def __init__(self):
    super().__init__()
    self.text_parts = []
    self.ignore_depth = 0
    self.ignore_tags = {"script", "style", "head", "header", "footer", "nav"}

  def handle_starttag(self, tag, attrs):
    if tag in self.ignore_tags:
      self.ignore_depth += 1
    elif tag == "br":
      self.text_parts.append("\n")
    elif tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
      self.text_parts.append("\n")

  def handle_endtag(self, tag):
    if tag in self.ignore_tags:
      self.ignore_depth = max(0, self.ignore_depth - 1)
    elif tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
      self.text_parts.append("\n")

  def handle_data(self, data):
    if self.ignore_depth == 0:
      self.text_parts.append(data)

  def get_text(self):
    import html as html_parser
    return html_parser.unescape("".join(self.text_parts))


def get_random_user_agent_headers() -> Dict[str, str]:
  """Returns a dictionary of browser headers with a randomized modern User-Agent and standard companion headers."""
  import random
  user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0"
  ]
  return {
    "User-Agent": random.choice(user_agents),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
  }


def tool_fetch_url(url: str, max_chars: int = 24000, sandbox_path: Optional[str] = None) -> str:
  """Fetch the text content of a public URL and convert it to clean text (removes HTML tags or parses PDF)."""
  try:
    headers = get_random_user_agent_headers()
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    content_type = ""
    if hasattr(response, "headers") and hasattr(response.headers, "get"):
      try:
        content_type_val = response.headers.get("Content-Type", "")
        if isinstance(content_type_val, str):
          content_type = content_type_val.lower()
      except Exception:
        pass
    is_pdf = "application/pdf" in content_type or url.lower().split('?')[0].endswith(".pdf")

    if is_pdf:
      try:
        import pypdf
      except ImportError:
        return "Error: URL points to a PDF, but 'pypdf' package is not installed. Please run 'pip install pypdf'."

      import io
      try:
        reader = pypdf.PdfReader(io.BytesIO(response.content))
        if reader.is_encrypted:
          return "[Error: The requested PDF is password-protected or encrypted.]"

        pages_text = []
        for idx, page in enumerate(reader.pages):
          page_text = page.extract_text()
          if page_text and page_text.strip():
            pages_text.append(f"--- Page {idx + 1} ---\n{page_text.strip()}")

        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
          full_text = _try_ocr_fallback(response.content)

      except Exception as pdf_err:
        return f"Error parsing PDF: {str(pdf_err)}"
    else:
      parser = HTMLTextExtractor()
      parser.feed(response.text)
      text = parser.get_text()

      cleaned_lines = []
      for line in text.split('\n'):
        stripped = line.strip()
        if stripped:
          cleaned_lines.append(stripped)
        elif cleaned_lines and cleaned_lines[-1] != "":
          cleaned_lines.append("")
      full_text = "\n".join(cleaned_lines).strip()

    if len(full_text) > max_chars:
      warning_msg = f"\n\n[WARNING: URL content truncated. Total length: {len(full_text)} characters.]"
      if sandbox_path:
        try:
          import hashlib
          url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]
          parsed_url = urllib.parse.urlparse(url)
          domain_part = re.sub(r'[^a-zA-Z0-9_]', '_', parsed_url.netloc)
          path_part = re.sub(r'[^a-zA-Z0-9_]', '_', parsed_url.path)[:20].strip('_')
          if path_part:
            filename = f"{domain_part}_{path_part}_{url_hash}.txt"
          else:
            filename = f"{domain_part}_{url_hash}.txt"
          cache_dir = os.path.join(sandbox_path, ".url_cache")
          os.makedirs(cache_dir, exist_ok=True)
          cache_file_path = os.path.join(cache_dir, filename)
          with open(cache_file_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
          relative_cache_path = os.path.join(".url_cache", filename)
          warning_msg = (
            f"\n\n[WARNING: URL content truncated. Total length: {len(full_text)} characters. "
            f"Full content cached in sandbox at: {relative_cache_path}. "
            f"You can search within this content using search_grep or read parts of it using read_file.]"
          )
        except Exception as cache_err:
          logger = logging.getLogger("chatty")
          if logger:
            logger.warning(f"Failed to cache URL content: {cache_err}")
      return full_text[:max_chars] + warning_msg
    return full_text
  except Exception as e:
    return f"Error fetching URL: {str(e)}"


def sanitize_tool_output(text: str) -> str:
  """Sanitizes tool output to prevent issues with JSON serialization and API gateways.

  Removes null bytes and replaces other control characters (except newline, carriage
  return, and tab) with their escaped or readable representation.
  """
  if not isinstance(text, str):
    return text
  # Remove null bytes entirely as they are blocked by many WAFs and JSON parsers
  text = text.replace("\x00", "\\x00")
  # Replace other control characters (0x00-0x1F, except \n, \r, \t)
  # to avoid JSON escaping issues in some gateways
  def replace_control(match):
    char = match.group(0)
    return f"\\u{ord(char):04x}"

  # Match characters in range 0x00-0x1F except 0x09 (\t), 0x0A (\n), 0x0D (\r)
  control_chars_re = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
  return control_chars_re.sub(replace_control, text)


def repair_json(json_str: str) -> str:
  """Attempts to repair common JSON malformations from LLMs."""
  if not json_str:
    return "{}"

  try:
    import json_repair
    repaired = json_repair.repair(json_str)
    if repaired:
      return repaired
  except ImportError:
    pass

  s = json_str.strip()
  import re
  s = re.sub(r"'([^']+)'\s*:", r'"\1":', s)
  s = re.sub(r":\s*'([^']*)'", r': "\1"', s)
  s = re.sub(r',\s*([\]}])', r'\1', s)

  braces = []
  in_string = False
  escaped = False
  for i, char in enumerate(s):
    if in_string:
      if escaped:
        escaped = False
      elif char == '\\':
        escaped = True
      elif char == '"':
        in_string = False
    else:
      if char == '"':
        in_string = True
      elif char in ('{', '['):
        braces.append(char)
      elif char == '}':
        if braces and braces[-1] == '{':
          braces.pop()
      elif char == ']':
        if braces and braces[-1] == '[':
          braces.pop()

  if in_string:
    s += '"'

  while braces:
    matching = braces.pop()
    if matching == '{':
      s += '}'
    elif matching == '[':
      s += ']'

  return s


def format_short_number(val: float) -> str:
  """Formats a number using metric/SI postfixes (K, M, G, T)."""
  if val < 0:
    return "-" + format_short_number(-val)
  if val < 1000:
    if isinstance(val, float) and val.is_integer():
      return str(int(val))
    return str(val)
  elif val < 1_000_000:
    num = val / 1000
    res = f"{num:.2f}"
    if "." in res:
      res = res.rstrip("0").rstrip(".")
    return f"{res}K"
  elif val < 1_000_000_000:
    num = val / 1_000_000
    res = f"{num:.2f}"
    if "." in res:
      res = res.rstrip("0").rstrip(".")
    return f"{res}M"
  elif val < 1_000_000_000_000:
    num = val / 1_000_000_000
    res = f"{num:.2f}"
    if "." in res:
      res = res.rstrip("0").rstrip(".")
    return f"{res}G"
  else:
    num = val / 1_000_000_000_000
    res = f"{num:.2f}"
    if "." in res:
      res = res.rstrip("0").rstrip(".")
    return f"{res}T"


def copy_to_clipboard(text: str) -> bool:
  """Tries to copy text to the system clipboard.

  First attempts to use `pyperclip` if it is installed, falling back
  to common command line clipboard utilities.
  """
  try:
    import pyperclip
    pyperclip.copy(text)
    return True
  except (ImportError, Exception):
    pass

  import subprocess
  import shutil

  # Try wl-copy (Wayland)
  if shutil.which("wl-copy"):
    try:
      subprocess.run(["wl-copy"], input=text, text=True, check=True)
      return True
    except Exception:
      pass

  # Try xclip (X11)
  if shutil.which("xclip"):
    try:
      subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, check=True)
      return True
    except Exception:
      pass

  # Try xsel (X11)
  if shutil.which("xsel"):
    try:
      subprocess.run(["xsel", "--clipboard", "--input"], input=text, text=True, check=True)
      return True
    except Exception:
      pass

  # Try pbcopy (macOS)
  if shutil.which("pbcopy"):
    try:
      subprocess.run(["pbcopy"], input=text, text=True, check=True)
      return True
    except Exception:
      pass

  # Try clip.exe (WSL/Windows)
  if shutil.which("clip.exe"):
    try:
      subprocess.run(["clip.exe"], input=text, text=True, check=True)
      return True
    except Exception:
      pass

  return False

