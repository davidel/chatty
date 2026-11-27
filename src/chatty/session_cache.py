import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from chatty.backup import ensure_gitignore_ignores_chatty

logger = logging.getLogger("chatty")

DEFAULT_MAX_CACHED_SESSIONS = 8


def get_session_cache_dir(sandbox_dir: str) -> str:
  """Returns the .chatty/cache/ directory path in the sandbox, ensuring it exists."""
  cache_dir = os.path.join(sandbox_dir, ".chatty", "cache")
  os.makedirs(cache_dir, exist_ok=True)
  return cache_dir


def format_size(size_bytes: int) -> str:
  """Formats byte count into human-readable string."""
  val = float(size_bytes)
  for unit in ["B", "KB", "MB", "GB"]:
    if val < 1024.0:
      if unit == "B":
        return f"{int(val)} B"
      return f"{val:.1f} {unit}"
    val /= 1024.0
  return f"{val:.1f} TB"


def list_cached_sessions(sandbox_dir: str) -> List[Dict[str, Any]]:
  """Lists cached sessions in .chatty/cache/, sorted by timestamp descending."""
  cache_dir = os.path.join(sandbox_dir, ".chatty", "cache")
  if not os.path.isdir(cache_dir):
    return []

  ignored_names = {"repo_map.json", "symbol_cache.json", "models.json", "mcp.json"}
  sessions: List[Dict[str, Any]] = []

  for filename in os.listdir(cache_dir):
    if filename in ignored_names or not filename.endswith(".json"):
      continue

    file_path = os.path.join(cache_dir, filename)
    if not os.path.isfile(file_path):
      continue

    try:
      with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception:
      continue

    if not isinstance(data, dict):
      continue

    # Identify if this is a session file
    if "messages" not in data and "provider" not in data and not filename.startswith("session_"):
      continue

    session_id = data.get("session_id")
    if not session_id:
      stem = os.path.splitext(filename)[0]
      if stem.startswith("session_"):
        stem = stem[len("session_"):]
      session_id = stem

    raw_ts = data.get("timestamp")
    if raw_ts is None:
      try:
        ts = os.path.getmtime(file_path)
      except Exception:
        ts = 0.0
    else:
      try:
        ts = float(raw_ts)
      except (ValueError, TypeError):
        try:
          ts = os.path.getmtime(file_path)
        except Exception:
          ts = 0.0

    try:
      formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
    except Exception:
      formatted_time = "Unknown"

    try:
      size = os.path.getsize(file_path)
    except Exception:
      size = 0

    sessions.append({
      "id": session_id,
      "timestamp": ts,
      "formatted_time": formatted_time,
      "size": size,
      "formatted_size": format_size(size),
      "path": file_path,
      "filename": filename,
    })

  sessions.sort(key=lambda s: s["timestamp"], reverse=True)
  return sessions


def prune_cached_sessions(sandbox_dir: str, max_sessions: int = DEFAULT_MAX_CACHED_SESSIONS) -> None:
  """Keeps only the most recent max_sessions in .chatty/cache/."""
  try:
    sessions = list_cached_sessions(sandbox_dir)
    if len(sessions) > max_sessions:
      to_remove = sessions[max_sessions:]
      for s in to_remove:
        try:
          if os.path.exists(s["path"]):
            os.remove(s["path"])
        except Exception as e:
          logger.warning(f"Failed to remove old session {s['path']}: {e}")
  except Exception as e:
    logger.warning(f"Error pruning cached sessions: {e}")


def save_cached_session(session: Any, max_sessions: int = DEFAULT_MAX_CACHED_SESSIONS) -> str:
  """Saves current session state to .chatty/cache/session_<id>.json and prunes old sessions."""
  cache_dir = get_session_cache_dir(session.sandbox)
  ensure_gitignore_ignores_chatty(session.sandbox)

  session_id = getattr(session, "session_id", None)
  if not session_id:
    session_id = str(uuid.uuid4())[:8]
    session.session_id = session_id

  save_path = os.path.join(cache_dir, f"session_{session_id}.json")
  session.save_session(save_path)
  prune_cached_sessions(session.sandbox, max_sessions=max_sessions)
  return save_path


def find_cached_session(sandbox_dir: str, target: str) -> Optional[str]:
  """Resolves a target string (ID, filename, or path) to a cached session file path if available."""
  if not target:
    return None

  clean_target = target.strip()

  # Check if target is directly a file on disk
  if os.path.isabs(clean_target) and os.path.isfile(clean_target):
    return clean_target

  direct_sandbox = os.path.join(sandbox_dir, clean_target)
  if os.path.isfile(direct_sandbox):
    return direct_sandbox

  if os.path.isfile(clean_target):
    return os.path.abspath(clean_target)

  cache_dir = os.path.join(sandbox_dir, ".chatty", "cache")
  if not os.path.isdir(cache_dir):
    return None

  # Direct cache filename candidates
  candidates = [
    os.path.join(cache_dir, clean_target),
    os.path.join(cache_dir, f"{clean_target}.json"),
    os.path.join(cache_dir, f"session_{clean_target}.json"),
  ]
  for c in candidates:
    if os.path.isfile(c):
      return c

  # Search list of cached sessions by ID, filename stem, or filename
  sessions = list_cached_sessions(sandbox_dir)
  lower_target = clean_target.lower()
  for s in sessions:
    s_id = str(s.get("id", "")).lower()
    fname = s.get("filename", "").lower()
    stem = os.path.splitext(fname)[0]
    if s_id == lower_target:
      return s["path"]
    if f"session_{s_id}" == lower_target:
      return s["path"]
    if fname == lower_target or stem == lower_target:
      return s["path"]

  return None
