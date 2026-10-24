import json
import logging
import os
from typing import Dict, Optional
from chatty.backup import ensure_gitignore_ignores_chatty

logger = logging.getLogger("chatty")

MODELS_FILE_NAME = "models.json"


def find_workspace_root() -> str:
  curr = os.path.abspath(os.getcwd())
  visited = set()
  while True:
    visited.add(curr)
    if os.path.exists(os.path.join(curr, ".git")):
      return curr
    parent = os.path.dirname(curr)
    if parent == curr or parent in visited:
      break
    curr = parent
  return os.path.abspath(os.getcwd())


def get_project_chatty_dir(sandbox: Optional[str] = None, create: bool = False) -> Optional[str]:
  """Locates or creates the .chatty/ directory for the current project."""
  if sandbox:
    target_dir = os.path.join(os.path.abspath(sandbox), ".chatty")
    if os.path.isdir(target_dir):
      return target_dir
    if create:
      try:
        os.makedirs(target_dir, exist_ok=True)
        ensure_gitignore_ignores_chatty(os.path.dirname(target_dir))
        return target_dir
      except Exception as e:
        logger.debug(f"Failed to create .chatty directory at {target_dir}: {e}")
        return None
    return None

  candidates = []
  ws_root = find_workspace_root()
  if ws_root:
    candidates.append(os.path.join(os.path.abspath(ws_root), ".chatty"))

  cwd = os.path.abspath(os.getcwd())
  candidates.append(os.path.join(cwd, ".chatty"))

  for cand in candidates:
    if os.path.isdir(cand):
      return cand

  if not create:
    return None

  target_dir = os.path.join(os.path.abspath(ws_root), ".chatty") if ws_root else os.path.join(cwd, ".chatty")
  try:
    os.makedirs(target_dir, exist_ok=True)
    ensure_gitignore_ignores_chatty(os.path.dirname(target_dir))
    return target_dir
  except Exception as e:
    logger.debug(f"Failed to create .chatty directory at {target_dir}: {e}")
    return None


def get_saved_models(sandbox: Optional[str] = None) -> Dict[str, str]:
  """Loads remembered models for each task from the project's .chatty/models.json."""
  candidates = []
  if sandbox:
    candidates.append(os.path.join(os.path.abspath(sandbox), ".chatty", MODELS_FILE_NAME))
  ws_root = find_workspace_root()
  if ws_root:
    candidates.append(os.path.join(os.path.abspath(ws_root), ".chatty", MODELS_FILE_NAME))
  candidates.append(os.path.join(os.path.abspath(os.getcwd()), ".chatty", MODELS_FILE_NAME))

  models_file = None
  for cand in candidates:
    if os.path.isfile(cand):
      models_file = cand
      break

  if not models_file:
    return {}

  try:
    with open(models_file, "r", encoding="utf-8") as f:
      data = json.load(f)
    if not isinstance(data, dict):
      return {}

    normalized = {}
    for k, v in data.items():
      if not isinstance(v, str) or not v.strip():
        continue
      k_norm = k.strip().lower()
      if k_norm in ("main", "model", "chat"):
        normalized["main"] = v.strip()
      elif k_norm in ("oracle", "oracle_model"):
        normalized["oracle"] = v.strip()
      elif k_norm in ("discovery", "discovery_model", "scout"):
        normalized["discovery"] = v.strip()
      else:
        normalized[k_norm] = v.strip()
    return normalized
  except Exception as e:
    logger.debug(f"Error reading {models_file}: {e}")
    return {}


def save_model_for_task(task: str, model_name: str, sandbox: Optional[str] = None) -> None:
  """Saves a model selection for a specific task into .chatty/models.json."""
  if not task or not model_name:
    return

  chatty_dir = get_project_chatty_dir(sandbox=sandbox, create=True)
  if not chatty_dir:
    return

  models_file = os.path.join(chatty_dir, MODELS_FILE_NAME)
  data = {}
  if os.path.isfile(models_file):
    try:
      with open(models_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        if isinstance(loaded, dict):
          data = loaded
    except Exception as e:
      logger.debug(f"Error reading existing {models_file}: {e}")

  task_norm = task.strip().lower()
  if task_norm in ("model", "chat"):
    task_norm = "main"
  elif task_norm == "oracle_model":
    task_norm = "oracle"
  elif task_norm in ("discovery_model", "scout"):
    task_norm = "discovery"

  data[task_norm] = model_name.strip()

  try:
    with open(models_file, "w", encoding="utf-8") as f:
      json.dump(data, f, indent=2)
  except Exception as e:
    logger.error(f"Failed to save models selection to {models_file}: {e}")


def save_models_for_tasks(task_models: Dict[str, Optional[str]], sandbox: Optional[str] = None) -> None:
  """Saves multiple model selections for tasks into .chatty/models.json."""
  for task, model in task_models.items():
    if model:
      save_model_for_task(task, model, sandbox=sandbox)
