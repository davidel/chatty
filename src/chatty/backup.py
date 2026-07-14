import os
import time
import logging
from typing import List, Tuple

logger = logging.getLogger("chatty")


def get_backups_dir(sandbox_dir: str) -> str:
  return os.path.join(sandbox_dir, ".chatty", "backups")


def ensure_gitignore_ignores_chatty(sandbox_dir: str) -> None:
  """Ensures .chatty/ is present in .gitignore so backups are not committed to git."""
  gitignore_path = os.path.join(sandbox_dir, ".gitignore")
  try:
    if os.path.exists(gitignore_path):
      with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
      if ".chatty/" in content or ".chatty" in content:
        return
      with open(gitignore_path, 'a', encoding='utf-8') as f:
        if content and not content.endswith('\n'):
          f.write('\n')
        f.write('# Chatty backups\n.chatty/\n')
    else:
      with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write('# Chatty backups\n.chatty/\n')
  except Exception:
    pass


def backup_file(sandbox_dir: str, rel_path: str) -> None:
  """Creates a timestamped backup of the file under .chatty/backups/path/to/file/timestamp.bak."""
  from chatty.safety import get_safe_path
  
  try:
    abs_path = get_safe_path(sandbox_dir, rel_path)
    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
      return
      
    ensure_gitignore_ignores_chatty(sandbox_dir)
    file_backups_dir = os.path.join(get_backups_dir(sandbox_dir), rel_path)
    os.makedirs(file_backups_dir, exist_ok=True)
    
    with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
      content = f.read()
      
    timestamp = int(time.time() * 1000)
    backup_path = os.path.join(file_backups_dir, f"{timestamp}.bak")
    while os.path.exists(backup_path):
      timestamp += 1
      backup_path = os.path.join(file_backups_dir, f"{timestamp}.bak")
    
    with open(backup_path, 'w', encoding='utf-8') as f:
      f.write(content)
      
    logger.info(f"Created backup of '{rel_path}' at '{os.path.relpath(backup_path, sandbox_dir)}'")
    prune_backups(file_backups_dir, max_backups=10)
  except Exception as e:
    logger.warning(f"Failed to backup file '{rel_path}': {e}")


def prune_backups(file_backups_dir: str, max_backups: int = 10) -> None:
  """Keep only the latest max_backups in the directory."""
  try:
    if not os.path.isdir(file_backups_dir):
      return
    files = []
    for f in os.listdir(file_backups_dir):
      path = os.path.join(file_backups_dir, f)
      if os.path.isfile(path) and f.endswith(".bak"):
        files.append(path)
        
    def get_timestamp(filepath):
      name = os.path.basename(filepath)
      try:
        return int(name.split(".")[0])
      except ValueError:
        return os.path.getmtime(filepath)
        
    files.sort(key=get_timestamp)
    if len(files) > max_backups:
      to_delete = files[:-max_backups]
      for path in to_delete:
        os.remove(path)
  except Exception:
    pass


def list_backups(sandbox_dir: str, rel_path: str) -> List[Tuple[int, str]]:
  """Returns a list of tuples (timestamp, format_time_str) of available backups for the given file."""
  file_backups_dir = os.path.join(get_backups_dir(sandbox_dir), rel_path)
  if not os.path.isdir(file_backups_dir):
    return []
    
  backups = []
  for f in os.listdir(file_backups_dir):
    path = os.path.join(file_backups_dir, f)
    if os.path.isfile(path) and f.endswith(".bak"):
      name = f[:-4]
      try:
        ts = int(name)
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts / 1000.0))
        backups.append((ts, time_str))
      except ValueError:
        pass
        
  backups.sort(key=lambda x: x[0], reverse=True)
  return backups


def restore_backup(sandbox_dir: str, rel_path: str, timestamp: int = None) -> str:
  """Restores a backup for the given file. If timestamp is None, restores the latest backup."""
  from chatty.safety import get_safe_path
  
  file_backups_dir = os.path.join(get_backups_dir(sandbox_dir), rel_path)
  if not os.path.isdir(file_backups_dir):
    return f"Error: No backups found for file '{rel_path}'."
    
  backups = list_backups(sandbox_dir, rel_path)
  if not backups:
    return f"Error: No backups found for file '{rel_path}'."
    
  target_ts = timestamp
  if target_ts is None:
    target_ts = backups[0][0]
    
  backup_file_path = os.path.join(file_backups_dir, f"{target_ts}.bak")
  if not os.path.exists(backup_file_path):
    return f"Error: Backup file with timestamp '{target_ts}' not found for '{rel_path}'."
    
  abs_path = get_safe_path(sandbox_dir, rel_path, write=True)
  os.makedirs(os.path.dirname(abs_path), exist_ok=True)
  
  with open(backup_file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    
  with open(abs_path, 'w', encoding='utf-8') as f:
    f.write(content)
    
  time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(target_ts / 1000.0))
  return f"Successfully restored '{rel_path}' to backup version from {time_str}."
