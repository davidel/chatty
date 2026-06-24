import os
from typing import List


def get_safe_path(sandbox_dir: str, target_path: str) -> str:
  """
  Resolves the absolute path of target_path and ensures it lies strictly inside sandbox_dir.
  Raises PermissionError if a directory traversal attempt is made.
  """
  abs_sandbox = os.path.realpath(sandbox_dir)
  
  # If path is absolute, check it. Otherwise, combine it with sandbox.
  if os.path.isabs(target_path):
    resolved_path = os.path.realpath(target_path)
  else:
    resolved_path = os.path.realpath(os.path.join(abs_sandbox, target_path))
    
  # Check if target is inside the sandbox folder by finding their common prefix
  common_prefix = os.path.commonpath([abs_sandbox, resolved_path])
  if common_prefix != abs_sandbox:
    raise PermissionError(
      f"Access Denied: Path '{target_path}' resolves to '{resolved_path}' "
      f"which is outside the sandbox directory '{abs_sandbox}'."
    )
  return resolved_path


def load_ignore_patterns(sandbox_dir: str) -> List[str]:
  """Loads default ignore patterns and appends patterns from .gitignore if present."""
  patterns = [
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".DS_Store",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".db",
    ".sqlite",
    ".bin"
  ]
  gitignore_path = os.path.join(sandbox_dir, ".gitignore")
  if os.path.exists(gitignore_path):
    try:
      with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            if line.endswith('/'):
              line = line[:-1]
            patterns.append(line)
    except Exception:
      pass
  return list(set(patterns))


def is_path_ignored(rel_path: str, patterns: List[str]) -> bool:
  """Checks if a relative path matches any of the ignore patterns."""
  import fnmatch
  segments = rel_path.split(os.sep)
  for pattern in patterns:
    for segment in segments:
      if fnmatch.fnmatch(segment, pattern):
        return True
    if fnmatch.fnmatch(rel_path, pattern):
      return True
  return False


def is_text_file(file_path: str) -> bool:
  """Check if a file is a text file by scanning for null bytes."""
  try:
    with open(file_path, "rb") as f:
      chunk = f.read(8192)
      return b"\0" not in chunk
  except Exception:
    return False


def count_lines(file_path: str) -> int:
  """Count the number of lines in a text file."""
  count = 0
  try:
    with open(file_path, "rb") as f:
      last_char = None
      for chunk in iter(lambda: f.read(65536), b""):
        if chunk:
          count += chunk.count(b"\n")
          last_char = chunk[-1]
      if last_char is not None and last_char != 10:
        count += 1
  except Exception:
    pass
  return count
