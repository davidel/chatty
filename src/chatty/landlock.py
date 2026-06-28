import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional


logger = logging.getLogger("chatty")


def compile_landlock_binary() -> Optional[str]:
  if sys.platform != "linux":
    return None

  cc_path = shutil.which("gcc") or shutil.which("clang")
  if not cc_path:
    logger.warning("Neither gcc nor clang found. Landlock wrapper cannot be compiled.")
    return None

  package_dir = os.path.dirname(os.path.abspath(__file__))
  source_path = os.path.join(package_dir, "landlock_exec.c")
  if not os.path.exists(source_path):
    logger.warning(f"Landlock source file not found at {source_path}")
    return None

  target_path = os.path.join(package_dir, "landlock_exec")
  is_writable = False
  try:
    test_path = target_path + ".test"
    with open(test_path, "w") as f:
      f.write("")
    os.unlink(test_path)
    is_writable = True
  except OSError:
    pass

  if not is_writable:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "chatty")
    try:
      os.makedirs(cache_dir, exist_ok=True)
      target_path = os.path.join(cache_dir, "landlock_exec")
    except OSError:
      target_path = os.path.join(tempfile.gettempdir(), "chatty_landlock_exec")

  need_compile = True
  if os.path.exists(target_path):
    try:
      src_mtime = os.path.getmtime(source_path)
      bin_mtime = os.path.getmtime(target_path)
      if bin_mtime >= src_mtime:
        need_compile = False
    except OSError:
      pass

  if need_compile:
    logger.info(f"Compiling Landlock wrapper {source_path} to {target_path}")
    try:
      cmd = [cc_path, "-O2", "-Wall", source_path, "-o", target_path]
      subprocess.run(cmd, capture_output=True, text=True, check=True)
      logger.info("Landlock wrapper compiled successfully.")
    except (subprocess.CalledProcessError, OSError) as e:
      stderr = getattr(e, "stderr", "")
      logger.warning(f"Failed to compile landlock wrapper: {e}. Output: {stderr}")
      return None

  return target_path


def wrap_command_with_landlock(binary_path: str, sandbox_dir: str, command: str) -> List[str]:
  rw_paths = [sandbox_dir, tempfile.gettempdir()]
  # Remove duplicate paths and resolve them
  rw_paths = list(dict.fromkeys(os.path.realpath(p) for p in rw_paths))

  args = [binary_path, "--ro", "/"]
  for path in rw_paths:
    args.extend(["--rw", path])

  args.extend(["--", "/bin/sh", "-c", command])
  return args
