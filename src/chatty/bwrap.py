import logging
import os
import shutil
import subprocess
from typing import List, Optional

logger = logging.getLogger("chatty")


def is_bwrap_available(bwrap_bin: Optional[str] = None) -> bool:
  """Checks if bubblewrap (bwrap) executable is present and able to create user namespaces."""
  bin_path = bwrap_bin or shutil.which("bwrap")
  if not bin_path or not os.path.exists(bin_path):
    return False

  try:
    res = subprocess.run(
      [bin_path, "--ro-bind", "/", "/", "--dev", "/dev", "true"],
      capture_output=True,
      timeout=2,
    )
    return res.returncode == 0
  except Exception as e:
    logger.debug(f"bwrap availability check failed: {e}")
    return False


def wrap_command_with_bwrap(
  bwrap_bin: Optional[str],
  sandbox_dir: str,
  command: str,
  ro_paths: Optional[List[str]] = None,
  rw_paths: Optional[List[str]] = None,
  cwd: Optional[str] = None,
) -> List[str]:
  """Wraps a shell command in a sandboxed bubblewrap invocation."""
  bin_path = bwrap_bin or shutil.which("bwrap") or "bwrap"
  abs_sandbox = os.path.realpath(sandbox_dir)
  working_dir = os.path.realpath(cwd) if cwd else abs_sandbox

  args = [
    bin_path,
    "--die-with-parent",
    "--unshare-user",
    "--unshare-ipc",
    "--unshare-pid",
    "--unshare-uts",
    "--unshare-cgroup",
    "--dev-bind", "/dev", "/dev",
    "--proc", "/proc",
  ]

  if os.path.exists("/sys"):
    args.extend(["--ro-bind-try", "/sys", "/sys"])

  args.extend(["--ro-bind", "/", "/"])
  args.extend(["--tmpfs", "/tmp"])

  if os.path.exists("/run/resolvconf/resolv.conf"):
    args.extend(["--ro-bind-try", "/run/resolvconf/resolv.conf", "/run/resolvconf/resolv.conf"])
  if os.path.exists("/run/systemd/resolve"):
    args.extend(["--ro-bind-try", "/run/systemd/resolve", "/run/systemd/resolve"])

  args.extend(["--bind", abs_sandbox, abs_sandbox])

  if ro_paths:
    for p in dict.fromkeys(os.path.realpath(path) for path in ro_paths if os.path.exists(path)):
      args.extend(["--ro-bind-try", p, p])

  if rw_paths:
    for p in dict.fromkeys(os.path.realpath(path) for path in rw_paths if os.path.exists(path)):
      args.extend(["--bind", p, p])

  args.extend(["--chdir", working_dir])
  args.extend(["--setenv", "SBOXED", "1"])
  args.extend(["--", "/bin/sh", "-c", command])
  return args
