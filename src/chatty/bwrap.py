import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("chatty")

_OVERLAY_SUPPORT_CACHE: Dict[str, bool] = {}


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


def get_bwrap_sbox_dir(sandbox_dir: str) -> str:
  """Returns the path to the single containment directory for bubblewrap writable mounts."""
  return os.path.join(os.path.realpath(sandbox_dir), ".chatty", "sbox")


def get_cached_micromamba_path() -> str:
  """Returns the host cache location for the bootstrapped micromamba executable."""
  cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "chatty", "bin")
  os.makedirs(cache_dir, exist_ok=True)
  return os.path.join(cache_dir, "micromamba")


def bootstrap_micromamba(cache_path: Optional[str] = None) -> Optional[str]:
  """Downloads the standalone static micromamba binary to the host cache if not present."""
  import io
  import platform
  import tarfile
  import urllib.request

  target_path = cache_path or get_cached_micromamba_path()
  if os.path.exists(target_path) and os.access(target_path, os.X_OK):
    return target_path

  machine = platform.machine().lower()
  arch_map = {
    "x86_64": "linux-64",
    "amd64": "linux-64",
    "aarch64": "linux-aarch64",
    "arm64": "linux-aarch64",
    "ppc64le": "linux-ppc64le",
  }
  arch = arch_map.get(machine, "linux-64")
  url = f"https://micro.mamba.pm/api/micromamba/{arch}/latest"

  try:
    logger.info(f"Downloading static micromamba binary from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "chatty"})
    with urllib.request.urlopen(req, timeout=30) as resp:
      data = resp.read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tar:
      member = tar.getmember("bin/micromamba")
      f = tar.extractfile(member)
      if f is None:
        return None
      temp_path = target_path + ".tmp"
      with open(temp_path, "wb") as out:
        out.write(f.read())
      os.chmod(temp_path, 0o755)
      os.replace(temp_path, target_path)

    logger.info(f"Successfully cached micromamba at {target_path}")
    return target_path
  except Exception as e:
    logger.warning(f"Failed to bootstrap micromamba: {e}")
    return None


def ensure_micromamba(sandbox_dir: Optional[str] = None) -> Optional[str]:
  """Ensures micromamba is available. If missing on host, auto-provisions into sandbox home."""
  host_bin = shutil.which("micromamba")
  if host_bin:
    return host_bin

  cached_bin = bootstrap_micromamba()
  if not cached_bin or not os.path.exists(cached_bin):
    return None

  if sandbox_dir:
    sbox_local_bin = os.path.join(get_bwrap_sbox_dir(sandbox_dir), "home", ".local", "bin")
    os.makedirs(sbox_local_bin, exist_ok=True)
    target_bin = os.path.join(sbox_local_bin, "micromamba")
    if not os.path.exists(target_bin):
      try:
        shutil.copy2(cached_bin, target_bin)
        os.chmod(target_bin, 0o755)
      except Exception as e:
        logger.debug(f"Could not copy micromamba into sandbox: {e}")
    return target_bin

  return cached_bin


def init_bwrap_sbox_dirs(sandbox_dir: str) -> Tuple[str, str, str]:
  """Initializes and returns (home_dir, upper_dir, work_dir) within <sandbox>/.chatty/sbox/."""
  sbox_dir = get_bwrap_sbox_dir(sandbox_dir)
  home_dir = os.path.join(sbox_dir, "home")
  upper_dir = os.path.join(sbox_dir, "upper")
  work_dir = os.path.join(sbox_dir, "work")

  os.makedirs(home_dir, exist_ok=True)
  os.makedirs(upper_dir, exist_ok=True)
  os.makedirs(work_dir, exist_ok=True)

  ensure_micromamba(sandbox_dir)

  return home_dir, upper_dir, work_dir


def safe_rmtree(path: str, bwrap_bin: Optional[str] = None) -> None:
  """Safely removes a directory tree, handling OverlayFS permissions and 0000 work directories."""
  if not os.path.exists(path):
    return

  for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
      p = os.path.join(root, name)
      try:
        os.chmod(p, 0o777)
        os.unlink(p)
      except Exception:
        pass
    for name in dirs:
      p = os.path.join(root, name)
      try:
        os.chmod(p, 0o777)
        os.rmdir(p)
      except Exception:
        pass

  try:
    shutil.rmtree(path)
  except Exception:
    bin_path = bwrap_bin or shutil.which("bwrap")
    if bin_path and os.path.exists(bin_path):
      try:
        subprocess.run(
          [bin_path, "--unshare-user", "--bind", path, "/target", "rm", "-rf", "/target"],
          capture_output=True,
          timeout=5,
        )
      except Exception:
        pass
    if os.path.exists(path):
      shutil.rmtree(path, ignore_errors=True)


def reset_bwrap_sbox(sandbox_dir: str, bwrap_bin: Optional[str] = None) -> None:
  """Resets the bubblewrap sandbox writable state by nuking and re-creating .chatty/sbox/."""
  sbox_dir = get_bwrap_sbox_dir(sandbox_dir)
  safe_rmtree(sbox_dir, bwrap_bin=bwrap_bin)
  init_bwrap_sbox_dirs(sandbox_dir)


def get_bwrap_sbox_status(sandbox_dir: str) -> Dict[str, Any]:
  """Returns disk usage and file counts for the bubblewrap containment directories."""
  sbox_dir = get_bwrap_sbox_dir(sandbox_dir)
  home_dir = os.path.join(sbox_dir, "home")
  upper_dir = os.path.join(sbox_dir, "upper")

  bootstrapped_mamba = os.path.join(home_dir, ".local", "bin", "micromamba")

  def _dir_stats(path: str) -> Tuple[int, int]:
    total_size = 0
    file_count = 0
    if not os.path.exists(path):
      return 0, 0
    for root, _, files in os.walk(path):
      for f in files:
        fp = os.path.join(root, f)
        if fp == bootstrapped_mamba:
          continue
        file_count += 1
        try:
          total_size += os.path.getsize(fp)
        except OSError:
          pass
    return total_size, file_count

  home_size, home_files = _dir_stats(home_dir)
  upper_size, upper_files = _dir_stats(upper_dir)

  return {
    "sbox_dir": sbox_dir,
    "home_bytes": home_size,
    "home_files": home_files,
    "upper_bytes": upper_size,
    "upper_files": upper_files,
    "total_bytes": home_size + upper_size,
    "total_files": home_files + upper_files,
  }


def is_bwrap_overlay_supported(sandbox_dir: str, bwrap_bin: Optional[str] = None) -> bool:
  """Probes whether unprivileged OverlayFS on /usr is supported on this kernel and filesystem."""
  sbox_dir = get_bwrap_sbox_dir(sandbox_dir)
  if sbox_dir in _OVERLAY_SUPPORT_CACHE:
    return _OVERLAY_SUPPORT_CACHE[sbox_dir]

  bin_path = bwrap_bin or shutil.which("bwrap")
  if not bin_path or not os.path.exists(bin_path):
    _OVERLAY_SUPPORT_CACHE[sbox_dir] = False
    return False

  probe_dir = os.path.join(sbox_dir, ".probe_overlay")
  upper = os.path.join(probe_dir, "upper")
  work = os.path.join(probe_dir, "work")

  try:
    os.makedirs(upper, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    res = subprocess.run(
      [
        bin_path,
        "--unshare-user",
        "--ro-bind", "/", "/",
        "--overlay-src", "/usr",
        "--overlay", upper, work, "/usr",
        "true",
      ],
      capture_output=True,
      timeout=2,
    )
    supported = (res.returncode == 0)
  except Exception as e:
    logger.debug(f"OverlayFS probe failed: {e}")
    supported = False
  finally:
    safe_rmtree(probe_dir, bwrap_bin=bin_path)

  _OVERLAY_SUPPORT_CACHE[sbox_dir] = supported
  return supported


def maybe_bind(
  args: List[str],
  path: str,
  bind_type: str,
  dest: Optional[str] = None,
) -> bool:
  """Checks if path exists, and if so, adds a bind to args using bind_type, path as source, and dest or path as target."""
  if os.path.exists(path):
    target = dest or path
    args.extend([bind_type, path, target])
    return True
  return False


def wrap_command_with_bwrap(
  bwrap_bin: Optional[str],
  sandbox_dir: str,
  command: str,
  ro_paths: Optional[List[str]] = None,
  rw_paths: Optional[List[str]] = None,
  cwd: Optional[str] = None,
  enable_overlay: bool = True,
  isolated_home: bool = True,
) -> List[str]:
  """Wraps a shell command in a sandboxed bubblewrap invocation with OverlayFS and isolated home."""
  bin_path = bwrap_bin or shutil.which("bwrap") or "bwrap"
  abs_sandbox = os.path.realpath(sandbox_dir)
  working_dir = os.path.realpath(cwd) if cwd else abs_sandbox

  home_dir, upper_dir, work_dir = init_bwrap_sbox_dirs(abs_sandbox)

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

  maybe_bind(args, "/sys", "--ro-bind-try")

  args.extend(["--ro-bind", "/", "/"])
  args.extend(["--tmpfs", "/tmp"])

  if enable_overlay and is_bwrap_overlay_supported(abs_sandbox, bin_path):
    args.extend(["--overlay-src", "/usr", "--overlay", upper_dir, work_dir, "/usr"])

  host_home = os.path.expanduser("~")
  if isolated_home and os.path.exists(host_home):
    args.extend(["--bind", home_dir, host_home])
    args.extend(["--setenv", "HOME", host_home])
    mamba_prefix = f"{host_home}/.micromamba"
    args.extend(["--setenv", "MAMBA_ROOT_PREFIX", mamba_prefix])
    bin_paths = f"{mamba_prefix}/bin:{host_home}/.local/bin:/usr/local/bin:/usr/bin:/bin"
    args.extend(["--setenv", "PATH", bin_paths])

  maybe_bind(args, "/run/resolvconf/resolv.conf", "--ro-bind-try")
  maybe_bind(args, "/run/systemd/resolve", "--ro-bind-try")

  args.extend(["--bind", abs_sandbox, abs_sandbox])

  if ro_paths:
    for p in dict.fromkeys(os.path.realpath(path) for path in ro_paths):
      maybe_bind(args, p, "--ro-bind-try")

  if rw_paths:
    for p in dict.fromkeys(os.path.realpath(path) for path in rw_paths):
      maybe_bind(args, p, "--bind")

  args.extend(["--chdir", working_dir])
  args.extend(["--", "/bin/sh", "-c", command])
  return args
