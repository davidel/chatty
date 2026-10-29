import abc
import atexit
import hashlib
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("chatty")


def get_default_dockerfile_path() -> str:
  current_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.join(current_dir, "docker", "Dockerfile")


def get_default_image_tag() -> str:
  try:
    import getpass
    username = getpass.getuser().lower()
    safe_user = "".join(c for c in username if c.isalnum() or c in ("_", "-", "."))
    if safe_user:
      return f"chatty-sandbox:{safe_user}"
  except Exception:
    pass
  return "chatty-sandbox:latest"


def get_host_git_config() -> Tuple[str, str]:
  name = ""
  email = ""
  try:
    res = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, timeout=2)
    if res.returncode == 0 and res.stdout.strip():
      name = res.stdout.strip()
  except Exception:
    pass
  try:
    res = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, timeout=2)
    if res.returncode == 0 and res.stdout.strip():
      email = res.stdout.strip()
  except Exception:
    pass
  return name, email


def is_docker_available(docker_bin: str = "docker", docker_host: Optional[str] = None) -> bool:
  if not shutil.which(docker_bin):
    return False
  env = os.environ.copy()
  if docker_host:
    env["DOCKER_HOST"] = docker_host
  try:
    res = subprocess.run(
      [docker_bin, "info"],
      capture_output=True,
      env=env,
      timeout=15
    )
    return res.returncode == 0
  except Exception:
    return False


class ExecutionBackend(abc.ABC):

  @abc.abstractmethod
  def initialize(self) -> None:
    pass

  @abc.abstractmethod
  def build_command_args(
    self,
    command: str,
    cwd: Optional[str] = None,
    task_id: Optional[str] = None
  ) -> Tuple[Union[List[str], str], bool, Optional[str]]:
    pass

  @abc.abstractmethod
  def terminate_task(self, task_id: str, proc: subprocess.Popen) -> None:
    pass

  @abc.abstractmethod
  def cleanup(self) -> None:
    pass

  @abc.abstractmethod
  def get_name(self) -> str:
    pass

  @property
  def stdin_mode(self) -> int:
    return subprocess.DEVNULL


class LocalBackend(ExecutionBackend):

  def __init__(
    self,
    sandbox: str,
    landlock_bin: Optional[str] = None,
    allowed_rw_paths: Optional[Union[List[str], Callable[[], List[str]]]] = None
  ):
    self.sandbox = sandbox
    self.landlock_bin = landlock_bin
    self._allowed_rw_paths = allowed_rw_paths

  def initialize(self) -> None:
    pass

  def _get_rw_paths(self) -> List[str]:
    if callable(self._allowed_rw_paths):
      return list(self._allowed_rw_paths())
    elif self._allowed_rw_paths:
      return list(self._allowed_rw_paths)
    return []

  def build_command_args(
    self,
    command: str,
    cwd: Optional[str] = None,
    task_id: Optional[str] = None
  ) -> Tuple[Union[List[str], str], bool, Optional[str]]:
    if self.landlock_bin and self.sandbox:
      from chatty.landlock import wrap_command_with_landlock
      rw_paths = self._get_rw_paths()
      cmd_args = wrap_command_with_landlock(
        self.landlock_bin,
        self.sandbox,
        command,
        rw_paths=rw_paths
      )
      return cmd_args, False, self.sandbox
    else:
      return command, True, self.sandbox

  def terminate_task(self, task_id: str, proc: subprocess.Popen) -> None:
    if proc and proc.poll() is None:
      try:
        os.killpg(proc.pid, signal.SIGKILL)
      except Exception:
        try:
          proc.kill()
        except Exception:
          pass

  def cleanup(self) -> None:
    pass

  def get_name(self) -> str:
    return "landlock" if self.landlock_bin else "local"


class DockerBackend(ExecutionBackend):

  def __init__(
    self,
    session_id: str,
    host_sandbox: str,
    image: Optional[str] = None,
    base_image: Optional[str] = None,
    dockerfile: Optional[str] = None,
    build_image: bool = False,
    allowed_ro_paths: Optional[List[str]] = None,
    allowed_rw_paths: Optional[List[str]] = None,
    docker_bin: str = "docker",
    container_workspace: str = "/workspace",
    docker_host: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    container_name: Optional[str] = None,
    persistent: bool = True,
    reset_container: bool = False,
    host_path_map: Optional[Union[str, Dict[str, str], List[Tuple[str, str]]]] = None,
    enable_auth_proxy: bool = True
  ):
    self.session_id = session_id
    self.host_sandbox = os.path.abspath(host_sandbox)
    self.image = image or get_default_image_tag()
    self.base_image = base_image
    self.dockerfile = dockerfile
    self.build_image = build_image
    self.allowed_ro_paths = allowed_ro_paths or []
    self.allowed_rw_paths = allowed_rw_paths or []
    self.docker_bin = docker_bin
    self.container_workspace = container_workspace
    self.docker_host = docker_host or os.environ.get("DOCKER_HOST")
    self.status_callback = status_callback
    self.persistent = persistent
    self.reset_on_start = reset_container
    self.enable_auth_proxy = enable_auth_proxy
    self.auth_proxy = None
    self.host_path_map = self._parse_host_path_map(
      host_path_map or os.environ.get("HOST_PATH_MAP") or os.environ.get("CHATTY_HOST_PATH_MAP")
    )
    if container_name:
      self.container_name = container_name
    elif not self.persistent:
      self.container_name = f"chatty-{self.session_id}"
    else:
      workspace_hash = hashlib.sha256(self.host_sandbox.encode("utf-8")).hexdigest()[:12]
      folder_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.basename(self.host_sandbox.rstrip("/")) or "root")
      self.container_name = f"chatty-{folder_name}-{workspace_hash}"
    self.is_running = False
    self._cleaned_up = False

  def _notify(self, message: str) -> None:
    logger.info(message)
    if self.status_callback:
      try:
        self.status_callback(message)
      except Exception:
        pass

  def _get_docker_env(self) -> Dict[str, str]:
    env = os.environ.copy()
    if self.docker_host:
      env["DOCKER_HOST"] = self.docker_host
    return env

  def initialize(self) -> None:
    if self.docker_host:
      os.environ["DOCKER_HOST"] = self.docker_host

    if not shutil.which(self.docker_bin):
      raise RuntimeError(f"Docker executable '{self.docker_bin}' not found in PATH.")

    if not is_docker_available(self.docker_bin, docker_host=self.docker_host):
      raise RuntimeError(
        "Docker daemon is not running or not accessible. "
        "Ensure Docker is running and your user has permissions to access the Docker socket."
      )

    self._notify(f"Initializing Docker companion container (image: {self.image})...")
    self._prune_stale_containers()

    rebuilt_image = False
    if self.dockerfile or self.build_image or self.base_image:
      self._notify(f"Building Docker companion image '{self.image}'...")
      self.build_companion_image(dockerfile=self.dockerfile, tag=self.image, base_image=self.base_image)
      rebuilt_image = True
    else:
      if not self._image_exists(self.image):
        if self.image.startswith("chatty-sandbox"):
          self._notify(f"Default image '{self.image}' not found. Building companion image (this may take a minute on first run)...")
          self.build_companion_image(dockerfile=get_default_dockerfile_path(), tag=self.image, base_image=self.base_image)
          rebuilt_image = True
        else:
          self._notify(f"Image '{self.image}' not found locally. Pulling companion image...")
          self._pull_image(self.image)

    if self.reset_on_start or rebuilt_image:
      subprocess.run([self.docker_bin, "rm", "-f", self.container_name], capture_output=True, env=self._get_docker_env())

    if self.enable_auth_proxy and self.auth_proxy is None:
      try:
        from chatty.auth_proxy import create_default_credential_proxy
        self.auth_proxy = create_default_credential_proxy()
        if self.auth_proxy.routes:
          self.auth_proxy.start()
          self._notify(
            f"Credential proxy active on port {self.auth_proxy.port} "
            f"({len(self.auth_proxy.routes)} service routes configured)."
          )
      except Exception as e:
        logger.warning(f"Failed to initialize credential proxy: {e}")

    self._start_container()
    atexit.register(self.cleanup)

  def _image_exists(self, tag: str) -> bool:
    res = subprocess.run(
      [self.docker_bin, "image", "inspect", tag],
      capture_output=True,
      env=self._get_docker_env()
    )
    return res.returncode == 0

  def _pull_image(self, tag: str) -> None:
    res = subprocess.run(
      [self.docker_bin, "pull", tag],
      capture_output=True,
      text=True,
      env=self._get_docker_env()
    )
    if res.returncode != 0:
      raise RuntimeError(f"Failed to pull Docker image '{tag}': {res.stderr}")

  def build_companion_image(
    self,
    dockerfile: Optional[str] = None,
    tag: Optional[str] = None,
    base_image: Optional[str] = None
  ) -> None:
    target_tag = tag or self.image
    target_base = base_image or self.base_image
    df_path = dockerfile or get_default_dockerfile_path()
    if not os.path.exists(df_path):
      raise FileNotFoundError(f"Dockerfile not found at '{df_path}'")
    context_dir = os.path.dirname(os.path.abspath(df_path))
    logger.info(f"Building Docker image '{target_tag}' using Dockerfile '{df_path}'...")
    uid = str(os.getuid()) if hasattr(os, "getuid") else "1000"
    gid = str(os.getgid()) if hasattr(os, "getgid") else "1000"
    try:
      import getpass
      username = getpass.getuser()
    except Exception:
      username = "chatty"

    cmd = [
      self.docker_bin, "build",
      "-t", target_tag,
      "--build-arg", f"USER_ID={uid}",
      "--build-arg", f"GROUP_ID={gid}",
      "--build-arg", f"USER_NAME={username}",
    ]
    if target_base:
      cmd.extend(["--build-arg", f"BASE_IMAGE={target_base}"])
    cmd.extend([
      "-f", df_path,
      context_dir
    ])
    res = subprocess.run(cmd, capture_output=True, text=True, env=self._get_docker_env())
    if res.returncode != 0:
      raise RuntimeError(f"Docker build failed for image '{target_tag}':\n{res.stderr}\n{res.stdout}")
    logger.info(f"Successfully built Docker companion image '{target_tag}'.")

  @staticmethod
  def _parse_host_path_map(mapping_val: Optional[Union[str, Dict[str, str], List[Tuple[str, str]]]]) -> List[Tuple[str, str]]:
    """Parses a host path map specification into a sorted list of prefix pairs.

    Accepts:
      - A string of key=value pairs separated by comma, semicolon, or colon:
        e.g., "/tmp=/tmp/sbox_tmp.davide:/home/davide=/home/davide_sbox"
      - A dictionary mapping sandbox prefixes to host prefixes.
      - A list of (sandbox_prefix, host_prefix) tuples.

    Returns:
      A list of (sandbox_prefix, host_prefix) tuples sorted by sandbox prefix length
      descending so that more specific prefixes take precedence.
    """
    if not mapping_val:
      return []

    items: List[Tuple[str, str]] = []
    if isinstance(mapping_val, dict):
      items = list(mapping_val.items())
    elif isinstance(mapping_val, list):
      items = [(str(s), str(h)) for s, h in mapping_val]
    elif isinstance(mapping_val, str):
      for entry in re.split(r"[,;:]", mapping_val):
        entry = entry.strip()
        if "=" in entry:
          s_prefix, h_prefix = entry.split("=", 1)
          items.append((s_prefix.strip(), h_prefix.strip()))

    parsed: List[Tuple[str, str]] = []
    for s_p, h_p in items:
      s_clean = os.path.normpath(s_p)
      h_clean = os.path.normpath(h_p)
      if s_clean and h_clean:
        parsed.append((s_clean, h_clean))

    # Sort by sandbox prefix length descending so nested paths match before parent paths
    parsed.sort(key=lambda m: len(m[0]), reverse=True)
    return parsed

  def _resolve_host_path(self, path: str) -> str:
    """Translates a sandbox namespace path to its underlying physical path on the host.

    When Chatty runs inside an outer container or sandbox namespace (such as Bubblewrap)
    and connects to a Docker daemon running on the host, volume mount sources must be
    specified using the host's filesystem paths rather than sandbox-internal paths.

    If path mappings are configured via the `HOST_PATH_MAP` environment variable
    (format: `/sandbox/path=/host/path:/other/sandbox=/other/host`) or passed directly
    to DockerBackend, matching sandbox path prefixes are translated to their corresponding
    host prefixes. More specific (longer) prefixes take precedence.

    If no mapping matches, the path is returned as-is (canonicalized via realpath).

    Args:
      path: The sandbox-relative or absolute filesystem path to resolve.

    Returns:
      The mapped host path if a matching prefix exists in HOST_PATH_MAP,
      otherwise the canonical realpath on the local filesystem.
    """
    real_p = os.path.realpath(path)
    for sandbox_prefix, host_prefix in self.host_path_map:
      s_clean = sandbox_prefix.rstrip("/")
      if real_p == sandbox_prefix or real_p.startswith(s_clean + "/"):
        rel = os.path.relpath(real_p, sandbox_prefix)
        return os.path.normpath(os.path.join(host_prefix, rel)) if rel != "." else host_prefix
    return real_p

  def _get_container_status(self) -> Optional[str]:
    try:
      res = subprocess.run(
        [self.docker_bin, "inspect", "-f", "{{.State.Status}}", self.container_name],
        capture_output=True,
        text=True,
        env=self._get_docker_env()
      )
      if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip().lower()
    except Exception:
      pass
    return None

  def reset_container(self) -> None:
    self._notify(f"Resetting Docker companion container '{self.container_name}'...")
    subprocess.run(
      [self.docker_bin, "rm", "-f", self.container_name],
      capture_output=True,
      env=self._get_docker_env()
    )
    self.is_running = False
    self._cleaned_up = False
    self._start_container()

  def _start_container(self) -> None:
    if self.is_running:
      return

    status = self._get_container_status()
    if status == "running":
      self.is_running = True
      self._notify(f"Reusing running companion container '{self.container_name}'.")
      self._bootstrap_container()
      return
    elif status in ("exited", "created"):
      self._notify(f"Starting existing companion container '{self.container_name}'...")
      res = subprocess.run(
        [self.docker_bin, "start", self.container_name],
        capture_output=True,
        text=True,
        env=self._get_docker_env()
      )
      if res.returncode == 0:
        self.is_running = True
        logger.info(f"Companion container '{self.container_name}' started successfully.")
        self._bootstrap_container()
        return
      logger.warning(f"Failed to start existing container '{self.container_name}': {res.stderr}. Recreating...")
      subprocess.run([self.docker_bin, "rm", "-f", self.container_name], capture_output=True, env=self._get_docker_env())
    elif status == "paused":
      subprocess.run([self.docker_bin, "unpause", self.container_name], capture_output=True, env=self._get_docker_env())
      self.is_running = True
      self._bootstrap_container()
      return

    self._notify(f"Starting Docker companion container '{self.container_name}'...")
    subprocess.run([self.docker_bin, "rm", "-f", self.container_name], capture_output=True, env=self._get_docker_env())

    host_mount_dir = self._resolve_host_path(self.host_sandbox)
    cmd = [
      self.docker_bin, "run", "-d",
      "--name", self.container_name,
    ]

    if not self.persistent:
      cmd.append("--rm")
      cmd.extend(["--label", "chatty.ephemeral=true"])
    else:
      cmd.extend([
        "--label", "chatty.persistent=true",
        "--label", f"chatty.workspace={self.host_sandbox}"
      ])

    cmd.extend([
      "--init",
      "--add-host", "host.docker.internal:host-gateway",
      "--shm-size=2g",
      "--label", "chatty.managed=true",
      "--label", f"chatty.session={self.session_id}",
      "-v", f"{host_mount_dir}:{self.container_workspace}:rw",
      "-w", self.container_workspace
    ])

    rw_dests = {os.path.normpath(p) for p in self.allowed_rw_paths}
    if "/tmp" not in rw_dests:
      cmd.extend(["--tmpfs", "/tmp:rw,exec,nosuid,nodev"])

    if hasattr(os, "getuid") and hasattr(os, "getgid"):
      cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

    try:
      import getpass
      username = getpass.getuser()
    except Exception:
      username = "chatty"

    user_home = f"/home/{username}"
    container_path = f"/workspace/.venv/bin:{user_home}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    git_name, git_email = get_host_git_config()
    if not git_name:
      git_name = username
    if not git_email:
      git_email = f"{username}@localhost"

    cmd.extend([
      "-e", f"HOME={user_home}",
      "-e", f"PATH={container_path}",
      "-e", f"GIT_AUTHOR_NAME={git_name}",
      "-e", f"GIT_AUTHOR_EMAIL={git_email}",
      "-e", f"GIT_COMMITTER_NAME={git_name}",
      "-e", f"GIT_COMMITTER_EMAIL={git_email}"
    ])

    if self.auth_proxy and self.auth_proxy.is_running:
      for env_k, env_v in self.auth_proxy.get_container_env().items():
        cmd.extend(["-e", f"{env_k}={env_v}"])

    mounted_destinations = {
      os.path.normpath(self.container_workspace),
      os.path.normpath(self.host_sandbox)
    }

    # Mount RW paths first (RW takes precedence if overlapping), then RO
    for paths, mode in ((self.allowed_rw_paths, "rw"), (self.allowed_ro_paths, "ro")):
      for p in paths:
        dest = os.path.normpath(p)
        if not dest or dest == "/":
          logger.warning("Skipping root '/' in Docker companion container volume mounts (destination '/' cannot be bound).")
          continue
        if dest in mounted_destinations:
          continue
        resolved = self._resolve_host_path(p)
        cmd.extend(["-v", f"{resolved}:{dest}:{mode}"])
        mounted_destinations.add(dest)

    cmd.extend([self.image, "sleep", "infinity"])

    logger.info(f"Starting companion container '{self.container_name}' with image '{self.image}'...")
    res = subprocess.run(cmd, capture_output=True, text=True, env=self._get_docker_env())
    if res.returncode != 0:
      raise RuntimeError(f"Failed to start companion Docker container '{self.container_name}': {res.stderr}")

    self.is_running = True
    logger.info(f"Companion container '{self.container_name}' started successfully.")
    self._bootstrap_container()

  def _bootstrap_container(self) -> None:
    subprocess.run(
      [self.docker_bin, "exec", self.container_name, "mkdir", "-p", "/tmp/.chatty"],
      capture_output=True,
      env=self._get_docker_env()
    )
    git_name, git_email = get_host_git_config()
    if git_name:
      subprocess.run(
        [self.docker_bin, "exec", self.container_name, "git", "config", "--global", "user.name", git_name],
        capture_output=True,
        env=self._get_docker_env()
      )
    if git_email:
      subprocess.run(
        [self.docker_bin, "exec", self.container_name, "git", "config", "--global", "user.email", git_email],
        capture_output=True,
        env=self._get_docker_env()
      )

  def _prune_stale_containers(self) -> None:
    try:
      cmd = [
        self.docker_bin, "ps", "-q", "-a",
        "--filter", "label=chatty.ephemeral=true"
      ]
      res = subprocess.run(cmd, capture_output=True, text=True, env=self._get_docker_env())
      if res.returncode == 0 and res.stdout.strip():
        ids = res.stdout.strip().split()
        for cid in ids:
          subprocess.run([self.docker_bin, "rm", "-f", cid], capture_output=True, env=self._get_docker_env())
    except Exception as e:
      logger.debug(f"Failed to prune stale containers: {e}")

  def ensure_running(self) -> None:
    """Ensures the companion container is running, reviving it if exited or paused."""
    if not self.is_running:
      self._start_container()
    else:
      status = self._get_container_status()
      if status != "running":
        self.is_running = False
        self._start_container()

  @property
  def stdin_mode(self) -> int:
    return subprocess.PIPE

  def build_command_args(
    self,
    command: str,
    cwd: Optional[str] = None,
    task_id: Optional[str] = None
  ) -> Tuple[Union[List[str], str], bool, Optional[str]]:
    self.ensure_running()
    exec_cwd = self.container_workspace
    if cwd and cwd != self.host_sandbox:
      try:
        rel = os.path.relpath(cwd, self.host_sandbox)
        if not rel.startswith(".."):
          exec_cwd = os.path.normpath(f"{self.container_workspace}/{rel}")
      except ValueError:
        pass

    if task_id:
      wrapped_cmd = f"echo $$ > /tmp/.chatty/{task_id}.pid; exec < /dev/null; exec /bin/sh -c {shlex.quote(command)}"
    else:
      wrapped_cmd = f"exec < /dev/null; exec /bin/sh -c {shlex.quote(command)}"

    cmd_args = [
      self.docker_bin, "exec", "-i",
      "-w", exec_cwd,
    ]
    if self.auth_proxy and self.auth_proxy.is_running:
      for env_k, env_v in self.auth_proxy.get_container_env().items():
        cmd_args.extend(["-e", f"{env_k}={env_v}"])
    cmd_args.extend([
      self.container_name,
      "/bin/sh", "-c", wrapped_cmd
    ])
    return cmd_args, False, None

  def terminate_task(self, task_id: str, proc: subprocess.Popen) -> None:
    if self.is_running and task_id:
      try:
        pid_res = subprocess.run(
          [self.docker_bin, "exec", self.container_name, "cat", f"/tmp/.chatty/{task_id}.pid"],
          capture_output=True,
          text=True,
          timeout=2,
          env=self._get_docker_env()
        )
        if pid_res.returncode == 0 and pid_res.stdout.strip().isdigit():
          inner_pid = pid_res.stdout.strip()
          kill_script = (
            f"kill -TERM -{inner_pid} 2>/dev/null || kill -TERM {inner_pid} 2>/dev/null; "
            f"pkill -TERM -P {inner_pid} 2>/dev/null || true; "
            f"sleep 0.05; "
            f"kill -KILL -{inner_pid} 2>/dev/null || kill -KILL {inner_pid} 2>/dev/null; "
            f"pkill -KILL -P {inner_pid} 2>/dev/null || true"
          )
          subprocess.run(
            [self.docker_bin, "exec", self.container_name, "/bin/sh", "-c", kill_script],
            capture_output=True,
            timeout=5,
            env=self._get_docker_env()
          )
        subprocess.run(
          [self.docker_bin, "exec", self.container_name, "rm", "-f", f"/tmp/.chatty/{task_id}.pid"],
          capture_output=True,
          timeout=2,
          env=self._get_docker_env()
        )
      except Exception as e:
        logger.debug(f"Error terminating inner container process for task {task_id}: {e}")

    if proc and proc.poll() is None:
      try:
        os.killpg(proc.pid, signal.SIGKILL)
      except Exception:
        try:
          proc.kill()
        except Exception:
          pass

  def cleanup(self) -> None:
    if self._cleaned_up:
      return
    self._cleaned_up = True
    if self.auth_proxy:
      try:
        self.auth_proxy.stop()
      except Exception as e:
        logger.debug(f"Error stopping credential proxy: {e}")
      self.auth_proxy = None
    if self.is_running:
      logger.info(f"Stopping companion container '{self.container_name}'...")
      try:
        subprocess.run(
          [self.docker_bin, "stop", "-t", "2", self.container_name],
          capture_output=True,
          timeout=10,
          env=self._get_docker_env()
        )
      except Exception as e:
        logger.warning(f"Error stopping container '{self.container_name}': {e}")
      finally:
        self.is_running = False

  def get_name(self) -> str:
    return "docker"
