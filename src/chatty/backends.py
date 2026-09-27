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
      timeout=5
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
    reset_container: bool = False
  ):
    self.session_id = session_id
    self.host_sandbox = os.path.abspath(host_sandbox)
    self.image = image or get_default_image_tag()
    self.dockerfile = dockerfile
    self.build_image = build_image
    self.allowed_ro_paths = allowed_ro_paths or []
    self.allowed_rw_paths = allowed_rw_paths or []
    self.docker_bin = docker_bin
    self.container_workspace = container_workspace
    self.docker_host = docker_host
    self.status_callback = status_callback
    self.persistent = persistent
    self.reset_on_start = reset_container
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
    if self.dockerfile or self.build_image:
      self._notify(f"Building Docker companion image '{self.image}'...")
      self.build_companion_image(dockerfile=self.dockerfile, tag=self.image)
      rebuilt_image = True
    else:
      if not self._image_exists(self.image):
        if self.image.startswith("chatty-sandbox"):
          self._notify(f"Default image '{self.image}' not found. Building companion image (this may take a minute on first run)...")
          self.build_companion_image(dockerfile=get_default_dockerfile_path(), tag=self.image)
          rebuilt_image = True
        else:
          self._notify(f"Image '{self.image}' not found locally. Pulling companion image...")
          self._pull_image(self.image)

    if self.reset_on_start or rebuilt_image:
      subprocess.run([self.docker_bin, "rm", "-f", self.container_name], capture_output=True, env=self._get_docker_env())

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

  def build_companion_image(self, dockerfile: Optional[str] = None, tag: Optional[str] = None) -> None:
    target_tag = tag or self.image
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
      "-f", df_path,
      context_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=self._get_docker_env())
    if res.returncode != 0:
      raise RuntimeError(f"Docker build failed for image '{target_tag}':\n{res.stderr}\n{res.stdout}")
    logger.info(f"Successfully built Docker companion image '{target_tag}'.")

  def _resolve_host_path(self, path: str) -> str:
    env_override = os.environ.get("CHATTY_HOST_SANDBOX_DIR")
    if env_override and path.startswith(self.host_sandbox):
      rel = os.path.relpath(path, self.host_sandbox)
      return os.path.normpath(os.path.join(env_override, rel)) if rel != "." else env_override

    real_p = os.path.realpath(path)
    if os.path.exists("/proc/self/mountinfo"):
      try:
        with open("/proc/self/mountinfo", "r") as f:
          for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
              root_mount = parts[3]
              target_mount = parts[4]
              if root_mount != "/" and not root_mount.startswith("/.."):
                if target_mount == "/tmp" and root_mount.startswith("/sbox_tmp"):
                  if real_p == "/tmp" or real_p.startswith("/tmp/"):
                    rel = os.path.relpath(real_p, "/tmp")
                    return os.path.normpath(os.path.join(f"/tmp{root_mount}", rel)) if rel != "." else f"/tmp{root_mount}"
                elif target_mount.startswith("/home/") and root_mount.endswith("_sbox"):
                  if real_p == target_mount or real_p.startswith(target_mount + "/"):
                    rel = os.path.relpath(real_p, target_mount)
                    return os.path.normpath(os.path.join(f"/home{root_mount}", rel)) if rel != "." else f"/home{root_mount}"
      except Exception:
        pass
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
        return
      logger.warning(f"Failed to start existing container '{self.container_name}': {res.stderr}. Recreating...")
      subprocess.run([self.docker_bin, "rm", "-f", self.container_name], capture_output=True, env=self._get_docker_env())
    elif status == "paused":
      subprocess.run([self.docker_bin, "unpause", self.container_name], capture_output=True, env=self._get_docker_env())
      self.is_running = True
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
      "--label", "chatty.managed=true",
      "--label", f"chatty.session={self.session_id}",
      "-v", f"{host_mount_dir}:{self.container_workspace}:rw",
      "-w", self.container_workspace
    ])

    if hasattr(os, "getuid") and hasattr(os, "getgid"):
      cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

    try:
      import getpass
      username = getpass.getuser()
    except Exception:
      username = "chatty"

    user_home = f"/home/{username}"
    container_path = f"/workspace/.venv/bin:{user_home}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    cmd.extend([
      "-e", f"HOME={user_home}",
      "-e", f"PATH={container_path}"
    ])

    mounted_destinations = {self.container_workspace}

    for p in self.allowed_ro_paths:
      resolved = self._resolve_host_path(p)
      norm = os.path.normpath(resolved)
      if not norm or norm == "/":
        logger.warning(f"Skipping root '/' in Docker companion container volume mounts (destination '/' cannot be bound).")
        continue
      if norm in mounted_destinations:
        continue
      cmd.extend(["-v", f"{resolved}:{resolved}:ro"])
      mounted_destinations.add(norm)

    for p in self.allowed_rw_paths:
      resolved = self._resolve_host_path(p)
      norm = os.path.normpath(resolved)
      if not norm or norm == "/":
        logger.warning(f"Skipping root '/' in Docker companion container volume mounts (destination '/' cannot be bound).")
        continue
      if norm in mounted_destinations:
        continue
      cmd.extend(["-v", f"{resolved}:{resolved}:rw"])
      mounted_destinations.add(norm)

    cmd.extend([self.image, "sleep", "infinity"])

    logger.info(f"Starting companion container '{self.container_name}' with image '{self.image}'...")
    res = subprocess.run(cmd, capture_output=True, text=True, env=self._get_docker_env())
    if res.returncode != 0:
      raise RuntimeError(f"Failed to start companion Docker container '{self.container_name}': {res.stderr}")

    self.is_running = True
    logger.info(f"Companion container '{self.container_name}' started successfully.")

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

  def build_command_args(
    self,
    command: str,
    cwd: Optional[str] = None,
    task_id: Optional[str] = None
  ) -> Tuple[Union[List[str], str], bool, Optional[str]]:
    exec_cwd = self.container_workspace
    if cwd and cwd != self.host_sandbox:
      try:
        rel = os.path.relpath(cwd, self.host_sandbox)
        if not rel.startswith(".."):
          exec_cwd = os.path.normpath(f"{self.container_workspace}/{rel}")
      except ValueError:
        pass

    if task_id:
      wrapped_cmd = f"echo $$ > /tmp/{task_id}.pid; exec /bin/sh -c {shlex.quote(command)}"
    else:
      wrapped_cmd = command

    cmd_args = [
      self.docker_bin, "exec", "-i",
      "-w", exec_cwd,
      self.container_name,
      "/bin/sh", "-c", wrapped_cmd
    ]
    return cmd_args, False, None

  def terminate_task(self, task_id: str, proc: subprocess.Popen) -> None:
    if self.is_running and task_id:
      try:
        pid_res = subprocess.run(
          [self.docker_bin, "exec", self.container_name, "cat", f"/tmp/{task_id}.pid"],
          capture_output=True,
          text=True,
          timeout=2,
          env=self._get_docker_env()
        )
        if pid_res.returncode == 0 and pid_res.stdout.strip().isdigit():
          inner_pid = pid_res.stdout.strip()
          subprocess.run(
            [self.docker_bin, "exec", self.container_name, "kill", "-9", inner_pid],
            capture_output=True,
            timeout=2,
            env=self._get_docker_env()
          )
        subprocess.run(
          [self.docker_bin, "exec", self.container_name, "rm", "-f", f"/tmp/{task_id}.pid"],
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
