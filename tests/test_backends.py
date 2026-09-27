import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

from chatty.backends import (
  DockerBackend,
  LocalBackend,
  get_default_dockerfile_path,
  is_docker_available
)
from chatty.session import ChatbotSession, SessionConfig


class TestLocalBackend(unittest.TestCase):

  def test_local_backend_without_landlock(self):
    backend = LocalBackend(sandbox="/tmp/sandbox", landlock_bin=None)
    backend.initialize()
    args, use_shell, cwd = backend.build_command_args("echo 123", cwd="/tmp/sandbox")
    self.assertEqual(args, "echo 123")
    self.assertTrue(use_shell)
    self.assertEqual(cwd, "/tmp/sandbox")
    self.assertEqual(backend.get_name(), "local")

  def test_local_backend_with_landlock(self):
    backend = LocalBackend(
      sandbox="/tmp/sandbox",
      landlock_bin="/tmp/landlock_exec",
      allowed_rw_paths=lambda: ["/tmp/extra"]
    )
    with patch("chatty.landlock.wrap_command_with_landlock") as mock_wrap:
      mock_wrap.return_value = ["/tmp/landlock_exec", "--ro", "/", "--", "/bin/sh", "-c", "echo 123"]
      args, use_shell, cwd = backend.build_command_args("echo 123", cwd="/tmp/sandbox")
      self.assertFalse(use_shell)
      self.assertEqual(cwd, "/tmp/sandbox")
      mock_wrap.assert_called_once()
      self.assertEqual(backend.get_name(), "landlock")


class TestDockerBackendUnit(unittest.TestCase):

  @patch("shutil.which")
  def test_docker_missing_executable(self, mock_which):
    mock_which.return_value = None
    self.assertFalse(is_docker_available("docker"))
    backend = DockerBackend(session_id="test1", host_sandbox="/tmp/sandbox")
    with self.assertRaises(RuntimeError) as ctx:
      backend.initialize()
    self.assertIn("not found in PATH", str(ctx.exception))

  @patch("chatty.backends.is_docker_available", return_value=False)
  @patch("shutil.which", return_value="/usr/bin/docker")
  def test_docker_daemon_not_running(self, mock_which, mock_avail):
    backend = DockerBackend(session_id="test1", host_sandbox="/tmp/sandbox")
    with self.assertRaises(RuntimeError) as ctx:
      backend.initialize()
    self.assertIn("daemon is not running", str(ctx.exception))

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_start_and_command_args(self, mock_run, mock_which, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    backend = DockerBackend(
      session_id="s123",
      host_sandbox="/tmp/sandbox",
      image="test-image:latest",
      allowed_ro_paths=["/tmp/ro_dir"],
      allowed_rw_paths=["/tmp/rw_dir"]
    )
    backend.initialize()

    self.assertTrue(backend.is_running)
    self.assertEqual(backend.get_name(), "docker")

    run_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] == "run"
    ]
    self.assertTrue(len(run_calls) > 0)
    run_cmd = run_calls[0][0][0]
    self.assertTrue(any(arg.startswith("HOME=") for arg in run_cmd))
    self.assertTrue(any(arg.startswith("PATH=") and ".local/bin" in arg for arg in run_cmd))

    # Verify command execution wrapper
    cmd_args, use_shell, cwd = backend.build_command_args("ls -la", cwd="/tmp/sandbox")
    self.assertFalse(use_shell)
    self.assertIsNone(cwd)
    self.assertEqual(cmd_args[:5], ["docker", "exec", "-i", "-w", "/workspace"])
    self.assertEqual(cmd_args[5], "chatty-s123")
    self.assertEqual(cmd_args[6:], ["/bin/sh", "-c", "ls -la"])

    # Verify task_id wrapping for background process tracking
    task_args, _, _ = backend.build_command_args("sleep 20", task_id="task_42")
    self.assertIn("echo $$ > /tmp/task_42.pid", task_args[-1])

    # Test task termination
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 9999

    cat_mock = MagicMock(returncode=0, stdout="1234\n", stderr="")
    kill_mock = MagicMock(returncode=0, stdout="", stderr="")
    mock_run.side_effect = [cat_mock, kill_mock, MagicMock(returncode=0)]

    with patch("os.killpg") as mock_killpg:
      backend.terminate_task("task_42", proc)
      mock_killpg.assert_called_once_with(9999, unittest.mock.ANY)

    # Test container cleanup
    backend.cleanup()
    self.assertFalse(backend.is_running)

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_build_image(self, mock_run, mock_which, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    backend = DockerBackend(
      session_id="build_test",
      host_sandbox="/tmp/sandbox",
      image="custom-image:v1",
      build_image=True
    )
    backend.initialize()

    build_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] == "build"
    ]
    self.assertTrue(len(build_calls) > 0)
    build_cmd = build_calls[0][0][0]
    self.assertEqual(build_cmd[3], "custom-image:v1")
    self.assertIn("--build-arg", build_cmd)

  def test_docker_host_env_handling(self):
    backend = DockerBackend(
      session_id="host_test",
      host_sandbox="/tmp/sandbox",
      docker_host="tcp://custom-host:2375"
    )
    env = backend._get_docker_env()
    self.assertEqual(env["DOCKER_HOST"], "tcp://custom-host:2375")

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_skips_root_mount(self, mock_run, mock_which, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    backend = DockerBackend(
      session_id="root_mount_test",
      host_sandbox="/tmp/sandbox",
      allowed_ro_paths=["/"],
      allowed_rw_paths=["/"]
    )
    backend.initialize()

    run_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] == "run"
    ]
    self.assertTrue(len(run_calls) > 0)
    run_cmd = run_calls[0][0][0]
    self.assertNotIn("/:/:ro", run_cmd)
    self.assertNotIn("/:/:rw", run_cmd)

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_status_callback(self, mock_run, mock_which, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    messages = []
    backend = DockerBackend(
      session_id="status_test",
      host_sandbox="/tmp/sandbox",
      status_callback=messages.append
    )
    backend.initialize()

    self.assertTrue(len(messages) >= 2)
    self.assertTrue(any("Initializing Docker companion container" in m for m in messages))
    self.assertTrue(any("Starting Docker companion container" in m for m in messages))


class TestSessionBackendWiring(unittest.TestCase):

  def test_session_backend_none(self):
    cfg = SessionConfig(
      provider="ollama",
      model="test-model",
      backend="none",
      headless=True
    )
    with patch("chatty.session.ChatbotSession.init_client"):
      session = ChatbotSession(config=cfg)
      self.assertEqual(session.backend.get_name(), "local")
      session.cleanup_background_commands()

  def test_default_dockerfile_exists(self):
    df_path = get_default_dockerfile_path()
    self.assertTrue(os.path.exists(df_path))

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("chatty.backends.DockerBackend.initialize")
  def test_auto_backend_selects_docker_when_available(self, mock_init, mock_avail):
    cfg = SessionConfig(
      provider="ollama",
      model="test-model",
      backend="auto",
      headless=True
    )
    with patch.dict(os.environ, {}, clear=False):
      os.environ.pop("PYTEST_CURRENT_TEST", None)
      with patch("chatty.session.ChatbotSession.init_client"):
        session = ChatbotSession(config=cfg)
        self.assertEqual(session.backend.get_name(), "docker")
        session.cleanup_background_commands()

  @patch("chatty.backends.is_docker_available", return_value=False)
  def test_auto_backend_selects_landlock_when_docker_unavailable(self, mock_avail):
    cfg = SessionConfig(
      provider="ollama",
      model="test-model",
      backend="auto",
      headless=True
    )
    with patch("chatty.session.ChatbotSession.init_client"):
      with patch("sys.platform", "linux"):
        session = ChatbotSession(config=cfg)
        if session.landlock_bin:
          self.assertEqual(session.backend.get_name(), "landlock")
        session.cleanup_background_commands()

  @patch("chatty.backends.is_docker_available", return_value=False)
  def test_auto_backend_fails_when_sandbox_specified_without_capabilities(self, mock_avail):
    cfg = SessionConfig(
      provider="ollama",
      model="test-model",
      backend="auto",
      sandbox_specified=True,
      headless=True
    )
    with patch("chatty.session.ChatbotSession.init_client"):
      with patch("sys.platform", "darwin"):
        with self.assertRaises(RuntimeError) as ctx:
          ChatbotSession(config=cfg)
        self.assertIn("No sandboxing capabilities found", str(ctx.exception))

  @patch("chatty.backends.is_docker_available", return_value=False)
  def test_auto_backend_falls_back_when_no_sandbox_specified_and_no_capabilities(self, mock_avail):
    cfg = SessionConfig(
      provider="ollama",
      model="test-model",
      backend="auto",
      sandbox_specified=False,
      headless=True
    )
    with patch("chatty.session.ChatbotSession.init_client"):
      with patch("sys.platform", "darwin"):
        session = ChatbotSession(config=cfg)
        self.assertEqual(session.backend.get_name(), "local")
        session.cleanup_background_commands()

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("chatty.backends.DockerBackend.initialize")
  def test_session_with_docker_host(self, mock_init, mock_avail):
    cfg = SessionConfig(
      provider="ollama",
      model="test-model",
      docker_host="tcp://127.0.0.1:2375",
      headless=True
    )
    with patch("chatty.session.ChatbotSession.init_client"):
      session = ChatbotSession(config=cfg)
      self.assertEqual(session.backend.get_name(), "docker")
      self.assertEqual(session.backend.docker_host, "tcp://127.0.0.1:2375")
      session.cleanup_background_commands()


class TestLiveDockerCompanionIntegration(unittest.TestCase):

  @pytest.mark.docker
  def test_live_docker_companion_flow(self):
    if not is_docker_available("docker"):
      self.skipTest("Docker daemon not available for live test")

    tmp_dir = tempfile.mkdtemp(prefix="chatty_live_docker_test_")
    try:
      backend = DockerBackend(
        session_id="livetest",
        host_sandbox=tmp_dir,
        image="alpine:latest"
      )
      backend.initialize()
      self.assertTrue(backend.is_running)

      # 1. Execute a command that creates a file
      cmd_args, _, _ = backend.build_command_args("echo 'live_companion_test' > /workspace/test.txt")
      res = subprocess.run(cmd_args, capture_output=True, text=True)
      self.assertEqual(res.returncode, 0)

      # 2. Verify file content via another exec command
      read_args, _, _ = backend.build_command_args("cat /workspace/test.txt")
      read_res = subprocess.run(read_args, capture_output=True, text=True)
      self.assertEqual(read_res.returncode, 0)
      self.assertEqual(read_res.stdout.strip(), "live_companion_test")

      # 3. Test background task tracking and kill
      task_args, _, _ = backend.build_command_args("sleep 60", task_id="task_live_1")
      proc = subprocess.Popen(task_args)
      import time
      time.sleep(1)

      # Check top inside container has sleep
      top_out = subprocess.check_output(["docker", "top", backend.container_name], text=True)
      self.assertIn("sleep 60", top_out)

      # Terminate task
      backend.terminate_task("task_live_1", proc)
      proc.wait(timeout=5)
      time.sleep(1)

      top_out_after = subprocess.check_output(["docker", "top", backend.container_name], text=True)
      self.assertNotIn("sleep 60", top_out_after)

      # Cleanup
      backend.cleanup()
      self.assertFalse(backend.is_running)

      # Confirm container is gone
      inspect_res = subprocess.run(["docker", "inspect", backend.container_name], capture_output=True)
      self.assertNotEqual(inspect_res.returncode, 0)
    finally:
      shutil.rmtree(tmp_dir, ignore_errors=True)
