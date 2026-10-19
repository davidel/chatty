import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pytest

from chatty.backends import (
  DockerBackend,
  LocalBackend,
  get_default_dockerfile_path,
  get_host_git_config,
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

    self.assertIn("--shm-size=2g", run_cmd)

    self.assertEqual(backend.stdin_mode, subprocess.PIPE)

    # Verify command execution wrapper
    cmd_args, use_shell, cwd = backend.build_command_args("ls -la", cwd="/tmp/sandbox")
    self.assertFalse(use_shell)
    self.assertIsNone(cwd)
    self.assertEqual(cmd_args[:5], ["docker", "exec", "-i", "-w", "/workspace"])
    self.assertEqual(cmd_args[5], backend.container_name)
    self.assertEqual(cmd_args[6:], ["/bin/sh", "-c", "exec < /dev/null; exec /bin/sh -c 'ls -la'"])

    # Verify task_id wrapping for background process tracking
    task_args, _, _ = backend.build_command_args("sleep 20", task_id="task_42")
    self.assertIn("echo $$ > /tmp/task_42.pid", task_args[-1])
    self.assertIn("exec < /dev/null", task_args[-1])

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
    self.assertTrue(any("USER_ID=" in arg for arg in build_cmd))
    self.assertTrue(any("USER_NAME=" in arg for arg in build_cmd))
    self.assertFalse(any("GIT_USER_NAME=" in arg for arg in build_cmd))

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("chatty.backends.get_host_git_config", return_value=("Git Dev", "dev@example.com"))
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_injects_git_identity_at_runtime(self, mock_run, mock_which, mock_git, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    backend = DockerBackend(
      session_id="git_env_test",
      host_sandbox="/tmp/sandbox"
    )
    backend.initialize()

    run_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] == "run"
    ]
    self.assertTrue(len(run_calls) > 0)
    run_cmd = run_calls[0][0][0]
    self.assertIn("-e", run_cmd)
    self.assertIn("GIT_AUTHOR_NAME=Git Dev", run_cmd)
    self.assertIn("GIT_AUTHOR_EMAIL=dev@example.com", run_cmd)
    self.assertIn("GIT_COMMITTER_NAME=Git Dev", run_cmd)
    self.assertIn("GIT_COMMITTER_EMAIL=dev@example.com", run_cmd)

  @patch("subprocess.run")
  def test_get_host_git_config(self, mock_run):
    mock_run.side_effect = [
      MagicMock(returncode=0, stdout="Test User\n"),
      MagicMock(returncode=0, stdout="test@example.com\n")
    ]
    name, email = get_host_git_config()
    self.assertEqual(name, "Test User")
    self.assertEqual(email, "test@example.com")

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
  def test_docker_backend_resolves_and_mounts_correct_container_destination(self, mock_run, mock_which, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    backend = DockerBackend(
      session_id="dest_map_test",
      host_sandbox="/tmp/sandbox",
      allowed_ro_paths=["/home/davide/.cache"],
      allowed_rw_paths=["/tmp"]
    )

    mapping = {
      "/tmp": "/tmp/sbox_tmp.davide",
      "/home/davide/.cache": "/home/davide_sbox/.cache",
      "/tmp/sandbox": "/tmp/sbox_tmp.davide/sandbox"
    }
    with patch.object(backend, "_resolve_host_path", side_effect=lambda p: mapping.get(p, p)):
      backend.initialize()

    run_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] == "run"
    ]
    self.assertTrue(len(run_calls) > 0)
    run_cmd = run_calls[0][0][0]

    # Verify that the container destination is the sandbox path, not the resolved host path
    self.assertIn("/tmp/sbox_tmp.davide:/tmp:rw", run_cmd)
    self.assertIn("/home/davide_sbox/.cache:/home/davide/.cache:ro", run_cmd)
    self.assertNotIn("/tmp/sbox_tmp.davide:/tmp/sbox_tmp.davide:rw", run_cmd)
    self.assertNotIn("/home/davide_sbox/.cache:/home/davide_sbox/.cache:ro", run_cmd)

  @patch.dict(os.environ, {"HOST_PATH_MAP": "/tmp=/tmp/sbox_tmp.user:/home/user=/home/user_sbox:/home/user/git-repos=/home/user/git-repos"})
  def test_resolve_host_path_with_env_var(self):
    backend = DockerBackend(session_id="test", host_sandbox="/tmp")

    # Resolving under /tmp maps to /tmp/sbox_tmp.user
    self.assertEqual(backend._resolve_host_path("/tmp/chatty"), "/tmp/sbox_tmp.user/chatty")
    self.assertEqual(backend._resolve_host_path("/tmp"), "/tmp/sbox_tmp.user")

    # Resolving under /home/user maps to /home/user_sbox
    self.assertEqual(backend._resolve_host_path("/home/user/myproject"), "/home/user_sbox/myproject")

    # Nested/more specific prefix takes precedence over parent prefix
    self.assertEqual(backend._resolve_host_path("/home/user/git-repos/chatty"), "/home/user/git-repos/chatty")

    # Path not matching any mapping returns itself
    self.assertEqual(backend._resolve_host_path("/var/log"), "/var/log")

  def test_docker_backend_runner_uses_pipe_stdin(self):
    from chatty.session import ChatbotSession
    session = ChatbotSession(sandbox="/tmp/sandbox")
    mock_backend = MagicMock()
    mock_backend.stdin_mode = subprocess.PIPE
    mock_backend.build_command_args.return_value = (["docker", "exec", "-i", "c", "/bin/sh", "-c", "echo 1"], False, None)
    session.backend = mock_backend

    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    mock_proc.returncode = 0
    mock_proc.stdin = MagicMock()

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
      res = session.tool_run_command("pytest")
      self.assertEqual(mock_popen.call_args[1].get("stdin"), subprocess.PIPE)
      mock_proc.stdin.close.assert_called_once()

  def test_resolve_host_path_direct_param_and_fallback(self):
    backend = DockerBackend(
      session_id="test",
      host_sandbox="/tmp",
      host_path_map={"/tmp": "/host/tmp"}
    )
    self.assertEqual(backend._resolve_host_path("/tmp/code"), "/host/tmp/code")
    self.assertEqual(backend._resolve_host_path("/other/path"), "/other/path")

  def test_parse_host_path_map_formats(self):
    # Comma-separated
    parsed = DockerBackend._parse_host_path_map("/a=/host/a,/b=/host/b")
    self.assertEqual(parsed, [("/a", "/host/a"), ("/b", "/host/b")])

    # Colon-separated
    parsed = DockerBackend._parse_host_path_map("/short=/h/s:/longer/path=/h/l")
    # Longer prefix sorted first
    self.assertEqual(parsed, [("/longer/path", "/h/l"), ("/short", "/h/s")])

    # Dict and List inputs
    self.assertEqual(DockerBackend._parse_host_path_map({"/x": "/y"}), [("/x", "/y")])
    self.assertEqual(DockerBackend._parse_host_path_map([("/x", "/y")]), [("/x", "/y")])
    self.assertEqual(DockerBackend._parse_host_path_map(None), [])

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_mount_priority_and_sandbox_dedup(self, mock_run, mock_which, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    backend = DockerBackend(
      session_id="priority_test",
      host_sandbox="/tmp/sandbox",
      allowed_ro_paths=["/tmp/overlap", "/tmp/sandbox"],
      allowed_rw_paths=["/tmp/overlap", "/tmp/sandbox"]
    )
    backend.initialize()

    run_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] == "run"
    ]
    self.assertTrue(len(run_calls) > 0)
    run_cmd = run_calls[0][0][0]

    # RW should take precedence over RO for /tmp/overlap
    resolved_overlap = backend._resolve_host_path("/tmp/overlap")
    self.assertIn(f"{resolved_overlap}:/tmp/overlap:rw", run_cmd)
    self.assertNotIn(f"{resolved_overlap}:/tmp/overlap:ro", run_cmd)

    # Sandbox dir is already mounted at /workspace, so it should not be mounted again at /tmp/sandbox
    resolved_sandbox = backend._resolve_host_path("/tmp/sandbox")
    self.assertNotIn(f"{resolved_sandbox}:/tmp/sandbox:rw", run_cmd)
    self.assertNotIn(f"{resolved_sandbox}:/tmp/sandbox:ro", run_cmd)

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

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_reuses_running_container(self, mock_run, mock_which, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="running\n", stderr="")

    backend = DockerBackend(
      session_id="running_test",
      host_sandbox="/tmp/sandbox"
    )
    backend.initialize()

    self.assertTrue(backend.is_running)
    start_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] in ("run", "start")
    ]
    self.assertEqual(len(start_calls), 0)

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_starts_exited_container(self, mock_run, mock_which, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="exited\n", stderr="")

    backend = DockerBackend(
      session_id="exited_test",
      host_sandbox="/tmp/sandbox"
    )
    backend.initialize()

    self.assertTrue(backend.is_running)
    start_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] == "start"
    ]
    self.assertEqual(len(start_calls), 1)
    self.assertEqual(start_calls[0][0][0][2], backend.container_name)

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_reset_container(self, mock_run, mock_which, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    backend = DockerBackend(
      session_id="reset_test",
      host_sandbox="/tmp/sandbox"
    )
    backend.reset_container()

    rm_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] == "rm"
    ]
    self.assertTrue(len(rm_calls) > 0)
    self.assertIn(backend.container_name, rm_calls[0][0][0])

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_ephemeral_mode(self, mock_run, mock_which, mock_avail):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    backend = DockerBackend(
      session_id="ephemeral123",
      host_sandbox="/tmp/sandbox",
      persistent=False
    )
    backend.initialize()

    self.assertEqual(backend.container_name, "chatty-ephemeral123")
    run_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] == "run"
    ]
    self.assertTrue(len(run_calls) > 0)
    run_cmd = run_calls[0][0][0]
    self.assertIn("--rm", run_cmd)
    self.assertIn("chatty.ephemeral=true", run_cmd)

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("shutil.which", return_value="/usr/bin/docker")
  @patch("subprocess.run")
  def test_docker_backend_build_command_args_ensures_running(self, mock_run, mock_which, mock_avail):
    # Simulate container inspection returning 'exited', so ensure_running starts it
    mock_run.side_effect = [
      MagicMock(returncode=0, stdout="exited\n", stderr=""), # inspect
      MagicMock(returncode=0, stdout="", stderr=""),          # start
    ]

    backend = DockerBackend(
      session_id="exited_cmd_test",
      host_sandbox="/tmp/sandbox"
    )
    backend.is_running = False

    cmd_args, use_shell, cwd = backend.build_command_args("pytest")
    self.assertEqual(cmd_args[:2], ["docker", "exec"])
    self.assertTrue(backend.is_running)

    # Verify docker start was called
    start_calls = [
      call for call in mock_run.call_args_list
      if call[0] and len(call[0][0]) > 2 and call[0][0][1] == "start"
    ]
    self.assertEqual(len(start_calls), 1)
    self.assertEqual(start_calls[0][0][0][2], backend.container_name)


class TestSessionBackendWiring(unittest.TestCase):

  def setUp(self):
    self.orig_cwd = os.getcwd()
    self.temp_dir = tempfile.mkdtemp()
    os.chdir(self.temp_dir)

  def tearDown(self):
    os.chdir(self.orig_cwd)
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def test_session_backend_none(self):
    cfg = SessionConfig(
      provider="ollama",
      model="test-model",
      sandbox=self.temp_dir,
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
      sandbox=self.temp_dir,
      backend="auto",
      headless=True
    )
    with patch("chatty.session.ChatbotSession.init_client"):
      session = ChatbotSession(config=cfg)
      self.assertEqual(session.backend.get_name(), "docker")
      session.cleanup_background_commands()

  @patch("chatty.backends.is_docker_available", return_value=False)
  def test_auto_backend_selects_landlock_when_docker_unavailable(self, mock_avail):
    cfg = SessionConfig(
      provider="ollama",
      model="test-model",
      sandbox=self.temp_dir,
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
      sandbox=self.temp_dir,
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
      sandbox=self.temp_dir,
      docker_host="tcp://127.0.0.1:2375",
      headless=True
    )
    with patch("chatty.session.ChatbotSession.init_client"):
      session = ChatbotSession(config=cfg)
      self.assertEqual(session.backend.get_name(), "docker")
      self.assertEqual(session.backend.docker_host, "tcp://127.0.0.1:2375")
      session.cleanup_background_commands()

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("chatty.backends.DockerBackend.initialize")
  def test_docker_backend_does_not_mount_default_temp_by_default(self, mock_init, mock_avail):
    cfg = SessionConfig(
      provider="ollama",
      model="test-model",
      sandbox=self.temp_dir,
      backend="docker",
      headless=True
    )
    with patch("chatty.session.ChatbotSession.init_client"):
      session = ChatbotSession(config=cfg)
      default_temp = os.path.realpath(tempfile.gettempdir())
      self.assertNotIn(default_temp, session.backend.allowed_rw_paths)
      session.cleanup_background_commands()

  @patch("chatty.backends.is_docker_available", return_value=True)
  @patch("chatty.backends.DockerBackend.initialize")
  def test_docker_backend_mounts_temp_when_explicitly_whitelisted(self, mock_init, mock_avail):
    default_temp = os.path.realpath(tempfile.gettempdir())
    cfg = SessionConfig(
      provider="ollama",
      model="test-model",
      sandbox=self.temp_dir,
      backend="docker",
      whitelist=[f"{default_temp}:rw"],
      headless=True
    )
    with patch("chatty.session.ChatbotSession.init_client"):
      session = ChatbotSession(config=cfg)
      self.assertIn(default_temp, session.backend.allowed_rw_paths)
      session.cleanup_background_commands()

  def test_format_file_delegates_to_backend(self):
    from chatty.tools.file_ops import tool_format_file
    with tempfile.TemporaryDirectory() as td:
      src_file = os.path.join(td, "main.cpp")
      with open(src_file, "w") as f:
        f.write("int main(){return 0;}")

      mock_backend = MagicMock()
      mock_backend.get_name.return_value = "docker"
      mock_backend.build_command_args.return_value = (["echo", "formatted"], False, None)
      mock_backend._get_docker_env.return_value = {}

      mock_session = MagicMock()
      mock_session.backend = mock_backend

      with patch("chatty.tools.file_ops._is_tool_available", return_value=True):
        with patch("subprocess.run") as mock_subproc:
          mock_subproc.return_value = subprocess.CompletedProcess(args=["echo"], returncode=0, stdout="", stderr="")
          tool_format_file(td, "main.cpp", formatter="clang-format", session=mock_session)

          mock_backend.build_command_args.assert_called_once()
          called_cmd = mock_backend.build_command_args.call_args[0][0]
          self.assertIn("clang-format", called_cmd)
          self.assertIn("main.cpp", called_cmd)


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
      try:
        backend.initialize()
      except RuntimeError as e:
        self.skipTest(f"Docker live initialization failed: {e}")
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
