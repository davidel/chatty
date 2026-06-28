import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession

class TestCommandSafety(unittest.TestCase):
  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = tempfile.mkdtemp()
    self.session = ChatbotSession(
      provider="ollama",
      model="mock-model",
      context_size=8192,
      sandbox=self.sandbox_dir
    )

  def tearDown(self):
    self.session.cleanup_background_commands()
    os.chdir(self.old_cwd)
    shutil.rmtree(self.sandbox_dir)

  def test_sandbox_chdir(self):
    self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(self.sandbox_dir))

  def test_blocked_commands(self):
    blocked_commands = [
      "grep -rn 'opcode' .",
      "cd /tmp/vpu && grep -n 'opcode' SOME_PATH | head -30",
      "find . -name '*.py'",
      "cat file.txt",
      "head file.txt",
      "tail file.txt",
      "sed -n '1,10p' file.txt",
      "awk '{print $1}' file.txt",
      "less file.txt",
      "more file.txt",
      "cat << 'EOF' > file.txt",
      "echo $(cat file.txt)",
      "echo `cat file.txt`",
      "pytest | grep Failed",
      "ninja test | head -n 20",
      "git diff | grep 'pattern'",
      "make run_all 2>&1 | grep -E \"SIM_OUT|SIM_RC\" | head -5",
      "cp a.txt b.txt",
      "mv a.txt b.txt",
      "rm a.txt",
      "rmdir mydir",
      "mkdir mydir",
      "ls",
      "dir",
      "ls -la",
      "dir -a",
      "echo $(ls)"
    ]
    for cmd in blocked_commands:
      with self.subTest(cmd=cmd):
        err = self.session.validate_command_safety(cmd)
        self.assertIsNotNone(err, f"Command should be blocked: {cmd}")
        self.assertTrue(err.startswith("Error:"), f"Expected error message, got: {err}")

  def test_allowed_commands(self):
    allowed_commands = [
      "pytest",
      "ninja test",
      "python -c 'import sys; print(sys.version)'",
      "make test",
      "git diff"
    ]
    for cmd in allowed_commands:
      with self.subTest(cmd=cmd):
        err = self.session.validate_command_safety(cmd)
        self.assertIsNone(err, f"Command should be allowed: {cmd}")

  def test_grep_messages(self):
    bare_greps = [
      "grep -rn 'opcode' .",
      "cd /tmp/vpu && grep -n 'opcode' SOME_PATH | head -30"
    ]
    for cmd in bare_greps:
      err = self.session.validate_command_safety(cmd)
      self.assertIsNotNone(err)
      self.assertIn("prohibited to search files", err)
      self.assertIn("search_grep", err)

    piped_greps = [
      "pytest | grep Failed",
      "git diff | grep 'pattern'",
      "cd /tmp/vpu && make run_all 2>&1 | grep \"\\[.*/41\\]\""
    ]
    for cmd in piped_greps:
      err = self.session.validate_command_safety(cmd)
      self.assertIsNotNone(err)
      self.assertIn("prohibited to filter output", err)
      self.assertIn("output_filter", err)
      self.assertIn("check_background_command", err)

  def test_head_tail_awk_messages(self):
    bare_cmds = [
      "head file.txt",
      "tail file.txt",
      "awk '{print $1}' file.txt"
    ]
    for cmd in bare_cmds:
      err = self.session.validate_command_safety(cmd)
      self.assertIsNotNone(err)
      self.assertIn("prohibited to inspect files", err)
      self.assertIn("read_file", err)

    piped_head_tail = [
      "pytest | head -n 20",
      "make run_all | tail -5",
    ]
    for cmd in piped_head_tail:
      err = self.session.validate_command_safety(cmd)
      self.assertIsNotNone(err)
      self.assertIn("prohibited to filter output", err)
      self.assertIn("head_lines", err)
      self.assertIn("tail_lines", err)

    piped_awk = [
      "git log | awk '{print $1}'",
    ]
    for cmd in piped_awk:
      err = self.session.validate_command_safety(cmd)
      self.assertIsNotNone(err)
      self.assertIn("prohibited to filter output", err)
      self.assertIn("output_filter", err)

  def test_wc_messages(self):
    bare_wcs = [
      "wc file.txt",
      "wc -l file.txt"
    ]
    for cmd in bare_wcs:
      err = self.session.validate_command_safety(cmd)
      self.assertIsNotNone(err)
      self.assertIn("prohibited to count lines, words, or bytes in files", err)
      self.assertIn("get_file_info", err)

    piped_wcs = [
      "echo test/test_programs/test_*.asm | wc -l",
      "git status | wc -l"
    ]
    for cmd in piped_wcs:
      err = self.session.validate_command_safety(cmd)
      self.assertIsNotNone(err)
      self.assertIn("prohibited to count lines", err)
      self.assertIn("get_file_info", err)

  def test_get_rich_status_bar(self):
    self.session.messages.append({"role": "user", "content": "Hello"})
    status_bar = self.session.get_rich_status_bar()
    from rich.table import Table
    self.assertIsInstance(status_bar, Table)

  def test_run_command_filtering(self):
    # Test simple execution with filter
    res = self.session.tool_run_command("printf 'line1\\nline2\\nFAIL: test_error\\nline4\\n'", output_filter="FAIL")
    self.assertIn("FAIL: test_error", res)
    self.assertNotIn("line1", res)
    self.assertNotIn("line2", res)
    self.assertNotIn("line4", res)

    # Test execution with tail_lines
    res2 = self.session.tool_run_command("printf 'a\\nb\\nc\\nd\\ne\\n'", tail_lines=2)
    self.assertIn("d\ne", res2)
    self.assertNotIn("a\nb\nc", res2)

    # Test execution with head_lines
    res2_head = self.session.tool_run_command("printf 'a\\nb\\nc\\nd\\ne\\n'", head_lines=2)
    self.assertIn("a\nb", res2_head)
    self.assertNotIn("c\nd\ne", res2_head)

    # Test execution with both filter and tail_lines
    res3 = self.session.tool_run_command("printf 'FAIL: 1\\nline2\\nFAIL: 2\\nline4\\nFAIL: 3\\n'", output_filter="FAIL", tail_lines=2)
    self.assertIn("FAIL: 2\nFAIL: 3", res3)
    self.assertNotIn("FAIL: 1", res3)

    # Test execution with filter, head_lines, and tail_lines
    res4 = self.session.tool_run_command("printf 'FAIL: 1\\nline2\\nFAIL: 2\\nline4\\nFAIL: 3\\n'", output_filter="FAIL", head_lines=2, tail_lines=1)
    self.assertIn("FAIL: 2", res4)
    self.assertNotIn("FAIL: 1", res4)
    self.assertNotIn("FAIL: 3", res4)

  def test_run_command_combine_stderr(self):
    # Test execution with combine_stderr=False (default behavior)
    res_sep = self.session.tool_run_command("echo hello && echo error >&2", combine_stderr=False)
    self.assertIn("Stdout:", res_sep)
    self.assertIn("Stderr:", res_sep)
    self.assertIn("hello", res_sep)
    self.assertIn("error", res_sep)

    # Test execution with combine_stderr=True (combined behavior)
    res_comb = self.session.tool_run_command("echo hello && echo error >&2", combine_stderr=True)
    self.assertIn("Stdout:", res_comb)
    self.assertNotIn("Stderr:", res_comb)
    self.assertIn("hello", res_comb)
    self.assertIn("error", res_comb)

  def test_kill_process_and_background_command(self):
    import unittest.mock as mock
    import subprocess

    mock_proc = mock.MagicMock()
    mock_proc.wait.side_effect = subprocess.TimeoutExpired("long_running_process 100", 10)
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None

    with mock.patch("subprocess.Popen", return_value=mock_proc):
      res = self.session.tool_run_command("long_running_process 100")
      self.assertIn("running in the background", res)
      self.assertIn("Task ID: task_1", res)

      # Now check that task_1 is in background_commands and carries the task ID in temp files
      self.assertIn("task_1", self.session.background_commands)
      task_info = self.session.background_commands["task_1"]
      self.assertIn("task_1_stdout_", task_info["stdout_path"])
      self.assertIn("task_1_stderr_", task_info["stderr_path"])

      # Let's check status
      status_res = self.session.tool_check_background_command("task_1")
      self.assertIn("STILL RUNNING", status_res)

      # Now kill the process
      with mock.patch("os.killpg") as mock_killpg:
        kill_res = self.session.tool_kill_process("task_1")
        self.assertIn("Successfully terminated background task 'task_1'", kill_res)
        mock_killpg.assert_called_once_with(12345, 9) # signal.SIGKILL is 9

      # Ensure it was removed from background_commands
      self.assertNotIn("task_1", self.session.background_commands)

  def test_kill_process_already_exited(self):
    import unittest.mock as mock
    import subprocess

    mock_proc = mock.MagicMock()
    mock_proc.wait.side_effect = subprocess.TimeoutExpired("long_running_process 100", 10)
    mock_proc.pid = 12345
    mock_proc.poll.return_value = 0

    with mock.patch("subprocess.Popen", return_value=mock_proc):
      res = self.session.tool_run_command("long_running_process 100")
      self.assertIn("Task ID: task_1", res)

      # Kill the process
      kill_res = self.session.tool_kill_process("task_1")
      self.assertIn("had already exited with code 0", kill_res)

      self.assertNotIn("task_1", self.session.background_commands)

  def test_check_background_command_timeout(self):
    import unittest.mock as mock
    import subprocess

    # Mock a process that stays running (poll returns None)
    mock_proc = mock.MagicMock()
    mock_proc.wait.side_effect = subprocess.TimeoutExpired("long_running", 10)
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None

    with mock.patch("subprocess.Popen", return_value=mock_proc):
      res = self.session.tool_run_command("long_running")
      self.assertIn("Task ID: task_1", res)

      # Check status with timeout=0.1
      # Since poll.return_value is always None, it will time out
      status_res = self.session.tool_check_background_command("task_1", timeout=0.1)
      self.assertIn("STILL RUNNING", status_res)
      self.assertIn("timed out after 0.1 seconds", status_res)

      # Now, let's test where it finishes during the wait
      # We can mock poll() to return None first, then 0
      # (using side_effect=[None, None, 0, 0, 0])
      mock_proc.poll.side_effect = [None, None, 0, 0, 0]
      # Clear the existing background commands so task_2 can be created
      self.session.cleanup_background_commands()
      self.session.next_task_id = 2

      res = self.session.tool_run_command("long_running_completes")
      self.assertIn("Task ID: task_2", res)

      status_res = self.session.tool_check_background_command("task_2", timeout=0.5)
      self.assertIn("FINISHED with exit code 0", status_res)
      self.assertNotIn("timed out", status_res)

  def test_check_background_command_filters_and_persistence(self):
    import unittest.mock as mock
    import subprocess
    import tempfile

    mock_proc = mock.MagicMock()
    mock_proc.wait.side_effect = subprocess.TimeoutExpired("long_running_with_output", 10)
    mock_proc.pid = 12345
    mock_proc.poll.return_value = 0

    with mock.patch("subprocess.Popen", return_value=mock_proc):
      stdout_f = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
      stdout_f.write("Line 1: Info\nLine 2: Warning\nLine 3: Error!\nLine 4: Success\n")
      stdout_f.close()

      stderr_f = tempfile.NamedTemporaryFile(delete=False, mode='w+t')
      stderr_f.close()

      task_id = "task_99"
      self.session.background_commands[task_id] = {
        "proc": mock_proc,
        "command": "dummy",
        "stdout_path": stdout_f.name,
        "stderr_path": stderr_f.name,
        "stdout_file": stdout_f,
        "stderr_file": stderr_f,
        "output_filter": None,
        "tail_lines": None,
        "head_lines": None
      }

      res = self.session.tool_check_background_command(task_id)
      self.assertIn("FINISHED with exit code 0", res)
      self.assertIn("Line 1: Info", res)
      self.assertIn("Line 4: Success", res)

      self.assertIn(task_id, self.session.background_commands)

      res_filter = self.session.tool_check_background_command(task_id, output_filter="Error")
      self.assertIn("FINISHED with exit code 0", res_filter)
      self.assertIn("Line 3: Error!", res_filter)
      self.assertNotIn("Line 1: Info", res_filter)

      res_tail = self.session.tool_check_background_command(task_id, tail_lines=2)
      self.assertIn("FINISHED with exit code 0", res_tail)
      self.assertIn("Line 3: Error!", res_tail)
      self.assertIn("Line 4: Success", res_tail)
      self.assertNotIn("Line 1: Info", res_tail)

      self.session.tool_kill_process(task_id)
      self.assertNotIn(task_id, self.session.background_commands)

  def test_blocked_file_ops_commands(self):
    file_ops_checks = [
      ("cp a.txt b.txt", "prohibited to copy files or directories", "copy_file"),
      ("mv a.txt b.txt", "prohibited to move or rename files or directories", "move_file"),
      ("rm a.txt", "prohibited to delete files or directories", "delete_file"),
      ("rmdir mydir", "prohibited to delete directories", "delete_directory"),
      ("mkdir mydir", "prohibited to create directories", "make_directory"),
    ]
    for cmd, msg_part, tool_name in file_ops_checks:
      with self.subTest(cmd=cmd):
        err = self.session.validate_command_safety(cmd)
        self.assertIsNotNone(err)
        self.assertIn(msg_part, err)
        self.assertIn(tool_name, err)

  def test_blocked_kill_commands(self):
    for cmd in ["kill 1234", "pkill -f server", "killall python"]:
      err = self.session.validate_command_safety(cmd)
      self.assertIsNotNone(err)
      self.assertIn("prohibited to terminate processes", err)
      self.assertIn("kill_process", err)

  def test_blocked_sleep_commands(self):
    for cmd in ["sleep 5", "sleep 10", "sleep 0.5"]:
      err = self.session.validate_command_safety(cmd)
      self.assertIsNotNone(err)
      self.assertIn("prohibited to pause execution", err)
      self.assertIn("sleep", err)

  def test_sleep_tool_execution(self):
    from chatty.tools import execute_tool, tool_sleep
    # Test tool_sleep directly
    res = tool_sleep(0.01)
    self.assertIn("Successfully slept for 0.01 seconds", res)

    res = tool_sleep(-1)
    self.assertIn("Error", res)

    res = tool_sleep(100)
    self.assertIn("Error", res)

    # Test via execute_tool
    res = execute_tool("sleep", {"seconds": 0.02}, self.session)
    self.assertIn("Successfully slept for 0.02 seconds", res)

    # Test sleep prohibited when background tasks are active
    self.session.background_commands["task_1"] = {"dummy": True}
    res = execute_tool("sleep", {"seconds": 0.02}, self.session)
    self.assertIn("prohibited while background tasks", res)
    del self.session.background_commands["task_1"]

  def test_background_command_pruning(self):
    import io
    import subprocess
    import unittest.mock as mock
    self.session.max_completed_tasks = 2

    mock_running_proc = mock.MagicMock(spec=subprocess.Popen)
    mock_running_proc.poll.return_value = None  # Simulates still running

    mock_completed_proc_1 = mock.MagicMock(spec=subprocess.Popen)
    mock_completed_proc_1.poll.return_value = 0  # Simulates finished cleanly

    mock_completed_proc_3 = mock.MagicMock(spec=subprocess.Popen)
    mock_completed_proc_3.poll.return_value = 0

    mock_completed_proc_4 = mock.MagicMock(spec=subprocess.Popen)
    mock_completed_proc_4.poll.return_value = 0

    mock_file = mock.MagicMock(spec=io.IOBase)

    self.session.background_commands = {
      "task_1": {
        "proc": mock_completed_proc_1,
        "command": "completed_1",
        "stdout_path": "path_1",
        "stdout_file": mock_file,
        "status": 0
      },
      "task_2": {
        "proc": mock_running_proc,
        "command": "running_2",
        "stdout_path": "path_2",
        "stdout_file": mock_file
      },
      "task_3": {
        "proc": mock_completed_proc_3,
        "command": "completed_3",
        "stdout_path": "path_3",
        "stdout_file": mock_file,
        "status": 0
      },
      "task_4": {
        "proc": mock_completed_proc_4,
        "command": "completed_4",
        "stdout_path": "path_4",
        "stdout_file": mock_file,
        "status": 0
      }
    }

    with mock.patch("os.unlink") as mock_unlink, \
         mock.patch("subprocess.Popen") as mock_popen:

      self.session._prune_background_commands()

      # Assertions are now perfectly reliable and accurate
      self.assertNotIn("task_1", self.session.background_commands)
      mock_unlink.assert_any_call("path_1")

      self.assertIn("task_2", self.session.background_commands)
      self.assertIn("task_3", self.session.background_commands)
      self.assertIn("task_4", self.session.background_commands)


from chatty.safety import validate_command_safety

class TestSafetyModuleAndDelegation(unittest.TestCase):
  def test_safety_module_direct(self):
    # Test safety check directly from safety module
    err = validate_command_safety("cat /etc/passwd")
    self.assertIsNotNone(err)
    self.assertIn("read_file", err)

    err = validate_command_safety("echo 'hello'")
    self.assertIsNone(err)

  def test_chatbot_session_attribute_delegation(self):
    # Test dynamic configuration getter / setter delegation
    temp_dir = tempfile.mkdtemp()
    try:
      session = ChatbotSession(
        provider="openrouter",
        model="google/gemini-2.5-flash",
        context_size=4096,
        sandbox=temp_dir
      )
      self.assertEqual(session.provider, "openrouter")
      self.assertEqual(session.model, "google/gemini-2.5-flash")
      self.assertEqual(session.context_size, 4096)
      
      # Change attributes
      session.provider = "ollama"
      session.model = "llama3"
      session.context_size = 8192
      
      # Verify changes propagate to config object
      self.assertEqual(session.config.provider, "ollama")
      self.assertEqual(session.config.model, "llama3")
      self.assertEqual(session.config.context_size, 8192)
      
      # Verify changes reflect back on properties
      self.assertEqual(session.provider, "ollama")
      self.assertEqual(session.model, "llama3")
      self.assertEqual(session.context_size, 8192)

      # Non-existent attribute lookup
      with self.assertRaises(AttributeError):
        _ = session.non_existent_attribute_123

    finally:
      shutil.rmtree(temp_dir)


if __name__ == "__main__":
  unittest.main()

