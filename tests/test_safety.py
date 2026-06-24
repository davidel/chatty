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
      "echo `cat file.txt`"
    ]
    for cmd in blocked_commands:
      with self.subTest(cmd=cmd):
        err = self.session.validate_command_safety(cmd)
        self.assertIsNotNone(err, f"Command should be blocked: {cmd}")
        self.assertTrue(err.startswith("Error:"), f"Expected error message, got: {err}")

  def test_allowed_commands(self):
    allowed_commands = [
      "pytest | grep Failed",
      "ninja test | head -n 20",
      "python -c 'import sys; print(sys.version)'",
      "make test",
      "git diff | grep 'pattern'"
    ]
    for cmd in allowed_commands:
      with self.subTest(cmd=cmd):
        err = self.session.validate_command_safety(cmd)
        self.assertIsNone(err, f"Command should be allowed: {cmd}")

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
      
      # Now check that task_1 is in background_commands
      self.assertIn("task_1", self.session.background_commands)
      
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
      self.session.background_commands.clear()
      self.session.next_task_id = 2
      
      res = self.session.tool_run_command("long_running_completes")
      self.assertIn("Task ID: task_2", res)
      
      status_res = self.session.tool_check_background_command("task_2", timeout=0.5)
      self.assertIn("FINISHED with exit code 0", status_res)
      self.assertNotIn("timed out", status_res)

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

if __name__ == "__main__":
  unittest.main()
