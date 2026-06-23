import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.cli import ChatbotSession

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

if __name__ == "__main__":
  unittest.main()
