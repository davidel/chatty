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

if __name__ == "__main__":
  unittest.main()
