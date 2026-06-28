import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.landlock import compile_landlock_binary, wrap_command_with_landlock


class TestLandlock(unittest.TestCase):

  def setUp(self):
    self.sandbox_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.sandbox_dir)

  def test_compile_binary(self):
    if sys.platform != "linux":
      self.skipTest("Landlock is only supported on Linux")

    binary_path = compile_landlock_binary()
    self.assertIsNotNone(binary_path)
    self.assertTrue(os.path.exists(binary_path))
    self.assertTrue(os.access(binary_path, os.X_OK))

  def test_landlock_restrictions(self):
    if sys.platform != "linux":
      self.skipTest("Landlock is only supported on Linux")

    binary_path = compile_landlock_binary()
    self.assertIsNotNone(binary_path)

    # 1. Test writing inside sandbox (should succeed)
    test_file_sandbox = os.path.join(self.sandbox_dir, "test_sb.txt")
    cmd_args = wrap_command_with_landlock(
      binary_path,
      self.sandbox_dir,
      f"echo 'hello' > {test_file_sandbox}"
    )

    proc = subprocess.run(cmd_args, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0, f"Failed to write in sandbox: {proc.stderr}")
    self.assertTrue(os.path.exists(test_file_sandbox))
    with open(test_file_sandbox, "r") as f:
      self.assertEqual(f.read().strip(), "hello")

    # 2. Test reading outside sandbox (should succeed since we have --ro /)
    test_read_passwd = os.path.join(self.sandbox_dir, "passwd.txt")
    cmd_args = wrap_command_with_landlock(
      binary_path,
      self.sandbox_dir,
      f"cat /etc/passwd > {test_read_passwd}"
    )

    proc = subprocess.run(cmd_args, capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0, f"Failed to read /etc/passwd: {proc.stderr}")
    self.assertTrue(os.path.exists(test_read_passwd))
    self.assertGreater(os.path.getsize(test_read_passwd), 0)

    # 3. Test writing outside sandbox (should fail with Permission denied)
    home_dir = os.path.expanduser("~")
    test_file_outside = os.path.join(home_dir, "chatty_landlock_test_outside.txt")
    # Clean up first if it exists
    if os.path.exists(test_file_outside):
      try:
        os.unlink(test_file_outside)
      except Exception:
        pass

    cmd_args = wrap_command_with_landlock(
      binary_path,
      self.sandbox_dir,
      f"echo 'blocked' > {test_file_outside}"
    )

    proc = subprocess.run(cmd_args, capture_output=True, text=True)
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("Permission denied", proc.stderr)
    self.assertFalse(os.path.exists(test_file_outside))

  def test_chatbot_session_integration(self):
    if sys.platform != "linux":
      self.skipTest("Landlock is only supported on Linux")

    from chatty.session import ChatbotSession
    session = ChatbotSession(
      provider="ollama",
      model="mock-model",
      sandbox=self.sandbox_dir
    )

    self.assertIsNotNone(session.landlock_bin)
    self.assertTrue(os.path.exists(session.landlock_bin))

    # Test running command inside sandbox via session
    res = session.tool_run_command("echo 'hello session' > test_session.txt")
    self.assertIn("Command exited with code 0", res)

    test_file_path = os.path.join(self.sandbox_dir, "test_session.txt")
    self.assertTrue(os.path.exists(test_file_path))
    with open(test_file_path, "r") as f:
      self.assertEqual(f.read().strip(), "hello session")

    # Test reading outside sandbox via session (should succeed because of --ro /)
    res_read = session.tool_run_command("python3 -c \"print(open('/etc/passwd').read()[:50])\"")
    self.assertIn("Command exited with code 0", res_read)
    self.assertIn("root:", res_read)

    # Test writing outside sandbox via session (should fail)
    home_dir = os.path.expanduser("~")
    test_file_outside = os.path.join(home_dir, "chatty_landlock_session_outside.txt")
    if os.path.exists(test_file_outside):
      try:
        os.unlink(test_file_outside)
      except Exception:
        pass

    res_write_outside = session.tool_run_command(f"echo 'blocked' > {test_file_outside}")
    self.assertIn("Permission denied", res_write_outside)
    self.assertFalse(os.path.exists(test_file_outside))

