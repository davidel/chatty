import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.commands import cmd_exit


class TestScratchDirectory(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
    os.chdir(self.temp_dir)
    self.sandbox_dir = os.path.join(self.temp_dir, "sandbox")
    os.makedirs(self.sandbox_dir, exist_ok=True)

  def tearDown(self):
    os.chdir(self.old_cwd)

  def test_scratch_dir_created_and_advertised(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True
    )
    scratch_dir = os.path.join(self.sandbox_dir, ".chatty", "scratch")
    self.assertTrue(os.path.exists(scratch_dir))
    self.assertTrue(os.path.isdir(scratch_dir))
    self.assertIn(".chatty/scratch/", session.system_prompt)

  def test_scratch_default_wipe_on_exit(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False
    )
    scratch_file = os.path.join(self.sandbox_dir, ".chatty", "scratch", "temp.txt")
    with open(scratch_file, "w", encoding="utf-8") as f:
      f.write("temporary data")
    self.assertTrue(os.path.exists(scratch_file))

    # Test default Enter (empty string) wipes
    with patch("builtins.input", return_value=""):
      cmd_exit(session, "")

    self.assertFalse(os.path.exists(scratch_file))

  def test_scratch_preserve_when_user_declines(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False
    )
    scratch_file = os.path.join(self.sandbox_dir, ".chatty", "scratch", "keep.txt")
    with open(scratch_file, "w", encoding="utf-8") as f:
      f.write("keep this")
    self.assertTrue(os.path.exists(scratch_file))

    with patch("builtins.input", return_value="n"):
      cmd_exit(session, "")

    self.assertTrue(os.path.exists(scratch_file))


if __name__ == "__main__":
  unittest.main()
