import os
import shutil
import tempfile
import unittest
import sys
from unittest.mock import patch, MagicMock

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.tools.file_ops import tool_write_file, tool_patch_file
from chatty.tools.registry import handle_write_file, handle_patch_file
from chatty.commands import cmd_auto_approve, cmd_config


class TestAutoApprove(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = self.enterContext(tempfile.TemporaryDirectory())

  def tearDown(self):
    os.chdir(self.old_cwd)

  def test_confirm_file_change_auto_approve_enabled(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=True
    )
    with patch("builtins.input") as mock_input:
      approved, feedback = session.confirm_file_change("foo.txt", "old", "new")
      mock_input.assert_not_called()
      self.assertTrue(approved)
      self.assertIsNone(feedback)

  def test_confirm_file_change_headless_auto_approves(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True,
      auto_approve=False
    )
    # In headless mode, auto_approve should be forced to True
    self.assertTrue(session.auto_approve)
    with patch("builtins.input") as mock_input:
      approved, feedback = session.confirm_file_change("foo.txt", "old", "new")
      mock_input.assert_not_called()
      self.assertTrue(approved)
      self.assertIsNone(feedback)

  def test_confirm_file_change_identical_content(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    with patch("builtins.input") as mock_input:
      approved, feedback = session.confirm_file_change("foo.txt", "content", "content")
      mock_input.assert_not_called()
      self.assertTrue(approved)
      self.assertIsNone(feedback)

  def test_confirm_file_change_user_approves(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    with patch("builtins.input", side_effect=["y"]):
      approved, feedback = session.confirm_file_change("foo.txt", "old", "new")
      self.assertTrue(approved)
      self.assertIsNone(feedback)

    with patch("builtins.input", side_effect=[""]):
      approved, feedback = session.confirm_file_change("foo.txt", "old", "new")
      self.assertTrue(approved)
      self.assertIsNone(feedback)

  def test_confirm_file_change_user_rejects(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    with patch("builtins.input", side_effect=["n"]):
      approved, feedback = session.confirm_file_change("foo.txt", "old", "new")
      self.assertFalse(approved)
      self.assertIsNone(feedback)

  def test_confirm_file_change_user_comment(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    with patch("builtins.input", side_effect=["c", "Please keep variable x as int"]):
      approved, feedback = session.confirm_file_change("foo.txt", "old", "new")
      self.assertFalse(approved)
      self.assertEqual(feedback, "Please keep variable x as int")

  def test_confirm_file_change_user_toggles_all(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    self.assertFalse(session.auto_approve)
    with patch("builtins.input", side_effect=["a"]):
      approved, feedback = session.confirm_file_change("foo.txt", "old", "new")
      self.assertTrue(approved)
      self.assertIsNone(feedback)
      self.assertTrue(session.auto_approve)

    # Next confirmation should auto-approve without calling input
    with patch("builtins.input") as mock_input:
      approved, feedback = session.confirm_file_change("bar.txt", "old", "new")
      mock_input.assert_not_called()
      self.assertTrue(approved)
      self.assertIsNone(feedback)

  def test_tool_write_file_interactive_approval(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    file_path = "greeting.txt"

    # 1. User rejects write
    with patch("builtins.input", side_effect=["n"]):
      res = tool_write_file(self.sandbox_dir, file_path, "Hello World", session=session)
      self.assertIn("Change rejected by user for file 'greeting.txt'", res)
      self.assertFalse(os.path.exists(os.path.join(self.sandbox_dir, file_path)))

    # 2. User rejects with instructions
    with patch("builtins.input", side_effect=["c", "Write in French instead"]):
      res = tool_write_file(self.sandbox_dir, file_path, "Hello World", session=session)
      self.assertIn("Change rejected by user with instructions: Write in French instead", res)
      self.assertFalse(os.path.exists(os.path.join(self.sandbox_dir, file_path)))

    # 3. User approves write
    with patch("builtins.input", side_effect=["y"]):
      res = tool_write_file(self.sandbox_dir, file_path, "Bonjour le monde", session=session)
      self.assertIn("Successfully wrote to file 'greeting.txt'", res)
      abs_p = os.path.join(self.sandbox_dir, file_path)
      self.assertTrue(os.path.exists(abs_p))
      with open(abs_p, "r", encoding="utf-8") as f:
        self.assertEqual(f.read(), "Bonjour le monde")

  def test_tool_patch_file_interactive_approval(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    file_path = "code.py"
    abs_p = os.path.join(self.sandbox_dir, file_path)
    with open(abs_p, "w", encoding="utf-8") as f:
      f.write("def add(a, b):\n  return a - b\n")

    patch_text = "<<<<<<< SEARCH\n  return a - b\n=======\n  return a + b\n>>>>>>> REPLACE"

    # 1. User rejects patch
    with patch("builtins.input", side_effect=["n"]):
      res = tool_patch_file(self.sandbox_dir, file_path, patch_text, session=session)
      self.assertIn("Change rejected by user for file 'code.py'", res)
      with open(abs_p, "r", encoding="utf-8") as f:
        self.assertEqual(f.read(), "def add(a, b):\n  return a - b\n")

    # 2. User approves patch
    with patch("builtins.input", side_effect=["y"]):
      res = tool_patch_file(self.sandbox_dir, file_path, patch_text, session=session)
      self.assertIn("Successfully updated file 'code.py'", res)
      with open(abs_p, "r", encoding="utf-8") as f:
        self.assertEqual(f.read(), "def add(a, b):\n  return a + b\n")

  def test_dry_run_patch_skips_approval(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    file_path = "dry.py"
    abs_p = os.path.join(self.sandbox_dir, file_path)
    with open(abs_p, "w", encoding="utf-8") as f:
      f.write("val = 1\n")

    patch_text = "<<<<<<< SEARCH\nval = 1\n=======\nval = 2\n>>>>>>> REPLACE"

    with patch("builtins.input") as mock_input:
      res = tool_patch_file(self.sandbox_dir, file_path, patch_text, dry_run=True, session=session)
      mock_input.assert_not_called()
      self.assertIn("[DRY RUN]", res)
      with open(abs_p, "r", encoding="utf-8") as f:
        self.assertEqual(f.read(), "val = 1\n")

  def test_handle_registry_dispatch_with_session(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    with patch("builtins.input", side_effect=["n"]):
      res = handle_write_file({"path": "reg.txt", "content": "data"}, session=session)
      self.assertIn("Change rejected by user", res)

  def test_cmd_auto_approve_toggle_and_set(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    self.assertFalse(session.auto_approve)

    # Toggle with no args -> True
    cmd_auto_approve(session, "")
    self.assertTrue(session.auto_approve)
    self.assertTrue(session.config.auto_approve)

    # Toggle with no args -> False
    cmd_auto_approve(session, "")
    self.assertFalse(session.auto_approve)
    self.assertFalse(session.config.auto_approve)

    # Explicit on
    cmd_auto_approve(session, "on")
    self.assertTrue(session.auto_approve)

    # Explicit off
    cmd_auto_approve(session, "off")
    self.assertFalse(session.auto_approve)

  def test_cmd_config_auto_approve(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=False,
      auto_approve=False
    )
    self.assertFalse(session.auto_approve)

    cmd_config(session, "auto_approve=true")
    self.assertTrue(session.auto_approve)
    self.assertTrue(session.config.auto_approve)

    cmd_config(session, "auto_approve=false")
    self.assertFalse(session.auto_approve)
    self.assertFalse(session.config.auto_approve)


if __name__ == "__main__":
  unittest.main()
