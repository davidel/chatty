import os
import shutil
import tempfile
import unittest
import sys
from unittest.mock import patch, MagicMock

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession, optional_live


class TestHeadlessMode(unittest.TestCase):

  def setUp(self):
    self.sandbox_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.sandbox_dir)

  def test_headless_config(self):
    session = ChatbotSession(
      provider="ollama",
      model="qwen2.5-coder:7b",
      sandbox=self.sandbox_dir,
      headless=True
    )
    self.assertTrue(session.headless)

  def test_headless_start_loop_raises(self):
    session = ChatbotSession(
      provider="ollama",
      model="qwen2.5-coder:7b",
      sandbox=self.sandbox_dir,
      headless=True
    )
    with self.assertRaises(RuntimeError):
      session.start_loop()

  def test_optional_live_headless(self):
    mock_console = MagicMock()
    with optional_live("dummy", mock_console, enabled=False) as live:
      live.update("new renderable")
    # Verify that rich's Live was not initialized/called (since enabled=False)
    mock_console.print.assert_not_called()


if __name__ == "__main__":
  unittest.main()
