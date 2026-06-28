import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.commands import COMMANDS, cmd_multiline, cmd_provider, cmd_model


class TestCommandsRegistry(unittest.TestCase):

  def setUp(self):
    self.sandbox_dir = tempfile.mkdtemp()
    self.session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir
    )

  def tearDown(self):
    shutil.rmtree(self.sandbox_dir)

  def test_registry_contains_commands(self):
    self.assertIn("/exit", COMMANDS)
    self.assertIn("/status", COMMANDS)
    self.assertIn("/provider", COMMANDS)
    self.assertIn("/model", COMMANDS)

  def test_cmd_multiline(self):
    self.assertFalse(self.session.multiline_mode)
    res = cmd_multiline(self.session, "")
    self.assertTrue(res)
    self.assertTrue(self.session.multiline_mode)
    cmd_multiline(self.session, "")
    self.assertFalse(self.session.multiline_mode)

  def test_cmd_provider_and_model(self):
    self.assertEqual(self.session.provider, "ollama")
    res = cmd_provider(self.session, "openrouter")
    self.assertTrue(res)
    self.assertEqual(self.session.provider, "openrouter")

    self.assertEqual(self.session.model, "test-model")
    res = cmd_model(self.session, "custom-model")
    self.assertTrue(res)
    self.assertEqual(self.session.model, "custom-model")


if __name__ == "__main__":
  unittest.main()
