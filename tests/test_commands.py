import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.commands import COMMANDS, cmd_multiline, cmd_provider, cmd_model, cmd_undo, cmd_pop
from chatty.utils import repair_json


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
    self.assertIn("/undo", COMMANDS)
    self.assertIn("/pop", COMMANDS)

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

  def test_cmd_undo(self):
    self.session.messages = [
      {"role": "user", "content": "hello"},
      {"role": "assistant", "content": "hi", "tool_calls": []},
      {"role": "tool", "content": "result"},
      {"role": "user", "content": "prompt2"},
    ]
    # Popping prompt2 (count=1)
    res = cmd_undo(self.session, "")
    self.assertTrue(res)
    self.assertEqual(len(self.session.messages), 3)
    self.assertEqual(self.session.messages[0]["content"], "hello")

    # Popping the previous turn (hello, hi, result)
    res2 = cmd_undo(self.session, "")
    self.assertTrue(res2)
    self.assertEqual(len(self.session.messages), 0)

  def test_cmd_pop(self):
    self.session.messages = [
      {"role": "user", "content": "m1"},
      {"role": "assistant", "content": "m2"},
      {"role": "user", "content": "m3"},
    ]
    res = cmd_pop(self.session, "2")
    self.assertTrue(res)
    self.assertEqual(len(self.session.messages), 1)
    self.assertEqual(self.session.messages[0]["content"], "m1")

  def test_repair_json(self):
    # Truncated JSON
    self.assertEqual(repair_json('{"key": "val'), '{"key": "val"}')
    # Single quotes
    self.assertEqual(repair_json("{'key': 'val'}"), '{"key": "val"}')
    # Trailing comma
    self.assertEqual(repair_json('{"key": "val",}'), '{"key": "val"}')


if __name__ == "__main__":
  unittest.main()
