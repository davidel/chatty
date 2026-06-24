import json
import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession


class TestSessionPersist(unittest.TestCase):

  def setUp(self):
    self.sandbox_dir = tempfile.mkdtemp()
    self.save_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.sandbox_dir)
    shutil.rmtree(self.save_dir)

  def test_save_and_load_session(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      context_size=8000,
      sandbox=self.sandbox_dir,
      max_loops=15
    )
    
    session.system_prompt = "Custom system instructions"
    session.messages = [
      {"role": "user", "content": "Hello agent"},
      {"role": "assistant", "content": "Hello user"}
    ]
    session.tool_calls_count = {"read_file": 3}
    session.external_binaries_count = 5
    session.external_binaries_breakdown = {"gcc": 5}
    session.api_key = "some_secret_key"
    session.url = "http://my-url:11434"
    
    # Save the session to a temporary file
    save_path = os.path.join(self.save_dir, "test_session.json")
    save_result = session.handle_command(f"/save_session {save_path}")
    self.assertTrue(save_result)
    
    # Check that file exists and holds details
    self.assertTrue(os.path.exists(save_path))
    with open(save_path, "r", encoding="utf-8") as f:
      data = json.load(f)
    self.assertEqual(data["provider"], "ollama")
    self.assertEqual(data["model"], "test-model")
    self.assertEqual(data["context_size"], 8000)
    self.assertEqual(data["max_loops"], 15)
    self.assertEqual(data["system_prompt"], "Custom system instructions")
    self.assertEqual(len(data["messages"]), 2)
    self.assertEqual(data["tool_calls_count"]["read_file"], 3)
    self.assertEqual(data["api_key"], "some_secret_key")
    self.assertEqual(data["url"], "http://my-url:11434")
    
    # Re-instantiate session to a clean slate
    session2 = ChatbotSession(
      provider="openrouter",
      model="different-model",
      context_size=5000,
      sandbox=self.sandbox_dir
    )
    self.assertEqual(session2.provider, "openrouter")
    self.assertEqual(session2.model, "different-model")
    
    # Load session back
    load_result = session2.handle_command(f"/load_session {save_path}")
    self.assertTrue(load_result)
    
    # Verify values are restored
    self.assertEqual(session2.provider, "ollama")
    self.assertEqual(session2.model, "test-model")
    self.assertEqual(session2.context_size, 8000)
    self.assertEqual(session2.max_loops, 15)
    self.assertEqual(session2.system_prompt, "Custom system instructions")
    self.assertEqual(len(session2.messages), 2)
    self.assertEqual(session2.messages[0]["content"], "Hello agent")
    self.assertEqual(session2.tool_calls_count["read_file"], 3)
    self.assertEqual(session2.external_binaries_count, 5)
    self.assertEqual(session2.external_binaries_breakdown["gcc"], 5)
    self.assertEqual(session2.api_key, "some_secret_key")
    self.assertEqual(session2.url, "http://my-url:11434")
