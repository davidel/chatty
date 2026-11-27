import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.commands import cmd_exit, COMMANDS
from chatty.session_cache import (
  get_session_cache_dir,
  list_cached_sessions,
  prune_cached_sessions,
  save_cached_session,
  find_cached_session,
)
from chatty.ui import ChattyCompleter
from prompt_toolkit.document import Document


class TestSessionCache(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
    os.chdir(self.temp_dir)
    self.sandbox_dir = os.path.join(self.temp_dir, "sandbox")
    os.makedirs(self.sandbox_dir, exist_ok=True)

  def tearDown(self):
    os.chdir(self.old_cwd)

  def test_auto_save_on_cmd_exit(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True
    )
    session.messages = [
      {"role": "user", "content": "Hello there"},
      {"role": "assistant", "content": "General Kenobi"}
    ]

    with patch("builtins.input", return_value=""):
      cmd_exit(session, "")

    cache_dir = os.path.join(self.sandbox_dir, ".chatty", "cache")
    self.assertTrue(os.path.isdir(cache_dir))
    expected_file = os.path.join(cache_dir, f"session_{session.session_id}.json")
    self.assertTrue(os.path.exists(expected_file))

    with open(expected_file, "r", encoding="utf-8") as f:
      data = json.load(f)
    self.assertEqual(data["session_id"], session.session_id)
    self.assertEqual(len(data["messages"]), 2)
    self.assertEqual(data["messages"][0]["content"], "Hello there")

  def test_auto_save_on_context_manager_exit(self):
    saved_session_id = None
    with ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True
    ) as session:
      saved_session_id = session.session_id
      session.messages = [{"role": "user", "content": "Context test"}]

    cache_dir = os.path.join(self.sandbox_dir, ".chatty", "cache")
    expected_file = os.path.join(cache_dir, f"session_{saved_session_id}.json")
    self.assertTrue(os.path.exists(expected_file))

  def test_prune_keeps_last_8_sessions(self):
    cache_dir = get_session_cache_dir(self.sandbox_dir)

    # Create 12 simulated cached session files with staggered timestamps
    now = time.time()
    for i in range(12):
      session_id = f"sess_{i:02d}"
      file_path = os.path.join(cache_dir, f"session_{session_id}.json")
      data = {
        "session_id": session_id,
        "timestamp": now + i * 10,
        "provider": "ollama",
        "messages": [{"role": "user", "content": f"Message {i}"}]
      }
      with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
      # Set file mtime as well
      os.utime(file_path, (now + i * 10, now + i * 10))

    sessions = list_cached_sessions(self.sandbox_dir)
    self.assertEqual(len(sessions), 12)

    prune_cached_sessions(self.sandbox_dir, max_sessions=8)

    remaining = list_cached_sessions(self.sandbox_dir)
    self.assertEqual(len(remaining), 8)

    # Oldest 4 (sess_00 to sess_03) should be deleted
    remaining_ids = [s["id"] for s in remaining]
    for i in range(4):
      self.assertNotIn(f"sess_{i:02d}", remaining_ids)
    # Newest 8 (sess_04 to sess_11) should be present
    for i in range(4, 12):
      self.assertIn(f"sess_{i:02d}", remaining_ids)

  def test_list_sessions_command(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True
    )

    with patch("chatty.commands.console.print") as mock_print:
      COMMANDS["/list_sessions"](session, "")
      mock_print.assert_called()
      # First call when empty should notify no sessions found
      args, _ = mock_print.call_args
      self.assertIn("No cached sessions found", str(args[0]))

    # Now auto-save a session
    session.messages = [{"role": "user", "content": "List test"}]
    session.auto_save_session()

    with patch("chatty.commands.console.print") as mock_print:
      COMMANDS["/list_sessions"](session, "")
      mock_print.assert_called()
      # Should have rendered a rich table
      rendered_table = mock_print.call_args_list[0][0][0]
      self.assertTrue(hasattr(rendered_table, "columns"))
      headers = [col.header for col in rendered_table.columns]
      self.assertEqual(headers, ["ID", "Timestamp", "Size"])

  def test_load_session_by_id(self):
    session1 = ChatbotSession(
      provider="ollama",
      model="initial-model",
      sandbox=self.sandbox_dir,
      headless=True
    )
    session1.messages = [{"role": "user", "content": "Restore me later"}]
    session1.system_prompt = "Special Prompt"
    saved_path = session1.auto_save_session()
    self.assertIsNotNone(saved_path)
    s_id = session1.session_id

    # Create session 2 and load by ID
    session2 = ChatbotSession(
      provider="openrouter",
      model="other-model",
      sandbox=self.sandbox_dir,
      headless=True
    )
    self.assertEqual(session2.provider, "openrouter")
    self.assertEqual(len(session2.messages), 0)

    # Load using ID
    res = session2.handle_command(f"/load_session {s_id}")
    self.assertTrue(res)
    self.assertEqual(session2.session_id, s_id)
    self.assertEqual(len(session2.messages), 1)
    self.assertEqual(session2.messages[0]["content"], "Restore me later")
    self.assertEqual(session2.system_prompt, "Special Prompt")

    # Load using session_<ID>
    session3 = ChatbotSession(
      provider="openrouter",
      model="clean-slate",
      sandbox=self.sandbox_dir,
      headless=True
    )
    res3 = session3.handle_command(f"/load_session session_{s_id}")
    self.assertTrue(res3)
    self.assertEqual(len(session3.messages), 1)
    self.assertEqual(session3.messages[0]["content"], "Restore me later")

  def test_autocomplete_ids_and_paths(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True
    )
    session.messages = [{"role": "user", "content": "Autocomplete test"}]
    session.auto_save_session()
    s_id = session.session_id

    # Also create a dummy local file to test path completion
    dummy_file = os.path.join(self.sandbox_dir, "my_saved_session.json")
    with open(dummy_file, "w", encoding="utf-8") as f:
      f.write("{}")

    completer = ChattyCompleter(session._commands.keys(), session=session)

    # Autocomplete after "/load_session "
    doc = Document("/load_session ", cursor_position=len("/load_session "))
    completions = list(completer.get_completions(doc, MagicMock()))
    completion_texts = [c.text for c in completions]

    self.assertIn(s_id, completion_texts)
    self.assertTrue(any("my_saved_session.json" in t for t in completion_texts))

    # Autocomplete with partial ID prefix
    doc_prefix = Document(f"/load_session {s_id[:4]}", cursor_position=len(f"/load_session {s_id[:4]}"))
    completions_prefix = list(completer.get_completions(doc_prefix, MagicMock()))
    prefix_texts = [c.text for c in completions_prefix]
    self.assertIn(s_id, prefix_texts)


if __name__ == "__main__":
  unittest.main()
