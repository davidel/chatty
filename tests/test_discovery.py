import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.discovery import (
  DISCOVERY_TOOL_NAMES,
  extract_session_context,
  build_discovery_system_prompt,
  run_discovery
)
from chatty.tools.registry import handle_discover_context, TOOL_REGISTRY
from chatty.commands import cmd_discover, cmd_discovery_model
from chatty.ui import ChattyCompleter
from prompt_toolkit.document import Document


class TestDiscoveryAgent(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = tempfile.mkdtemp()
    self.session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True,
      discovery_model="test-scout",
      discovery_loops=5
    )

  def tearDown(self):
    os.chdir(self.old_cwd)
    shutil.rmtree(self.sandbox_dir)

  def test_discovery_tool_restrictions(self):
    # Only read-only tools should be allowed
    self.assertIn("read_file", DISCOVERY_TOOL_NAMES)
    self.assertIn("search_grep", DISCOVERY_TOOL_NAMES)
    self.assertIn("locate_files", DISCOVERY_TOOL_NAMES)
    self.assertIn("get_outline", DISCOVERY_TOOL_NAMES)
    self.assertIn("find_symbol", DISCOVERY_TOOL_NAMES)

    # Prohibited write / execution tools
    self.assertNotIn("write_file", DISCOVERY_TOOL_NAMES)
    self.assertNotIn("patch_file", DISCOVERY_TOOL_NAMES)
    self.assertNotIn("run_command", DISCOVERY_TOOL_NAMES)
    self.assertNotIn("delete_file", DISCOVERY_TOOL_NAMES)
    self.assertNotIn("discover_context", DISCOVERY_TOOL_NAMES)

  def test_extract_session_context_empty(self):
    self.session.messages.clear()
    context = extract_session_context(self.session)
    self.assertEqual(context, "")

  def test_extract_session_context_with_history(self):
    self.session.messages.append({
      "role": "user",
      "content": "Please inspect the authentication module."
    })
    self.session.messages.append({
      "role": "assistant",
      "content": "Looking at auth.py",
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "read_file",
          "arguments": '{"path": "src/auth.py"}'
        }
      }]
    })

    context = extract_session_context(self.session)
    self.assertIn("## Existing Session Context", context)
    self.assertIn("src/auth.py", context)
    self.assertIn("Please inspect the authentication module.", context)

  def test_build_discovery_system_prompt(self):
    task = "Find where user tokens are verified"
    prompt = build_discovery_system_prompt(self.session, task)
    self.assertIn("Workspace Discovery & Reconnaissance Agent", prompt)
    self.assertIn(task, prompt)
    self.assertIn("DO NOT write code", prompt)
    self.assertIn("### Target Files & Line Ranges", prompt)
    self.assertIn("### Key Symbols & Definitions", prompt)

  def test_discover_context_tool_registry(self):
    self.assertIn("discover_context", TOOL_REGISTRY)

    err = handle_discover_context({}, self.session)
    self.assertIn("Missing parameter 'task'", err)

  @patch("chatty.discovery.run_discovery")
  def test_handle_discover_context(self, mock_run):
    mock_run.return_value = "### Discovered Context\n- file.py: L1-L10"
    res = handle_discover_context({"task": "investigate login"}, self.session)
    self.assertEqual(res, "### Discovered Context\n- file.py: L1-L10")
    mock_run.assert_called_once_with(self.session, "investigate login")

  def test_cmd_discovery_model(self):
    # View model
    res = cmd_discovery_model(self.session, "")
    self.assertTrue(res)

    # Switch model
    res = cmd_discovery_model(self.session, "fast-scout-model")
    self.assertTrue(res)
    self.assertEqual(self.session.get_discovery_model(), "fast-scout-model")

  @patch("chatty.discovery.run_discovery")
  def test_cmd_discover(self, mock_run):
    # Empty query
    res = cmd_discover(self.session, "")
    self.assertTrue(res)

    # Valid query
    mock_run.return_value = "### Target Files\n- `main.py`: L1-L20"
    init_msg_count = len(self.session.messages)
    res = cmd_discover(self.session, "check startup sequence")
    self.assertTrue(res)
    mock_run.assert_called_once_with(self.session, "check startup sequence")

    # Context was injected into messages
    self.assertEqual(len(self.session.messages), init_msg_count + 2)
    self.assertEqual(self.session.messages[-2]["role"], "user")
    self.assertIn("check startup sequence", self.session.messages[-2]["content"])
    self.assertIn("main.py", self.session.messages[-2]["content"])

  def test_chatty_completer_discovery_model(self):
    self.session.available_models = [
      {"id": "qwen2.5-coder:7b"},
      {"id": "qwen2.5-coder:32b"},
      {"id": "gpt-4o"}
    ]
    completer = ChattyCompleter(["/discovery_model", "/discover_model", "/oracle", "/model"], session=self.session)

    # Test /discovery_model completion
    doc = Document("/discovery_model qwen")
    completions = list(completer.get_completions(doc, None))
    comp_texts = [c.text for c in completions]
    self.assertIn("qwen2.5-coder:7b", comp_texts)
    self.assertIn("qwen2.5-coder:32b", comp_texts)
    self.assertNotIn("gpt-4o", comp_texts)

    # Test /discover_model completion
    doc2 = Document("/discover_model gp")
    completions2 = list(completer.get_completions(doc2, None))
    comp_texts2 = [c.text for c in completions2]
    self.assertIn("gpt-4o", comp_texts2)

  def test_session_state_persistence(self):
    self.session.discovery_model = "scout-persisted"
    self.session.discovery_loops = 25
    save_path = os.path.join(self.sandbox_dir, "session_dump.json")
    self.session.save_session(save_path)

    new_session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True
    )
    new_session.load_session(save_path)
    self.assertEqual(new_session.get_discovery_model(), "scout-persisted")
    self.assertEqual(new_session.discovery_loops, 25)

  def test_openrouter_provider_dynamic_discovery_model(self):
    from chatty.providers import OpenRouterProvider
    prov = OpenRouterProvider()

    mock_models = [
      {
        "id": "expensive/model",
        "pricing_input": 5.0,
        "pricing_output": 15.0,
        "supported_parameters": ["tools"]
      },
      {
        "id": "cheap/general-chat",
        "name": "General Chat",
        "description": "A general chat bot",
        "pricing_input": 0.05,
        "pricing_output": 0.10,
        "supported_parameters": ["tools"]
      },
      {
        "id": "cheap/fast-coder",
        "name": "Fast Coder",
        "description": "A fast programming model",
        "pricing_input": 0.08,
        "pricing_output": 0.15,
        "supported_parameters": ["tools"]
      }
    ]

    with patch.object(prov, "fetch_models", return_value=mock_models):
      chosen = prov.get_default_discovery_model()
      # Should prefer the coding model within price bounds
      self.assertEqual(chosen, "cheap/fast-coder")

  def test_ollama_provider_discovery_model(self):
    from chatty.providers import OllamaProvider
    prov = OllamaProvider()

    mock_models = [
      {"id": "llama3:8b", "name": "llama3"},
      {"id": "qwen2.5-coder:7b", "name": "qwen2.5-coder"}
    ]

    with patch.object(prov, "fetch_models", return_value=mock_models):
      chosen = prov.get_default_discovery_model()
      self.assertEqual(chosen, "qwen2.5-coder:7b")


if __name__ == "__main__":
  unittest.main()
