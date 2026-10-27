import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from chatty.profiles import get_profile, AgentProfile, CODING_TOOLS
from chatty.session import ChatbotSession, SessionConfig
from chatty.tools import execute_tool
from chatty.commands import cmd_config
from chatty.providers import OllamaProvider, OpenRouterProvider


class TestAgentProfiles(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = self.enterContext(tempfile.TemporaryDirectory())

  def tearDown(self):
    os.chdir(self.old_cwd)

  def test_get_profile_lookup(self):
    coder = get_profile("coder")
    self.assertEqual(coder.name, "coder")
    self.assertTrue(coder.enable_repo_map)
    self.assertTrue(coder.prefer_coding_models)
    self.assertEqual(len(coder.excluded_tools), 0)

    general = get_profile("general")
    self.assertEqual(general.name, "general")
    self.assertFalse(general.enable_repo_map)
    self.assertFalse(general.prefer_coding_models)
    self.assertIn("patch_file", general.excluded_tools)
    self.assertIn("format_file", general.excluded_tools)
    self.assertIn("run_tests", general.excluded_tools)
    self.assertIn("get_outline", general.excluded_tools)
    self.assertIn("find_symbol", general.excluded_tools)
    self.assertIn("discover_context", general.excluded_tools)

    # Unknown profile falls back to coder
    fallback = get_profile("nonexistent")
    self.assertEqual(fallback.name, "coder")

    # None / empty falls back to coder
    fallback_none = get_profile(None)
    self.assertEqual(fallback_none.name, "coder")

  def test_coder_profile_session_defaults(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True,
      profile="coder"
    )
    self.assertEqual(session.agent_profile.name, "coder")
    self.assertTrue(session.repo_map)
    tool_names = [t["function"]["name"] for t in session.get_tools()]
    self.assertIn("patch_file", tool_names)
    self.assertIn("format_file", tool_names)
    self.assertIn("find_symbol", tool_names)
    self.assertIn("read_file", tool_names)
    self.assertIn("run_command", tool_names)
    self.assertIn("patch_file", session.system_prompt)

  def test_general_profile_session_defaults(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True,
      profile="general"
    )
    self.assertEqual(session.agent_profile.name, "general")
    self.assertFalse(session.repo_map)
    tool_names = [t["function"]["name"] for t in session.get_tools()]
    self.assertNotIn("patch_file", tool_names)
    self.assertNotIn("format_file", tool_names)
    self.assertNotIn("run_tests", tool_names)
    self.assertNotIn("get_outline", tool_names)
    self.assertNotIn("find_symbol", tool_names)
    self.assertNotIn("discover_context", tool_names)

    # Generic tools are preserved
    self.assertIn("read_file", tool_names)
    self.assertIn("write_file", tool_names)
    self.assertIn("run_command", tool_names)
    self.assertIn("web_research", tool_names)
    self.assertIn("ask_question", tool_names)

    # System prompt does not reference patch_file or format_file
    self.assertNotIn("patch_file", session.system_prompt)
    self.assertNotIn("format_file", session.system_prompt)
    self.assertNotIn("find_symbol", session.system_prompt)

  def test_execute_tool_blocks_disabled_profile_tools(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True,
      profile="general"
    )
    res = execute_tool("patch_file", {"path": "test.txt", "search": "a", "replace": "b"}, session)
    self.assertIn("Error: Tool 'patch_file' is disabled for the 'general' profile.", res)

  def test_explicit_repo_map_override(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True,
      profile="general",
      repo_map=True
    )
    self.assertTrue(session.repo_map)

    session2 = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True,
      profile="coder",
      repo_map=False
    )
    self.assertFalse(session2.repo_map)

  def test_cmd_config_switch_profile(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True,
      profile="coder"
    )
    self.assertEqual(session.agent_profile.name, "coder")

    res = cmd_config(session, "profile=general")
    self.assertTrue(res)
    self.assertEqual(session.agent_profile.name, "general")
    self.assertEqual(session.profile, "general")
    tool_names = [t["function"]["name"] for t in session.get_tools()]
    self.assertNotIn("patch_file", tool_names)

  def test_provider_model_selection_by_profile(self):
    prov = OllamaProvider()
    self.assertEqual(prov.get_default_model(profile="coder"), "qwen2.5-coder:7b")
    self.assertEqual(prov.get_default_model(profile="general"), "llama3.2:3b")

    mock_models = [
      {"id": "deepseek/deepseek-chat", "pricing_input": 0.05, "pricing_output": 0.10, "supported_parameters": ["tools"]},
      {"id": "qwen/qwen-2.5-coder-32b", "pricing_input": 0.05, "pricing_output": 0.10, "supported_parameters": ["tools"]}
    ]
    or_prov = OpenRouterProvider()
    with patch.object(or_prov, "fetch_models", return_value=mock_models):
      chosen_coder = or_prov.get_default_discovery_model(profile="coder")
      self.assertEqual(chosen_coder, "qwen/qwen-2.5-coder-32b")

      chosen_general = or_prov.get_default_discovery_model(profile="general")
      self.assertEqual(chosen_general, "deepseek/deepseek-chat")


if __name__ == "__main__":
  unittest.main()
