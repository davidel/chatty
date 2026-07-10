import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.tools import execute_tool, TOOLS_SCHEMA
from chatty.commands import COMMANDS


class TestOracle(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = tempfile.mkdtemp()
    self.session = ChatbotSession(
      provider="ollama",
      model="test-model",
      models=["test-model", "alternative-model"],
      oracle_model="custom-oracle",
      sandbox=self.sandbox_dir
    )

  def tearDown(self):
    os.chdir(self.old_cwd)
    shutil.rmtree(self.sandbox_dir)

  def test_oracle_model_config(self):
    self.assertEqual(self.session.oracle_model, "custom-oracle")
    self.assertEqual(self.session.get_oracle_model(), "custom-oracle")

  def test_oracle_model_fallback_ollama(self):
    session2 = ChatbotSession(
      provider="ollama",
      model="test-model",
      models=["test-model", "alternative-model"],
      sandbox=self.sandbox_dir
    )
    self.assertEqual(session2.get_oracle_model(), "alternative-model")
    session3 = ChatbotSession(
      provider="ollama",
      model="test-model",
      models=["test-model"],
      sandbox=self.sandbox_dir
    )
    self.assertEqual(session3.get_oracle_model(), "test-model")

  def test_oracle_model_fallback_openrouter(self):
    session2 = ChatbotSession(
      provider="openrouter",
      model="test-model",
      models=["test-model"],
      sandbox=self.sandbox_dir
    )
    self.assertEqual(session2.get_oracle_model(), "google/gemini-2.5-pro")

  def test_oracle_command(self):
    # View oracle command
    res = self.session.handle_command("/oracle")
    self.assertTrue(res)
    # Change oracle command
    res = self.session.handle_command("/oracle new-oracle-model")
    self.assertTrue(res)
    self.assertEqual(self.session.oracle_model, "new-oracle-model")

  def test_oracle_session_save_load(self):
    self.session.oracle_model = "saved-oracle"
    save_path = os.path.join(self.sandbox_dir, "saved_session.json")
    res = self.session.handle_command(f"/save_session {save_path}")
    self.assertTrue(res)
    session2 = ChatbotSession(
      provider="ollama",
      model="other",
      sandbox=self.sandbox_dir
    )
    res = session2.handle_command(f"/load_session {save_path}")
    self.assertTrue(res)
    self.assertEqual(session2.oracle_model, "saved-oracle")

  @patch("chatty.session.openai.OpenAI")
  def test_consult_oracle_streaming(self, mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    self.session.client = mock_client
    # Mock chunk response
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="world!"))]
    mock_client.chat.completions.create.return_value = [chunk1, chunk2]
    # Verify consult_oracle
    res = self.session.consult_oracle("Test query")
    self.assertEqual(res, "Hello world!")
    mock_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_client.chat.completions.create.call_args
    self.assertEqual(kwargs["model"], "custom-oracle")
    self.assertEqual(kwargs["messages"][1]["content"], "Test query")

  @patch("chatty.session.openai.OpenAI")
  def test_ask_oracle_tool(self, mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    self.session.client = mock_client
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="Oracle suggestion"))]
    mock_client.chat.completions.create.return_value = [chunk]
    # Find tool schema
    tool_names = [t["function"]["name"] for t in TOOLS_SCHEMA]
    self.assertIn("ask_oracle", tool_names)
    # Execute tool
    res = execute_tool("ask_oracle", {"query": "Explain quantum physics"}, self.session)
    self.assertEqual(res, "Oracle suggestion")

  def test_model_and_provider_resolution(self):
    # Test without colon
    model, extra = self.session._resolve_model_and_provider("xiaomi/mimo-v2.5")
    self.assertEqual(model, "xiaomi/mimo-v2.5")
    self.assertIsNone(extra)

    # Test with standard suffix (nitro)
    model, extra = self.session._resolve_model_and_provider("xiaomi/mimo-v2.5:nitro")
    self.assertEqual(model, "xiaomi/mimo-v2.5:nitro")
    self.assertIsNone(extra)

    # Test with provider suffix (xiaomi)
    model, extra = self.session._resolve_model_and_provider("xiaomi/mimo-v2.5:xiaomi")
    self.assertEqual(model, "xiaomi/mimo-v2.5")
    self.assertEqual(extra, {
      "provider": {
        "order": ["xiaomi"],
        "allow_fallbacks": False
      }
    })
