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

  def test_oracle_model_no_fallback(self):
    session2 = ChatbotSession(
      provider="ollama",
      model="test-model",
      models=["test-model", "alternative-model"],
      sandbox=self.sandbox_dir
    )
    self.assertIsNone(session2.get_oracle_model())
    session3 = ChatbotSession(
      provider="openrouter",
      model="test-model",
      models=["test-model"],
      sandbox=self.sandbox_dir
    )
    self.assertIsNone(session3.get_oracle_model())

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
    # Verify it is in active session tools
    active_tools = self.session.get_tools()
    tool_names = [t["function"]["name"] for t in active_tools]
    self.assertIn("ask_oracle", tool_names)
    # Execute tool
    res = execute_tool("ask_oracle", {"query": "Explain quantum physics"}, self.session)
    self.assertEqual(res, "Oracle suggestion")

  def test_ask_oracle_tool_omitted_when_not_configured(self):
    session2 = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir
    )
    active_tools = session2.get_tools()
    tool_names = [t["function"]["name"] for t in active_tools]
    self.assertNotIn("ask_oracle", tool_names)

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

    # Test with both variant suffix and provider suffix
    model, extra = self.session._resolve_model_and_provider("xiaomi/mimo-v2.5:nitro:together")
    self.assertEqual(model, "xiaomi/mimo-v2.5:nitro")
    self.assertEqual(extra, {
      "provider": {
        "order": ["together"],
        "allow_fallbacks": False
      }
    })

  def test_is_retryable_exception(self):
    import openai
    # Construct mock/real APIStatusError
    mock_response = MagicMock()
    mock_response.status_code = 400
    
    # 1. Test "provider returned error" and rate limit indicators
    err_msg = "Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '...'}}}"
    exc = openai.APIStatusError(message=err_msg, response=mock_response, body=None)
    self.assertTrue(self.session._is_retryable_exception(exc))
    
    # 2. Test "rate-limited" indicator
    err_msg2 = "xiaomi/mimo-v2.5 is temporarily rate-limited upstream"
    exc2 = openai.APIStatusError(message=err_msg2, response=mock_response, body=None)
    self.assertTrue(self.session._is_retryable_exception(exc2))

    # 3. Test non-retryable 400 error
    err_msg3 = "Invalid request parameter: max_tokens"
    exc3 = openai.APIStatusError(message=err_msg3, response=mock_response, body=None)
    self.assertFalse(self.session._is_retryable_exception(exc3))

  @patch("chatty.session.openai.OpenAI")
  def test_ask_oracle_with_file_expansion(self, mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    self.session.client = mock_client
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="Oracle suggestion"))]
    mock_client.chat.completions.create.return_value = [chunk]

    # Create dummy files in the sandbox
    file_path = os.path.join(self.sandbox_dir, "hello.txt")
    with open(file_path, "w", encoding="utf-8") as f:
      f.write("Hello, World!")

    other_file_path = os.path.join(self.sandbox_dir, "test.py")
    with open(other_file_path, "w", encoding="utf-8") as f:
      f.write("print('test')")

    # Execute consult_oracle with @hello.txt and @test.py and check substitution
    query = "Check this @hello.txt and also @test.py. What about @nonexistent.txt or non-file @someone?"
    res = self.session.consult_oracle(query)

    self.assertEqual(res, "Oracle suggestion")
    mock_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_client.chat.completions.create.call_args
    sent_query = kwargs["messages"][1]["content"]

    self.assertIn("--- START OF FILE hello.txt ---", sent_query)
    self.assertIn("Hello, World!", sent_query)
    self.assertIn("--- END OF FILE hello.txt ---", sent_query)
    self.assertIn("--- START OF FILE test.py ---", sent_query)
    self.assertIn("print('test')", sent_query)
    self.assertIn("--- END OF FILE test.py ---", sent_query)
    self.assertIn("@nonexistent.txt", sent_query)
    self.assertNotIn("--- START OF FILE nonexistent.txt ---", sent_query)
    self.assertIn("@someone", sent_query)

  @patch("chatty.session.openai.OpenAI")
  def test_ask_oracle_with_file_expansion_outside_sandbox(self, mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    self.session.client = mock_client
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="Oracle suggestion"))]
    mock_client.chat.completions.create.return_value = [chunk]

    # Reference file outside sandbox, e.g. ../outside.txt
    query = "Check @../outside.txt"
    res = self.session.consult_oracle(query)

    self.assertEqual(res, "Oracle suggestion")
    mock_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_client.chat.completions.create.call_args
    sent_query = kwargs["messages"][1]["content"]

    self.assertIn("[Error reading file ../outside.txt: Access Denied:", sent_query)
