import logging
import os
import re
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.logging_setup import GlogFormatter


class TestGlogFormatter(unittest.TestCase):

  def test_formatter_layout(self):
    # Setup standard logger and a custom handler with GlogFormatter
    logger = logging.getLogger("test_glog")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Use StringIO to capture logs in-memory
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(GlogFormatter())
    logger.addHandler(handler)

    try:
      logger.info("This is an info message.")
      log_output = stream.getvalue().strip()

      # Log format: Lyyyymmdd hh:mm:ss.uuuuuu process file:line] msg
      # e.g., I20260621 08:54:00.123456 12345 test_logging.py:24] This is an info message.
      pattern = r"^I\d{8} \d{2}:\d{2}:\d{2}\.\d{6} \d+ test_logging\.py:\d+\] This is an info message\.$"
      self.assertTrue(
        re.match(pattern, log_output),
        f"Log output '{log_output}' did not match pattern '{pattern}'"
      )
    finally:
      logger.removeHandler(handler)

  def test_formatter_levels(self):
    # Verify mapping of level letters:
    # DEBUG -> D, INFO -> I, WARNING -> W, ERROR -> E, CRITICAL -> F
    logger = logging.getLogger("test_glog_levels")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(GlogFormatter())
    logger.addHandler(handler)

    try:
      logger.debug("debug message")
      logger.warning("warning message")
      logger.error("error message")
      logger.critical("critical message")

      lines = stream.getvalue().strip().split("\n")
      self.assertEqual(len(lines), 4)

      self.assertTrue(lines[0].startswith("D"))
      self.assertTrue(lines[1].startswith("W"))
      self.assertTrue(lines[2].startswith("E"))
      self.assertTrue(lines[3].startswith("F"))
    finally:
      logger.removeHandler(handler)


class TestLLMConversationLogging(unittest.TestCase):

  def test_debug_logging_enabled(self):
    import io
    from unittest.mock import Mock
    from chatty.session import ChatbotSession
    
    # Configure in-memory logger capture at DEBUG level
    logger = logging.getLogger("chatty")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    
    try:
      session = ChatbotSession(
        provider="openrouter",
        model="google/gemini-2.5-flash",
        context_size=10000,
        sandbox="/tmp",
        api_key="sk-or-v1-secretapikeyhere"
      )
      
      # Mock the OpenAI client structure
      session.client = Mock()
      session.client.base_url = "https://openrouter.ai/api/v1"
      session.client.api_key = "sk-or-v1-secretapikeyhere"
      session.client.default_headers = {
        "Authorization": "Bearer sk-or-v1-secretapikeyhere",
        "HTTP-Referer": "https://github.com/davidel/chatty"
      }
      session.client.timeout = 30.0
      session.client.max_retries = 2
      
      # Log LLM Request
      test_messages = [{"role": "user", "content": "Hello LLM!"}]
      session._log_llm_request(test_messages, tools=[{"type": "function", "function": {"name": "test_tool"}}])
      
      log_output = stream.getvalue()
      
      # Verify details are logged in DEBUG mode
      self.assertIn("=== LLM Request Details ===", log_output)
      self.assertIn("Provider: openrouter", log_output)
      self.assertIn("Model: google/gemini-2.5-flash", log_output)
      self.assertIn("Base URL: https://openrouter.ai/api/v1", log_output)
      # Verify API Key is masked
      self.assertNotIn("secretapikeyhere", log_output)
      self.assertIn("sk-or-v1...", log_output)
      self.assertIn("HTTP-Referer", log_output)
      self.assertIn("Hello LLM!", log_output)
      self.assertIn("test_tool", log_output)
      
      # Clear stream
      stream.truncate(0)
      stream.seek(0)
      
      # Log LLM Response Summary
      class MockUsage:
        prompt_tokens = 10
        completion_tokens = 20
        total_tokens = 30
        
      session._log_llm_response_summary(
        content_accumulated="Assistant text response",
        tool_calls_accumulated=[{"id": "call_123", "type": "function"}],
        extra_fields_accumulated={"reasoning": "Thought process..."},
        finish_reason="stop",
        usage=MockUsage(),
        response_model="google/gemini-2.5-flash-actual",
        system_fingerprint="fp_123",
        chunk_id="chunk_abc"
      )
      
      log_output_resp = stream.getvalue()
      self.assertIn("=== LLM Response Details ===", log_output_resp)
      self.assertIn("Chunk ID: chunk_abc", log_output_resp)
      self.assertIn("Response Model: google/gemini-2.5-flash-actual", log_output_resp)
      self.assertIn("System Fingerprint: fp_123", log_output_resp)
      self.assertIn("Finish Reason: stop", log_output_resp)
      self.assertIn("prompt_tokens=10", log_output_resp)
      self.assertIn("completion_tokens=20", log_output_resp)
      self.assertIn("total_tokens=30", log_output_resp)
      self.assertIn("Extra Field (reasoning): Thought process...", log_output_resp)
      self.assertIn("Assistant response content", log_output_resp)
      self.assertIn("call_123", log_output_resp)
      
    finally:
      logger.removeHandler(handler)
      logger.setLevel(old_level)

  def test_debug_logging_disabled_at_info(self):
    import io
    from unittest.mock import Mock
    from chatty.session import ChatbotSession
    
    # Configure in-memory logger capture at INFO level
    logger = logging.getLogger("chatty")
    old_level = logger.level
    logger.setLevel(logging.INFO)
    
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    
    try:
      session = ChatbotSession(
        provider="openrouter",
        model="google/gemini-2.5-flash",
        context_size=10000,
        sandbox="/tmp"
      )
      session.client = Mock()
      session.client.default_headers = {}
      
      # Clear standard initialization logs
      stream.truncate(0)
      stream.seek(0)
      
      session._log_llm_request([{"role": "user", "content": "Hello"}], None)
      session._log_llm_response_summary("Hi", [], {}, "stop")
      
      log_output = stream.getvalue()
      self.assertEqual(log_output, "")
      
    finally:
      logger.removeHandler(handler)
      logger.setLevel(old_level)

