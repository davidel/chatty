import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import sys
from types import SimpleNamespace

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.llm import ThinkingBudgetExceeded


class TestThinkingLoops(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = tempfile.mkdtemp()
    self.session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True
    )
    # Set the thinking budget on the existing config object
    self.session.config.max_thinking_chars = 100
    self.session.config.max_thinking_leeway_chars = 0

  def tearDown(self):
    os.chdir(self.old_cwd)
    shutil.rmtree(self.sandbox_dir)

  @patch("chatty.session.openai.OpenAI")
  def test_thinking_loop_aborts_and_retries_with_nudge(self, mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    self.session.client = mock_client

    # Attempt 1: Stream that exceeds thinking budget (180 chars of reasoning)
    delta1 = SimpleNamespace(
      content=None,
      tool_calls=None,
      reasoning="Thinking " * 20,  # 180 chars, exceeds budget of 100
      reasoning_content=None,
      reasoning_details=None,
      thought_signature=None,
      model_extra=None
    )
    choice1 = SimpleNamespace(
      delta=delta1,
      finish_reason=None
    )
    chunk1 = SimpleNamespace(
      choices=[choice1],
      id="chunk-1",
      model="test-model",
      system_fingerprint=None
    )
    
    # Attempt 2: Successful stream with final content
    delta2 = SimpleNamespace(
      content="Final response after correction.",
      tool_calls=None,
      reasoning=None,
      reasoning_content=None,
      reasoning_details=None,
      thought_signature=None,
      model_extra=None
    )
    choice2 = SimpleNamespace(
      delta=delta2,
      finish_reason="stop"
    )
    chunk2 = SimpleNamespace(
      choices=[choice2],
      id="chunk-2",
      model="test-model",
      system_fingerprint=None
    )

    # We mock stream close
    mock_stream1 = MagicMock()
    mock_stream1.__iter__.return_value = [chunk1]
    mock_stream1.close = MagicMock()

    mock_stream2 = MagicMock()
    mock_stream2.__iter__.return_value = [chunk2]

    # Captured messages sent to API in the second call
    captured_messages_on_retry = []

    def mock_create(*args, **kwargs):
      # Capture messages passed to the API call
      nonlocal captured_messages_on_retry
      captured_messages_on_retry = kwargs.get("messages", [])
      if mock_client.chat.completions.create.call_count == 1:
        return mock_stream1
      return mock_stream2

    mock_client.chat.completions.create.side_effect = mock_create

    # Run the cycle
    self.session.run_llm_cycle()

    # Assertions
    # 1. First stream close was called
    mock_stream1.close.assert_called_once()
    
    # 2. Re-tried API call was made
    self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    # 3. Second call included the nudge message containing the previous thoughts
    nudge_msg = captured_messages_on_retry[-1]
    self.assertEqual(nudge_msg["role"], "user")
    self.assertIn("aborted your previous attempt", nudge_msg["content"])
    self.assertIn("Thinking Thinking", nudge_msg["content"])

    # 4. Final message in session matches final response content
    self.assertEqual(self.session.messages[-1]["content"], "Final response after correction.")

  @patch("chatty.session.openai.OpenAI")
  def test_thinking_loop_interrupts_at_sentence_boundary(self, mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    self.session.client = mock_client

    self.session.config.max_thinking_chars = 50
    self.session.config.max_thinking_leeway_chars = 100

    # We will simulate a stream that yields chunks:
    # 1. "This is a test. We should keep thinking" (length 38) -> total 38 (under 50, no abort)
    # 2. " and thinking" (length 13) -> total 51 (over 50, but no punctuation, should not abort)
    # 3. ". Now we stop." (length 14) -> total 65 (over 50, ends with punctuation, should abort)
    # 4. " This should not be reached"
    chunks = []
    texts = [
      "This is a test. We should keep thinking",
      " and thinking",
      ". Now we stop.",
      " This should not be reached"
    ]
    for idx, text in enumerate(texts):
      delta = SimpleNamespace(
        content=None,
        tool_calls=None,
        reasoning=text,
        reasoning_content=None,
        reasoning_details=None,
        thought_signature=None,
        model_extra=None
      )
      choice = SimpleNamespace(
        delta=delta,
        finish_reason=None
      )
      chunk = SimpleNamespace(
        choices=[choice],
        id=f"chunk-{idx}",
        model="test-model",
        system_fingerprint=None
      )
      chunks.append(chunk)

    mock_stream1 = MagicMock()
    mock_stream1.__iter__.return_value = chunks
    mock_stream1.close = MagicMock()

    # Attempt 2: Successful stream with final response
    delta2 = SimpleNamespace(
      content="Success.",
      tool_calls=None,
      reasoning=None,
      reasoning_content=None,
      reasoning_details=None,
      thought_signature=None,
      model_extra=None
    )
    choice2 = SimpleNamespace(
      delta=delta2,
      finish_reason="stop"
    )
    chunk2 = SimpleNamespace(
      choices=[choice2],
      id="chunk-success",
      model="test-model",
      system_fingerprint=None
    )
    mock_stream2 = MagicMock()
    mock_stream2.__iter__.return_value = [chunk2]

    captured_messages_on_retry = []

    def mock_create(*args, **kwargs):
      nonlocal captured_messages_on_retry
      captured_messages_on_retry = kwargs.get("messages", [])
      if mock_client.chat.completions.create.call_count == 1:
        return mock_stream1
      return mock_stream2

    mock_client.chat.completions.create.side_effect = mock_create

    # Run the cycle
    self.session.run_llm_cycle()

    # Assertions
    # The stream should be closed after chunk 2 (index 2, which is ". Now we stop.")
    mock_stream1.close.assert_called_once()
    self.assertEqual(mock_client.chat.completions.create.call_count, 2)
    
    # Verify the accumulated thinking contains only up to ". Now we stop."
    nudge_msg = captured_messages_on_retry[-1]
    self.assertEqual(nudge_msg["role"], "user")
    self.assertIn("aborted your previous attempt", nudge_msg["content"])
    self.assertIn("This is a test. We should keep thinking and thinking. Now we stop.", nudge_msg["content"])
    self.assertNotIn("This should not be reached", nudge_msg["content"])


if __name__ == "__main__":
  unittest.main()
