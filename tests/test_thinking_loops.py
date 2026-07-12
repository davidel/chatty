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

  @patch("chatty.session.openai.OpenAI")
  @patch("builtins.input")
  def test_interactive_thinking_stop(self, mock_input, mock_openai):
    self.session.headless = False
    mock_input.return_value = "s"

    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    self.session.client = mock_client

    # Attempt 1: Stream that exceeds thinking budget
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
      content="Final response.",
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

    mock_stream1 = MagicMock()
    mock_stream1.__iter__.return_value = [chunk1]
    mock_stream1.close = MagicMock()

    mock_stream2 = MagicMock()
    mock_stream2.__iter__.return_value = [chunk2]

    def mock_create(*args, **kwargs):
      if mock_client.chat.completions.create.call_count == 1:
        return mock_stream1
      return mock_stream2

    mock_client.chat.completions.create.side_effect = mock_create

    # Run the cycle
    self.session.run_llm_cycle()

    # Assertions
    mock_stream1.close.assert_called_once()
    self.assertEqual(mock_client.chat.completions.create.call_count, 2)
    self.assertEqual(self.session.messages[-1]["content"], "Final response.")

  @patch("chatty.session.openai.OpenAI")
  @patch("builtins.input")
  def test_interactive_thinking_let_go(self, mock_input, mock_openai):
    self.session.headless = False
    mock_input.return_value = "l"

    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    self.session.client = mock_client

    # Single stream with reasoning that exceeds budget, followed by final content in same stream
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
    
    delta2 = SimpleNamespace(
      content="Success without retry.",
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

    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = [chunk1, chunk2]
    mock_stream.close = MagicMock()

    mock_client.chat.completions.create.return_value = mock_stream

    # Run the cycle
    self.session.run_llm_cycle()

    # Assertions
    # 1. The stream was not closed early
    mock_stream.close.assert_not_called()
    # 2. Only one API call was made
    self.assertEqual(mock_client.chat.completions.create.call_count, 1)
    # 3. Final message is correct
    self.assertEqual(self.session.messages[-1]["content"], "Success without retry.")
    # 4. Config max_thinking_chars was updated
    self.assertEqual(self.session.config.max_thinking_chars, 280)

  @patch("chatty.session.openai.OpenAI")
  @patch("builtins.input")
  def test_interactive_thinking_whitelist(self, mock_input, mock_openai):
    self.session.headless = False
    mock_input.return_value = "w"

    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    self.session.client = mock_client

    delta1 = SimpleNamespace(
      content=None,
      tool_calls=None,
      reasoning="Thinking " * 20,
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
    
    delta2 = SimpleNamespace(
      content="Done.",
      tool_calls=None,
      reasoning="Thinking " * 40,
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

    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = [chunk1, chunk2]
    mock_stream.close = MagicMock()

    mock_client.chat.completions.create.return_value = mock_stream

    # Run the cycle
    self.session.run_llm_cycle()

    # Assertions
    # 1. input() was only called once
    mock_input.assert_called_once()
    # 2. Only one API call was made
    self.assertEqual(mock_client.chat.completions.create.call_count, 1)

  def test_detect_repetition_loop(self):
    from chatty.llm import detect_repetition_loop
    
    # 1. Normal non-repetitive text (should be False)
    text1 = "This is a normal thinking path where we explore various ideas. " * 3
    self.assertFalse(detect_repetition_loop(text1))

    # 2. Block of length 300+ repeating twice consecutively (should be True)
    block_large = "This is a large block of thinking text that is definitely longer than three hundred characters. " \
                  "We want to verify that when it repeats twice consecutively, it triggers the repetition detector. " \
                  "Let us add some more characters to make sure it exceeds the three hundred character limit. " \
                  "Indeed, it is quite long and detailed. "
    text2 = block_large + block_large
    self.assertTrue(detect_repetition_loop(text2))

    # 3. Block of length 100+ repeating three times consecutively (should be True)
    block_medium = "This is a medium block of thinking text that is at least one hundred characters. " \
                   "Let's add more text to make it longer than 100. "
    text3 = block_medium * 3
    self.assertTrue(detect_repetition_loop(text3))

    # 4. Short repeating block under 100 characters (should be False)
    block_short = "short repetition "
    text4 = block_short * 10
    self.assertFalse(detect_repetition_loop(text4))

  @patch("chatty.session.openai.OpenAI")
  def test_thinking_loop_aborts_on_repetition(self, mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    self.session.client = mock_client

    # A repeating stream of thoughts
    block = "This is a long thinking path that will repeat and trigger loop detection. Let's make sure it is at least 150 characters long so it triggers on the third repetition. Yes, indeed, we are writing a very verbose repeating block. "
    
    # We will simulate 3 chunks yielding this block successively
    chunks = []
    for idx in range(3):
      delta = SimpleNamespace(
        content=None,
        tool_calls=None,
        reasoning=block,
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

    def mock_create(*args, **kwargs):
      if mock_client.chat.completions.create.call_count == 1:
        return mock_stream1
      return mock_stream2

    mock_client.chat.completions.create.side_effect = mock_create

    # Run the cycle
    self.session.run_llm_cycle()

    # The stream should be aborted after the third chunk (index 2)
    mock_stream1.close.assert_called_once()
    self.assertEqual(mock_client.chat.completions.create.call_count, 2)
    self.assertEqual(self.session.messages[-1]["content"], "Success.")


if __name__ == "__main__":
  unittest.main()
