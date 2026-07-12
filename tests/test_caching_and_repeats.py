import unittest
import os
import shutil
import tempfile
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession


class TestCachingAndRepeats(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = tempfile.mkdtemp()

  def tearDown(self):
    os.chdir(self.old_cwd)
    shutil.rmtree(self.sandbox_dir)

  def test_token_caching_invalidation(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      context_size=8000,
      sandbox=self.sandbox_dir,
      max_loops=15
    )

    # Initial state
    self.assertIsNone(session._cached_history_tokens)

    # Calling get_rich_status_bar should compute and cache tokens
    session.get_rich_status_bar()
    self.assertIsNotNone(session._cached_history_tokens)
    initial_tokens = session._cached_history_tokens

    # Appending a message should invalidate the cache
    session.messages.append({"role": "user", "content": "Hello"})
    self.assertIsNone(session._cached_history_tokens)

    # Computing again
    session.get_rich_status_bar()
    self.assertIsNotNone(session._cached_history_tokens)
    self.assertNotEqual(initial_tokens, session._cached_history_tokens)

    # Setter should invalidate cache
    session.messages = []
    self.assertIsNone(session._cached_history_tokens)

  def test_duplicate_metadata_filtering(self):
    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      context_size=8000,
      sandbox=self.sandbox_dir,
      max_loops=15
    )

    # Mock chunk structure to simulate OpenAI delta response
    class MockDelta:

      def __init__(self, **kwargs):
        for k, v in kwargs.items():
          setattr(self, k, v)

    class MockChoice:

      def __init__(self, delta):
        self.delta = delta

    class MockChunk:

      def __init__(self, choices):
        self.choices = choices

    # Test cases: delta chunks stream
    stream = [
      MockChunk([MockChoice(MockDelta(content=None, reasoning="🧠", thought_signature="sig1"))]),
      MockChunk([MockChoice(MockDelta(content=None, reasoning="🧠", thought_signature="sig1"))]),
      MockChunk([MockChoice(MockDelta(content="Hello", reasoning="🧠", thought_signature="sig1"))]),
      MockChunk([MockChoice(MockDelta(content=" world", reasoning="🧠", thought_signature="sig1"))]),
    ]

    extra_fields_accumulated = {
      "reasoning": None,
      "reasoning_content": None,
      "reasoning_details": None,
      "thought_signature": None
    }
    last_extra_fields = {
      "reasoning": None,
      "reasoning_content": None,
      "reasoning_details": None,
      "thought_signature": None
    }

    # Simulate how the stream loop extracts and accumulates fields
    extra_fields = ["reasoning", "reasoning_content", "reasoning_details", "thought_signature"]
    for chunk in stream:
      choice = chunk.choices[0]
      delta = choice.delta
      for field in extra_fields:
        val = getattr(delta, field, None)
        if val is not None:
          if field in ("reasoning", "reasoning_content"):
            if isinstance(val, str) and isinstance(extra_fields_accumulated[field], str):
              if val.startswith(extra_fields_accumulated[field]):
                extra_fields_accumulated[field] = val
              else:
                is_duplicate = (val == last_extra_fields[field])
                is_metadata_like = len(val) > 1 or any(ord(c) > 0x2000 for c in val)
                if not (is_duplicate and is_metadata_like):
                  extra_fields_accumulated[field] += val
            else:
              extra_fields_accumulated[field] = val
          else:
            extra_fields_accumulated[field] = val

          last_extra_fields[field] = val

    # Verify results
    # reasoning was a duplicate emoji '🧠', so it should NOT have been appended to '🧠🧠🧠🧠'
    self.assertEqual(extra_fields_accumulated["reasoning"], "🧠")
    # thought_signature should NOT have been appended to 'sig1sig1sig1sig1'
    self.assertEqual(extra_fields_accumulated["thought_signature"], "sig1")

  def test_delta_reasoning_accumulation(self):
    extra_fields_accumulated = {"reasoning": None}
    last_extra_fields = {"reasoning": None}
    
    # Delta-style streaming: each chunk has new content
    chunks = ["I", " think", " we", " need", " to", " patch"]
    for val in chunks:
      if isinstance(val, str) and isinstance(extra_fields_accumulated["reasoning"], str):
        if val.startswith(extra_fields_accumulated["reasoning"]):
          extra_fields_accumulated["reasoning"] = val
        else:
          is_duplicate = (val == last_extra_fields["reasoning"])
          is_metadata_like = len(val) > 1 or any(ord(c) > 0x2000 for c in val)
          if not (is_duplicate and is_metadata_like):
            extra_fields_accumulated["reasoning"] += val
      else:
        extra_fields_accumulated["reasoning"] = val
      last_extra_fields["reasoning"] = val
      
    self.assertEqual(extra_fields_accumulated["reasoning"], "I think we need to patch")

  def test_full_reasoning_accumulation(self):
    extra_fields_accumulated = {"reasoning": None}
    last_extra_fields = {"reasoning": None}
    
    # Full-style streaming: each chunk has the full accumulated string so far
    chunks = ["I", "I think", "I think we", "I think we need", "I think we need to", "I think we need to patch"]
    for val in chunks:
      if isinstance(val, str) and isinstance(extra_fields_accumulated["reasoning"], str):
        if val.startswith(extra_fields_accumulated["reasoning"]):
          extra_fields_accumulated["reasoning"] = val
        else:
          is_duplicate = (val == last_extra_fields["reasoning"])
          is_metadata_like = len(val) > 1 or any(ord(c) > 0x2000 for c in val)
          if not (is_duplicate and is_metadata_like):
            extra_fields_accumulated["reasoning"] += val
      else:
        extra_fields_accumulated["reasoning"] = val
      last_extra_fields["reasoning"] = val
      
    self.assertEqual(extra_fields_accumulated["reasoning"], "I think we need to patch")
