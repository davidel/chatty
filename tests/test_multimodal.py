import base64
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from chatty.attachments import (
  StagedAttachment,
  format_multimodal_user_message,
  stage_attachment
)
from chatty.session import ChatbotSession


class TestMultimodalPipeline(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = self.enterContext(tempfile.TemporaryDirectory())
    self.session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True
    )

  def tearDown(self):
    os.chdir(self.old_cwd)

  def test_format_multimodal_empty_attachments(self):
    msg = format_multimodal_user_message("Plain text prompt", [])
    self.assertEqual(msg["role"], "user")
    self.assertEqual(msg["content"], "Plain text prompt")

  def test_format_multimodal_with_images(self):
    raw_bytes = b"FAKE_PNG_BINARY_DATA"
    att = StagedAttachment(
      original_path="/tmp/test.png",
      filename="test.png",
      staged_path="/tmp/staged_test.png",
      media_type="image",
      page_numbers=[1],
      images=[(1, raw_bytes)]
    )

    msg = format_multimodal_user_message("Inspect this drawing", [att])
    self.assertEqual(msg["role"], "user")
    self.assertIsInstance(msg["content"], list)
    self.assertEqual(len(msg["content"]), 2)

    # First block is text
    self.assertEqual(msg["content"][0]["type"], "text")
    self.assertEqual(msg["content"][0]["text"], "Inspect this drawing")

    # Second block is image_url
    self.assertEqual(msg["content"][1]["type"], "image_url")
    expected_b64 = base64.b64encode(raw_bytes).decode("ascii")
    self.assertEqual(
      msg["content"][1]["image_url"]["url"],
      f"data:image/png;base64,{expected_b64}"
    )
    self.assertEqual(msg["content"][1]["image_url"]["detail"], "high")

  def test_format_multimodal_with_pdf_text_and_images(self):
    raw_bytes1 = b"PAGE1_DATA"
    raw_bytes2 = b"PAGE2_DATA"
    att = StagedAttachment(
      original_path="/tmp/doc.pdf",
      filename="doc.pdf",
      staged_path="/tmp/staged_doc.pdf",
      media_type="pdf",
      page_numbers=[1, 2],
      images=[(1, raw_bytes1), (2, raw_bytes2)],
      extracted_text="Part Dimension: 42mm"
    )

    msg = format_multimodal_user_message("Build this part", [att])
    self.assertEqual(msg["role"], "user")
    self.assertIsInstance(msg["content"], list)
    self.assertEqual(len(msg["content"]), 4)

    types = [b["type"] for b in msg["content"]]
    self.assertEqual(types, ["text", "text", "image_url", "image_url"])
    self.assertIn("Part Dimension: 42mm", msg["content"][1]["text"])

  def test_add_user_message_clears_staged(self):
    raw_bytes = b"IMG_BYTES"
    att = StagedAttachment(
      original_path="/tmp/test.png",
      filename="test.png",
      staged_path="/tmp/staged_test.png",
      media_type="image",
      page_numbers=[1],
      images=[(1, raw_bytes)]
    )
    self.session.staged_attachments.append(att)

    msg = self.session.add_user_message("Here is the blueprint")
    self.assertEqual(len(self.session.messages), 1)
    self.assertIsInstance(msg["content"], list)
    self.assertEqual(len(self.session.staged_attachments), 0)

  def test_calculate_tokens_multimodal(self):
    raw_bytes = b"IMG_BYTES"
    att = StagedAttachment(
      original_path="/tmp/test.png",
      filename="test.png",
      staged_path="/tmp/staged_test.png",
      media_type="image",
      page_numbers=[1],
      images=[(1, raw_bytes)]
    )
    self.session.staged_attachments.append(att)
    self.session.add_user_message("Check this")

    tokens = self.session._calculate_tokens_for_messages(self.session.messages)
    # Should include text estimate + 1000 image tokens
    self.assertGreater(tokens, 1000)

  def test_prune_history_with_multimodal_messages(self):
    raw_bytes = b"IMG_BYTES"
    att = StagedAttachment(
      original_path="/tmp/test.png",
      filename="test.png",
      staged_path="/tmp/staged_test.png",
      media_type="image",
      page_numbers=[1],
      images=[(1, raw_bytes)]
    )
    self.session.staged_attachments.append(att)
    self.session.add_user_message("First query with image")
    self.session.messages.append({"role": "assistant", "content": "I received the drawing."})
    self.session.messages.append({"role": "user", "content": "Second query text only"})

    pruned = self.session.prune_history()
    self.assertGreater(len(pruned), 0)

  def test_llm_cycle_receives_multimodal_payload(self):
    raw_bytes = b"DRAWING_PIXELS"
    att = StagedAttachment(
      original_path="/tmp/schematic.png",
      filename="schematic.png",
      staged_path="/tmp/staged_schematic.png",
      media_type="image",
      page_numbers=[1],
      images=[(1, raw_bytes)]
    )
    self.session.staged_attachments.append(att)
    self.session.add_user_message("Create this CAD model")

    mock_client = MagicMock()
    self.session.client = mock_client

    chunk = SimpleNamespace(
      choices=[SimpleNamespace(delta=SimpleNamespace(content="Done building CAD model", tool_calls=None))],
      usage=None,
      model_extra=None
    )
    mock_client.chat.completions.create.return_value = [chunk]

    self.session.run_llm_cycle()

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    sent_messages = call_kwargs["messages"]

    # Verify user message in kwargs is multimodal
    user_sent = [m for m in sent_messages if m.get("role") == "user"][0]
    self.assertIsInstance(user_sent["content"], list)
    image_blocks = [b for b in user_sent["content"] if b.get("type") == "image_url"]
    self.assertEqual(len(image_blocks), 1)
    self.assertIn("data:image/png;base64,", image_blocks[0]["image_url"]["url"])


if __name__ == "__main__":
  unittest.main()
