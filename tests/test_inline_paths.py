import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from prompt_toolkit.document import Document

from chatty.attachments import (
  StagedAttachment,
  extract_inline_paths,
  resolve_inline_path,
  stage_text_file,
  process_inline_attachments,
)
from chatty.session import ChatbotSession, ChattyCompleter


class TestInlinePaths(unittest.TestCase):
  def setUp(self):
    self.old_cwd = os.getcwd()
    self.temp_dir = tempfile.mkdtemp()
    self.sandbox_dir = os.path.join(self.temp_dir, "sandbox")
    os.makedirs(self.sandbox_dir, exist_ok=True)
    os.chdir(self.sandbox_dir)

    self.session = ChatbotSession.__new__(ChatbotSession)
    self.session.sandbox = self.sandbox_dir
    self.session.allowed_ro_paths = []
    self.session.allowed_rw_paths = []
    self.session.auto_approve = False
    self.session.messages = []
    self.session.staged_attachments = []
    self.session.provider = "openrouter"
    self.session.model = "anthropic/claude-3.5-sonnet"
    self.session.model_usage = {}
    self.session.context_limit = 100000
    self.session.prompt_caching = False
    self.session.agent_profile = "general"
    self.session.profile = "general"
    self.session._print = MagicMock()

  def tearDown(self):
    os.chdir(self.old_cwd)
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def test_extract_inline_paths(self):
    text = (
      "Can you inspect @drawing.pdf and @\"my blueprint.pdf\":1-2? "
      "Also check @'layout.svg':all, @photo.png., and (@box.cad). "
      "Do not trigger on email@example.com or regular text."
    )
    extracted = extract_inline_paths(text)
    expected = [
      ("drawing.pdf", None),
      ("my blueprint.pdf", "1-2"),
      ("layout.svg", "all"),
      ("photo.png", None),
      ("box.cad", None),
    ]
    self.assertEqual(extracted, expected)

  def test_resolve_inline_path(self):
    file_in_sandbox = os.path.join(self.sandbox_dir, "model.step")
    with open(file_in_sandbox, "w") as f:
      f.write("STEP-FILE")

    # Relative to sandbox
    resolved = resolve_inline_path("model.step", self.sandbox_dir)
    self.assertEqual(resolved, os.path.abspath(file_in_sandbox))

    # Absolute path
    resolved_abs = resolve_inline_path(file_in_sandbox, self.sandbox_dir)
    self.assertEqual(resolved_abs, os.path.abspath(file_in_sandbox))

    # Nonexistent
    resolved_none = resolve_inline_path("nonexistent.cad", self.sandbox_dir)
    self.assertIsNone(resolved_none)

  def test_stage_text_file(self):
    code_path = os.path.join(self.sandbox_dir, "test_code.py")
    with open(code_path, "w", encoding="utf-8") as f:
      f.write("def hello():\n  return 'world'\n")

    att, err = stage_text_file(self.sandbox_dir, code_path)
    self.assertIsNone(err)
    self.assertIsNotNone(att)
    self.assertEqual(att.media_type, "text")
    self.assertEqual(att.filename, "test_code.py")
    self.assertIn("def hello():", att.extracted_text)

  def test_process_inline_attachments(self):
    png_path = os.path.join(self.sandbox_dir, "diagram.png")
    with open(png_path, "wb") as f:
      f.write(b"\x89PNG\r\n\x1a\nfake_bytes")

    txt_path = os.path.join(self.sandbox_dir, "notes.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
      f.write("Project specifications and tolerances.")

    prompt = "Review @diagram.png and @notes.txt along with @missing.png and duplicate @diagram.png"
    attachments, warnings = process_inline_attachments(prompt, self.sandbox_dir)

    self.assertEqual(len(attachments), 2)
    types = {att.media_type for att in attachments}
    self.assertIn("image", types)
    self.assertIn("text", types)

  def test_add_user_message_with_inline_multimodal(self):
    png_path = os.path.join(self.sandbox_dir, "bracket.png")
    with open(png_path, "wb") as f:
      f.write(b"PNG_PIXELS")

    msg = self.session.add_user_message("Can you build this CAD model from @bracket.png?")
    self.assertEqual(len(self.session.messages), 1)
    self.assertIsInstance(msg["content"], list)

    types = [block["type"] for block in msg["content"]]
    self.assertIn("text", types)
    self.assertIn("image_url", types)
    self.assertEqual(msg["content"][0]["text"], "Can you build this CAD model from @bracket.png?")

  def test_add_user_message_combines_staged_and_inline(self):
    existing_att = StagedAttachment(
      original_path="/tmp/staged_prev.png",
      filename="staged_prev.png",
      staged_path="/tmp/staged_prev.png",
      media_type="image",
      page_numbers=[1],
      images=[(1, b"PREV_BYTES")]
    )
    self.session.staged_attachments.append(existing_att)

    code_path = os.path.join(self.sandbox_dir, "config.py")
    with open(code_path, "w", encoding="utf-8") as f:
      f.write("DEBUG = True")

    msg = self.session.add_user_message("Check @config.py")
    self.assertEqual(len(self.session.staged_attachments), 0)
    self.assertIsInstance(msg["content"], list)

    texts = [b["text"] for b in msg["content"] if b["type"] == "text"]
    combined_text = "\n".join(texts)
    self.assertIn("config.py", combined_text)
    self.assertIn("DEBUG = True", combined_text)

    has_image = any(b["type"] == "image_url" for b in msg["content"])
    self.assertTrue(has_image)

  def test_chatty_completer_inline_at_paths(self):
    completer = ChattyCompleter(["/exit", "/help"], session=self.session)

    test_file = os.path.join(self.sandbox_dir, "schematic_draft.png")
    with open(test_file, "w") as f:
      f.write("draft")

    # Autocomplete after '@'
    doc = Document("inspect @schematic_dr")
    completions = list(completer.get_completions(doc, None))
    self.assertTrue(any(c.display_text == "schematic_draft.png" for c in completions))

    # Autocomplete directly at '@'
    doc_at = Document("inspect @")
    completions_at = list(completer.get_completions(doc_at, None))
    self.assertTrue(any(c.display_text == "schematic_draft.png" for c in completions_at))

    # Autocomplete with quote '@"'
    doc_quoted = Document('inspect @"schematic_dr')
    completions_quoted = list(completer.get_completions(doc_quoted, None))
    self.assertTrue(any(c.display_text == "schematic_draft.png" for c in completions_quoted))

    # Does not trigger on email address
    doc_email = Document("contact info@domain")
    completions_email = list(completer.get_completions(doc_email, None))
    self.assertEqual(len(completions_email), 0)


if __name__ == "__main__":
  unittest.main()
