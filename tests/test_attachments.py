import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from chatty.attachments import (
  parse_page_spec,
  detect_pdf_engine,
  get_pdf_total_pages,
  rasterize_pdf_pages,
  stage_attachment,
  get_attachments_dir,
  StagedAttachment
)
from chatty.session import ChatbotSession
from chatty.commands import COMMANDS


class TestAttachments(unittest.TestCase):

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

  def test_parse_page_spec_defaults(self):
    # None or empty string defaults to 1-3
    self.assertEqual(parse_page_spec(None), [1, 2, 3])
    self.assertEqual(parse_page_spec(""), [1, 2, 3])
    self.assertEqual(parse_page_spec("   "), [1, 2, 3])

    # When total_pages is fewer than 3, clamp to total_pages
    self.assertEqual(parse_page_spec(None, total_pages=2), [1, 2])
    self.assertEqual(parse_page_spec(None, total_pages=1), [1])
    self.assertEqual(parse_page_spec("", total_pages=10), [1, 2, 3])

  def test_parse_page_spec_all(self):
    self.assertEqual(parse_page_spec("all", total_pages=5), [1, 2, 3, 4, 5])
    self.assertEqual(parse_page_spec("ALL", total_pages=2), [1, 2])

  def test_parse_page_spec_ranges_and_lists(self):
    self.assertEqual(parse_page_spec("2"), [2])
    self.assertEqual(parse_page_spec("1-4"), [1, 2, 3, 4])
    self.assertEqual(parse_page_spec("1,3,5"), [1, 3, 5])
    self.assertEqual(parse_page_spec("1-2, 5, 8-9"), [1, 2, 5, 8, 9])

    # Out of range clamped when total_pages provided
    self.assertEqual(parse_page_spec("1-10", total_pages=3), [1, 2, 3])

    # Invalid entries skipped
    self.assertEqual(parse_page_spec("abc, 2, xyz"), [2])
    self.assertEqual(parse_page_spec("-1, 0, 1"), [1])

  def test_detect_pdf_engine_missing(self):
    # When none of pypdfium2, pdf2image, fitz are installed
    with patch.dict("sys.modules", {"pypdfium2": None, "pdf2image": None, "fitz": None}):
      engine, rec = detect_pdf_engine()
      self.assertIsNone(engine)
      self.assertIsNotNone(rec)
      self.assertIn("pip install pypdfium2", rec)
      self.assertIn("sudo apt install poppler-utils && pip install pdf2image", rec)

  def test_detect_pdf_engine_pypdfium2(self):
    mock_pypdfium2 = MagicMock()
    with patch.dict("sys.modules", {"pypdfium2": mock_pypdfium2}):
      engine, rec = detect_pdf_engine()
      self.assertEqual(engine, "pypdfium2")
      self.assertIsNone(rec)

  def test_detect_pdf_engine_pdf2image(self):
    mock_pdf2image = MagicMock()
    with patch.dict("sys.modules", {"pypdfium2": None, "pdf2image": mock_pdf2image}):
      with patch("shutil.which", return_value="/usr/bin/pdftoppm"):
        engine, rec = detect_pdf_engine()
        self.assertEqual(engine, "pdf2image")
        self.assertIsNone(rec)

  def test_detect_pdf_engine_fitz(self):
    mock_fitz = MagicMock()
    with patch.dict("sys.modules", {"pypdfium2": None, "pdf2image": None, "fitz": mock_fitz}):
      engine, rec = detect_pdf_engine()
      self.assertEqual(engine, "fitz")
      self.assertIsNone(rec)

  def test_stage_attachment_image(self):
    png_path = os.path.join(self.sandbox_dir, "test.png")
    with open(png_path, "wb") as f:
      f.write(b"\x89PNG\r\n\x1a\nFakePngBytes")

    att, err = stage_attachment(self.sandbox_dir, png_path)
    self.assertIsNone(err)
    self.assertIsNotNone(att)
    self.assertEqual(att.filename, "test.png")
    self.assertEqual(att.media_type, "image")
    self.assertEqual(len(att.images), 1)
    self.assertEqual(att.images[0][1], b"\x89PNG\r\n\x1a\nFakePngBytes")
    self.assertTrue(os.path.exists(att.staged_path))

  def test_stage_attachment_svg(self):
    svg_path = os.path.join(self.sandbox_dir, "drawing.svg")
    svg_content = "<svg width='100' height='100'><circle cx='50' cy='50' r='40'/></svg>"
    with open(svg_path, "w", encoding="utf-8") as f:
      f.write(svg_content)

    att, err = stage_attachment(self.sandbox_dir, svg_path)
    self.assertIsNone(err)
    self.assertIsNotNone(att)
    self.assertEqual(att.filename, "drawing.svg")
    self.assertEqual(att.media_type, "svg")
    self.assertEqual(att.extracted_text, svg_content)
    self.assertTrue(os.path.exists(att.staged_path))

  def test_stage_attachment_errors(self):
    # Nonexistent file
    att, err = stage_attachment(self.sandbox_dir, "does_not_exist.png")
    self.assertIsNone(att)
    self.assertIn("File not found", err)

    # Directory instead of file
    att, err = stage_attachment(self.sandbox_dir, self.sandbox_dir)
    self.assertIsNone(att)
    self.assertIn("Path is not a file", err)

    # Unsupported format
    txt_path = os.path.join(self.sandbox_dir, "test.txt")
    with open(txt_path, "w") as f:
      f.write("hello")
    att, err = stage_attachment(self.sandbox_dir, txt_path)
    self.assertIsNone(att)
    self.assertIn("Unsupported attachment format", err)

  def test_cmd_attach_usage_and_errors(self):
    with patch("chatty.commands.console.print") as mock_print:
      res = COMMANDS["/attach"](self.session, "")
      self.assertTrue(res)
      mock_print.assert_any_call("[bold red]Usage: /attach <file_path> [--pages <spec>][/bold red]")

    with patch("chatty.commands.console.print") as mock_print:
      res = COMMANDS["/attach"](self.session, "missing.png")
      self.assertTrue(res)
      mock_print.assert_any_call("[bold red]Error: File 'missing.png' does not exist or is not a file.[/bold red]")

  def test_cmd_attach_image_success(self):
    png_path = os.path.join(self.sandbox_dir, "diagram.png")
    with open(png_path, "wb") as f:
      f.write(b"PNG_DATA")

    with patch("chatty.commands.console.print") as mock_print:
      res = COMMANDS["/attach"](self.session, "diagram.png")
      self.assertTrue(res)
      self.assertEqual(len(self.session.staged_attachments), 1)
      self.assertEqual(self.session.staged_attachments[0].filename, "diagram.png")
      mock_print.assert_any_call("[bold green]Successfully staged attachment:[/bold green] [cyan]diagram.png[/cyan]")

    # /upload alias also works
    with patch("chatty.commands.console.print") as mock_print:
      res = COMMANDS["/upload"](self.session, "diagram.png")
      self.assertTrue(res)
      self.assertEqual(len(self.session.staged_attachments), 2)

    # /attach list displays staged files
    with patch("chatty.commands.console.print") as mock_print:
      res = COMMANDS["/attach"](self.session, "list")
      self.assertTrue(res)

    # /attach clear clears attachments
    with patch("chatty.commands.console.print") as mock_print:
      res = COMMANDS["/attach"](self.session, "clear")
      self.assertTrue(res)
      self.assertEqual(len(self.session.staged_attachments), 0)
      mock_print.assert_any_call("[bold yellow]Cleared staged attachments.[/bold yellow]")

  def test_cmd_attach_pdf_pages_parsing(self):
    pdf_path = os.path.join(self.sandbox_dir, "schematic.pdf")
    with open(pdf_path, "wb") as f:
      f.write(b"%PDF-1.4 FakePDFContent")

    with patch("chatty.attachments.get_pdf_total_pages", return_value=10):
      with patch("chatty.attachments.detect_pdf_engine", return_value=(None, "Visual PDF rendering requires a PDF-to-image engine.")):
        res = COMMANDS["/attach"](self.session, "schematic.pdf --pages 1,3,5")
        self.assertTrue(res)
        self.assertEqual(len(self.session.staged_attachments), 1)
        att = self.session.staged_attachments[0]
        self.assertEqual(att.filename, "schematic.pdf")
        self.assertEqual(att.page_numbers, [1, 3, 5])
        self.assertIn("Visual PDF rendering requires a PDF-to-image engine", att.warning)

  def test_chatty_completer_attach(self):
    from prompt_toolkit.document import Document
    from chatty.session import ChattyCompleter

    completer = ChattyCompleter(["/attach", "/upload"], session=self.session)

    # Create dummy attachment file
    dummy_file = os.path.join(self.sandbox_dir, "schematic_sample.pdf")
    with open(dummy_file, "w") as f:
      f.write("")

    completions = list(completer.get_completions(Document("/attach schematic_sam"), None))
    self.assertTrue(any(c.display_text == "schematic_sample.pdf" for c in completions))

    completions_upload = list(completer.get_completions(Document("/upload schematic_sam"), None))
    self.assertTrue(any(c.display_text == "schematic_sample.pdf" for c in completions_upload))


if __name__ == "__main__":
  unittest.main()
