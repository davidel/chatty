import os
import sys
import unittest
from unittest.mock import patch, Mock

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.utils import tool_fetch_url


class TestPDFHandling(unittest.TestCase):

  @patch('requests.get')
  @patch('pypdf.PdfReader')
  def test_fetch_pdf_success(self, mock_pdf_reader, mock_get):
    # Mock requests.get response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.content = b"%PDF-1.4 mock data"
    mock_get.return_value = mock_response

    # Mock PdfReader
    mock_reader = Mock()
    mock_reader.is_encrypted = False
    
    mock_page1 = Mock()
    mock_page1.extract_text.return_value = "Page 1 Text Content"
    mock_page2 = Mock()
    mock_page2.extract_text.return_value = "Page 2 Text Content"
    
    mock_reader.pages = [mock_page1, mock_page2]
    mock_pdf_reader.return_value = mock_reader

    res = tool_fetch_url("https://example.com/document.pdf")
    self.assertIn("--- Page 1 ---", res)
    self.assertIn("Page 1 Text Content", res)
    self.assertIn("--- Page 2 ---", res)
    self.assertIn("Page 2 Text Content", res)

  @patch('requests.get')
  @patch('pypdf.PdfReader')
  def test_fetch_pdf_encrypted(self, mock_pdf_reader, mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.content = b"%PDF-1.4 mock encrypted"
    mock_get.return_value = mock_response

    mock_reader = Mock()
    mock_reader.is_encrypted = True
    mock_pdf_reader.return_value = mock_reader

    res = tool_fetch_url("https://example.com/encrypted.pdf")
    self.assertEqual(res, "[Error: The requested PDF is password-protected or encrypted.]")

  @patch('requests.get')
  @patch('pypdf.PdfReader')
  @patch('shutil.which')
  def test_fetch_pdf_scanned_no_ocr(self, mock_which, mock_pdf_reader, mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.content = b"%PDF-1.4 mock scanned"
    mock_get.return_value = mock_response

    mock_reader = Mock()
    mock_reader.is_encrypted = False
    mock_page = Mock()
    mock_page.extract_text.return_value = "   "  # Empty text extraction
    mock_reader.pages = [mock_page]
    mock_pdf_reader.return_value = mock_reader

    # Mock no binaries in PATH
    mock_which.return_value = None

    res = tool_fetch_url("https://example.com/scanned.pdf")
    self.assertIn("Scanned PDF detected, but tesseract or pdftoppm is missing", res)

  @patch('requests.get')
  @patch('pypdf.PdfReader')
  @patch('shutil.which')
  def test_fetch_pdf_scanned_with_ocr(self, mock_which, mock_pdf_reader, mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.content = b"%PDF-1.4 mock scanned"
    mock_get.return_value = mock_response

    mock_reader = Mock()
    mock_reader.is_encrypted = False
    mock_page = Mock()
    mock_page.extract_text.return_value = ""  # Empty text extraction
    mock_reader.pages = [mock_page]
    mock_pdf_reader.return_value = mock_reader

    # Mock binaries are present
    mock_which.side_effect = lambda x: "/usr/bin/" + x

    # Mock OCR packages present in sys.modules
    mock_pdf2image = Mock()
    mock_pytesseract = Mock()
    
    # We patch the dynamic imports
    with patch.dict(sys.modules, {'pdf2image': mock_pdf2image, 'pytesseract': mock_pytesseract}):
      mock_image = Mock()
      mock_pdf2image.convert_from_bytes.return_value = [mock_image]
      mock_pytesseract.image_to_string.return_value = "OCR Text Content"

      res = tool_fetch_url("https://example.com/scanned.pdf")
      self.assertIn("--- Page 1 (OCR) ---", res)
      self.assertIn("OCR Text Content", res)


if __name__ == "__main__":
  unittest.main()
