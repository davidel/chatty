import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.utils import format_short_number


class TestShortNumber(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = tempfile.mkdtemp()

  def tearDown(self):
    os.chdir(self.old_cwd)
    shutil.rmtree(self.sandbox_dir)

  def test_format_short_number_small(self):
    self.assertEqual(format_short_number(0), "0")
    self.assertEqual(format_short_number(950), "950")
    self.assertEqual(format_short_number(950.5), "950.5")
    self.assertEqual(format_short_number(-950.5), "-950.5")

  def test_format_short_number_kilo(self):
    self.assertEqual(format_short_number(1000), "1K")
    self.assertEqual(format_short_number(1500), "1.5K")
    self.assertEqual(format_short_number(1550), "1.55K")
    self.assertEqual(format_short_number(999900), "999.9K")

  def test_format_short_number_mega(self):
    self.assertEqual(format_short_number(1000000), "1M")
    self.assertEqual(format_short_number(1500000), "1.5M")
    self.assertEqual(format_short_number(29880645), "29.88M")
    self.assertEqual(format_short_number(-29880645), "-29.88M")

  def test_format_short_number_giga_and_tera(self):
    self.assertEqual(format_short_number(1000000000), "1G")
    self.assertEqual(format_short_number(1500000000), "1.5G")
    self.assertEqual(format_short_number(1000000000000), "1T")
    self.assertEqual(format_short_number(1500000000000), "1.5T")

  def test_session_status_bar_formatted(self):
    session = ChatbotSession(
      provider="ollama",
      model="mock-model",
      context_size=10000,
      sandbox=self.sandbox_dir
    )
    session.model_usage = {
      "mock-model": {
        "prompt_tokens": 20000000,
        "completion_tokens": 9880645
      }
    }
    # Total cumulative usage = 29880645
    # Check get_rich_status_bar output contains formatted usage
    from rich.console import Console
    c = Console(width=200, record=True)
    c.print(session.get_rich_status_bar())
    rendered = c.export_text()
    self.assertIn("Usage: 29.88M", rendered)


if __name__ == "__main__":
  unittest.main()
