import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.tools import execute_tool, TOOLS_SCHEMA
from chatty.tools.file_ops import tool_write_file, tool_read_file
from chatty.tools.search_ops import tool_locate_files
from chatty.tools.system_ops import tool_sleep
from chatty.tools.web_ops import tool_search_web


class TestToolsModularity(unittest.TestCase):

  def setUp(self):
    self.sandbox_dir = self.enterContext(tempfile.TemporaryDirectory())

  def test_schema_validity(self):
    self.assertTrue(len(TOOLS_SCHEMA) > 0)
    names = [t["function"]["name"] for t in TOOLS_SCHEMA]
    self.assertIn("read_file", names)
    self.assertIn("write_file", names)
    self.assertIn("search_grep", names)
    self.assertIn("hex_dump", names)

  def test_file_ops_modular(self):
    write_res = tool_write_file(self.sandbox_dir, "test.txt", "Hello Modular Tools")
    self.assertEqual(write_res, "Successfully wrote to file 'test.txt'.")
    self.assertTrue(os.path.exists(os.path.join(self.sandbox_dir, "test.txt")))

    read_res = tool_read_file(self.sandbox_dir, "test.txt")
    self.assertEqual(read_res, "Hello Modular Tools")

  def test_read_file_clamping_and_bounds(self):
    tool_write_file(self.sandbox_dir, "multi.txt", "line1\nline2\nline3\n")

    # 1. Clamp when end_line exceeds total lines
    res = tool_read_file(self.sandbox_dir, "multi.txt", start_line=2, end_line=10)
    self.assertIn("line2\nline3\n", res)
    self.assertIn("WARNING: Requested end_line (10) exceeds total file lines (3)", res)

    # 2. Clamp with line_numbers=True
    res_ln = tool_read_file(self.sandbox_dir, "multi.txt", start_line=2, end_line=10, line_numbers=True)
    self.assertIn("2: line2\n3: line3\n", res_ln)
    self.assertIn("WARNING: Requested end_line (10) exceeds total file lines (3)", res_ln)

    # 3. start_line out of range is a hard error
    res_err = tool_read_file(self.sandbox_dir, "multi.txt", start_line=5, end_line=10)
    self.assertEqual(res_err, "Error: start_line 5 is out of range. The file 'multi.txt' has 3 lines.")

    # 4. end_line < start_line is a hard error
    res_inv = tool_read_file(self.sandbox_dir, "multi.txt", start_line=2, end_line=1)
    self.assertEqual(res_inv, "Error: end_line 1 cannot be less than start_line 2.")

    # 5. Empty file returns empty string
    tool_write_file(self.sandbox_dir, "empty.txt", "")
    res_empty = tool_read_file(self.sandbox_dir, "empty.txt", start_line=1, end_line=5)
    self.assertEqual(res_empty, "")

  def test_search_ops_modular(self):
    tool_write_file(self.sandbox_dir, "test_search.py", "print('hello')")
    locate_res = tool_locate_files(self.sandbox_dir, "*.py")
    self.assertIn("test_search.py", locate_res)

  def test_system_ops_modular(self):
    sleep_res = tool_sleep(0.01)
    self.assertIn("Successfully slept for 0.01 seconds", sleep_res)

  def test_patch_file_with_search_replace(self):
    class MockSession:
      def __init__(self, sandbox):
        self.sandbox = sandbox
    session = MockSession(self.sandbox_dir)
    
    test_file = "test_sr.txt"
    with open(os.path.join(self.sandbox_dir, test_file), "w") as f:
      f.write("Line 1\nLine 2\nLine 3\n")
      
    res = execute_tool(
      "patch_file",
      {
        "path": test_file,
        "search": "Line 2",
        "replace": "Line 2 modified"
      },
      session
    )
    self.assertIn("Successfully updated file", res)
    with open(os.path.join(self.sandbox_dir, test_file), "r") as f:
      content = f.read()
    self.assertEqual(content, "Line 1\nLine 2 modified\nLine 3\n")

  def test_patch_file_backslash_tolerance(self):
    class MockSession:
      def __init__(self, sandbox):
        self.sandbox = sandbox
    session = MockSession(self.sandbox_dir)
    
    test_file = "test_backslash.txt"
    with open(os.path.join(self.sandbox_dir, test_file), "w") as f:
      f.write("scdiag-pretrain --datasets HAM10000 \"redlessone/Derm1M\" \\\\\n")
      
    patch_data = """<<<<<<< SEARCH
scdiag-pretrain --datasets HAM10000 "redlessone/Derm1M" \\
=======
scdiag-pretrain --model convvit \\
                --datasets HAM10000 "redlessone/Derm1M" \\
>>>>>>> REPLACE"""
    
    res = execute_tool(
      "patch_file",
      {
        "path": test_file,
        "patch": patch_data
      },
      session
    )
    self.assertIn("Successfully updated file", res)
    with open(os.path.join(self.sandbox_dir, test_file), "r") as f:
      content = f.read()
    self.assertIn("scdiag-pretrain --model convvit \\", content)

  def test_patch_file_dry_run_via_registry(self):
    class MockSession:
      def __init__(self, sandbox):
        self.sandbox = sandbox
    session = MockSession(self.sandbox_dir)

    test_file = "test_reg_dry.txt"
    with open(os.path.join(self.sandbox_dir, test_file), "w") as f:
      f.write("Line A\nLine B\n")

    res = execute_tool(
      "patch_file",
      {
        "path": test_file,
        "search": "Line B",
        "replace": "Line B Replaced",
        "dry_run": True
      },
      session
    )
    self.assertIn("[DRY RUN]", res)
    self.assertIn("would apply successfully", res)
    with open(os.path.join(self.sandbox_dir, test_file), "r") as f:
      self.assertEqual(f.read(), "Line A\nLine B\n")

  def test_brightdata_web_search(self):
    from unittest.mock import patch, MagicMock

    fake_response = MagicMock()
    fake_response.json.return_value = {
      "organic": [
        {
          "title": "Bright Data Result",
          "link": "https://brightdata.com/example",
          "description": "Example description from Bright Data"
        }
      ]
    }
    fake_response.raise_for_status = MagicMock()

    env = {
      "BRIGHTDATA_API_KEY": "test-key-123",
      "BRIGHTDATA_ZONE": "custom_serp"
    }
    with patch.dict(os.environ, env, clear=True):
      with patch("requests.post", return_value=fake_response) as mock_post:
        res = tool_search_web("python programming", max_results=5)
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        self.assertEqual(call_args[0], "https://api.brightdata.com/request")
        self.assertEqual(call_kwargs["headers"]["Authorization"], "Bearer test-key-123")
        self.assertEqual(call_kwargs["json"]["zone"], "custom_serp")
        self.assertIn("Bright Data SERP API", res)
        self.assertIn("Bright Data Result", res)
        self.assertIn("https://brightdata.com/example", res)
        self.assertIn("Example description from Bright Data", res)

  def test_brightdata_web_search_empty(self):
    from unittest.mock import patch, MagicMock

    fake_response = MagicMock()
    fake_response.json.return_value = {"organic": []}
    fake_response.raise_for_status = MagicMock()

    env = {"BRIGHT_DATA_API_KEY": "test-key-fallback"}
    with patch.dict(os.environ, env, clear=True):
      with patch("requests.post", return_value=fake_response):
        res = tool_search_web("empty query")
        self.assertIn("[Bright Data SERP API] No results found.", res)



if __name__ == "__main__":
  unittest.main()
