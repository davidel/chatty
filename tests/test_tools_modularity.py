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
    self.sandbox_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.sandbox_dir)

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


if __name__ == "__main__":
  unittest.main()
