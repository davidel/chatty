import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.cli import (
  tool_move_file,
  tool_copy_file,
  tool_delete_file,
  tool_make_directory,
  tool_get_file_info
)

class TestSandboxOps(unittest.TestCase):
  def setUp(self):
    self.sandbox_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.sandbox_dir)

  def test_make_directory(self):
    # Test simple creation
    res = tool_make_directory(self.sandbox_dir, "test_dir")
    self.assertIn("Successfully created directory", res)
    self.assertTrue(os.path.isdir(os.path.join(self.sandbox_dir, "test_dir")))

    # Test nested creation
    res = tool_make_directory(self.sandbox_dir, "parent/child/grandchild")
    self.assertIn("Successfully created directory", res)
    self.assertTrue(os.path.isdir(os.path.join(self.sandbox_dir, "parent/child/grandchild")))

    # Test directory already exists
    res = tool_make_directory(self.sandbox_dir, "test_dir")
    self.assertIn("already exists", res)

    # Test path exists but is a file
    file_path = os.path.join(self.sandbox_dir, "some_file.txt")
    with open(file_path, "w") as f:
      f.write("test")
    res = tool_make_directory(self.sandbox_dir, "some_file.txt")
    self.assertIn("Error", res)

    # Test sandbox traversal
    res = tool_make_directory(self.sandbox_dir, "../outside_dir")
    self.assertIn("Access Denied", res)

  def test_copy_file(self):
    # Create source file
    src_file = os.path.join(self.sandbox_dir, "src.txt")
    with open(src_file, "w") as f:
      f.write("hello world")

    # Copy file
    res = tool_copy_file(self.sandbox_dir, "src.txt", "dest.txt")
    self.assertIn("Successfully copied", res)
    dest_path = os.path.join(self.sandbox_dir, "dest.txt")
    self.assertTrue(os.path.exists(dest_path))
    with open(dest_path, "r") as f:
      self.assertEqual(f.read(), "hello world")

    # Copy directory recursively
    dir_src = os.path.join(self.sandbox_dir, "dir_src")
    os.makedirs(os.path.join(dir_src, "sub"))
    with open(os.path.join(dir_src, "sub/file.txt"), "w") as f:
      f.write("nested")
    
    res = tool_copy_file(self.sandbox_dir, "dir_src", "dir_dest")
    self.assertIn("Successfully copied", res)
    self.assertTrue(os.path.isdir(os.path.join(self.sandbox_dir, "dir_dest/sub")))
    with open(os.path.join(self.sandbox_dir, "dir_dest/sub/file.txt"), "r") as f:
      self.assertEqual(f.read(), "nested")

    # Test copy non-existent file
    res = tool_copy_file(self.sandbox_dir, "nonexistent.txt", "dest2.txt")
    self.assertIn("does not exist", res)

    # Test sandbox traversal (src outside)
    res = tool_copy_file(self.sandbox_dir, "../outside.txt", "local.txt")
    self.assertIn("Access Denied", res)

    # Test sandbox traversal (dest outside)
    res = tool_copy_file(self.sandbox_dir, "src.txt", "../outside.txt")
    self.assertIn("Access Denied", res)

  def test_move_file(self):
    # Create source file
    src_file = os.path.join(self.sandbox_dir, "src.txt")
    with open(src_file, "w") as f:
      f.write("move me")

    # Move file
    res = tool_move_file(self.sandbox_dir, "src.txt", "dest.txt")
    self.assertIn("Successfully moved", res)
    self.assertFalse(os.path.exists(src_file))
    dest_path = os.path.join(self.sandbox_dir, "dest.txt")
    self.assertTrue(os.path.exists(dest_path))
    with open(dest_path, "r") as f:
      self.assertEqual(f.read(), "move me")

    # Move directory
    dir_src = os.path.join(self.sandbox_dir, "dir_src")
    os.makedirs(os.path.join(dir_src, "sub"))
    with open(os.path.join(dir_src, "sub/file.txt"), "w") as f:
      f.write("nested")

    res = tool_move_file(self.sandbox_dir, "dir_src", "dir_dest")
    self.assertIn("Successfully moved", res)
    self.assertFalse(os.path.exists(dir_src))
    self.assertTrue(os.path.isdir(os.path.join(self.sandbox_dir, "dir_dest/sub")))

    # Test move non-existent file
    res = tool_move_file(self.sandbox_dir, "nonexistent.txt", "dest2.txt")
    self.assertIn("does not exist", res)

    # Test sandbox traversal
    res = tool_move_file(self.sandbox_dir, "dest.txt", "../outside.txt")
    self.assertIn("Access Denied", res)

  def test_delete_file(self):
    # Create file to delete
    file_path = os.path.join(self.sandbox_dir, "delete_me.txt")
    with open(file_path, "w") as f:
      f.write("delete")

    # Delete file
    res = tool_delete_file(self.sandbox_dir, "delete_me.txt")
    self.assertIn("Successfully deleted file", res)
    self.assertFalse(os.path.exists(file_path))

    # Create directory to delete
    dir_path = os.path.join(self.sandbox_dir, "delete_dir")
    os.makedirs(dir_path)

    # Delete directory
    res = tool_delete_file(self.sandbox_dir, "delete_dir")
    self.assertIn("Successfully deleted directory", res)
    self.assertFalse(os.path.exists(dir_path))

    # Test delete non-existent file
    res = tool_delete_file(self.sandbox_dir, "nonexistent.txt")
    self.assertIn("does not exist", res)

    # Test sandbox traversal
    res = tool_delete_file(self.sandbox_dir, "../outside.txt")
    self.assertIn("Access Denied", res)

  def test_get_file_info(self):
    # Test non-existent path
    res = tool_get_file_info(self.sandbox_dir, "nonexistent.txt")
    self.assertIn("does not exist", res)

    # Test directory info (should not have "Lines:")
    dir_name = "test_dir"
    os.makedirs(os.path.join(self.sandbox_dir, dir_name))
    res = tool_get_file_info(self.sandbox_dir, dir_name)
    self.assertIn("Type: Directory", res)
    self.assertNotIn("Lines:", res)

    # Test empty text file info
    empty_file = "empty.txt"
    with open(os.path.join(self.sandbox_dir, empty_file), "w") as f:
      pass
    res = tool_get_file_info(self.sandbox_dir, empty_file)
    self.assertIn("Type: File", res)
    self.assertIn("Lines: 0", res)

    # Test text file with standard newlines
    text_file = "text.txt"
    with open(os.path.join(self.sandbox_dir, text_file), "w") as f:
      f.write("line1\nline2\nline3\n")
    res = tool_get_file_info(self.sandbox_dir, text_file)
    self.assertIn("Lines: 3", res)

    # Test text file without trailing newline
    text_file_no_trail = "text_no_trail.txt"
    with open(os.path.join(self.sandbox_dir, text_file_no_trail), "w") as f:
      f.write("line1\nline2")
    res = tool_get_file_info(self.sandbox_dir, text_file_no_trail)
    self.assertIn("Lines: 2", res)

    # Test binary file (should not have "Lines:")
    bin_file = "binary.bin"
    with open(os.path.join(self.sandbox_dir, bin_file), "wb") as f:
      f.write(b"hello\x00world\n")
    res = tool_get_file_info(self.sandbox_dir, bin_file)
    self.assertIn("Type: File", res)
    self.assertNotIn("Lines:", res)


if __name__ == "__main__":
  unittest.main()
