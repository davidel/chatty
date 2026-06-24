import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.tools import (
  tool_move_file,
  tool_copy_file,
  tool_delete_file,
  tool_delete_directory,
  tool_make_directory,
  tool_get_file_info,
  tool_search_grep,
  tool_multi_patch
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

    # Create directory to test failure
    dir_path = os.path.join(self.sandbox_dir, "delete_dir")
    os.makedirs(dir_path)

    # Test delete file on directory (should fail)
    res = tool_delete_file(self.sandbox_dir, "delete_dir")
    self.assertIn("is a directory", res)
    self.assertTrue(os.path.exists(dir_path))

    # Delete file
    res = tool_delete_file(self.sandbox_dir, "delete_me.txt")
    self.assertIn("Successfully deleted file", res)
    self.assertFalse(os.path.exists(file_path))

    # Test delete non-existent file
    res = tool_delete_file(self.sandbox_dir, "nonexistent.txt")
    self.assertIn("does not exist", res)

    # Test sandbox traversal
    res = tool_delete_file(self.sandbox_dir, "../outside.txt")
    self.assertIn("Access Denied", res)

  def test_delete_directory(self):
    # Create empty directory to delete
    dir_path = os.path.join(self.sandbox_dir, "empty_dir")
    os.makedirs(dir_path)

    # Create non-empty directory
    nested_dir_path = os.path.join(self.sandbox_dir, "nested_dir")
    os.makedirs(nested_dir_path)
    file_path = os.path.join(nested_dir_path, "file.txt")
    with open(file_path, "w") as f:
      f.write("hello")

    # Test delete directory on a file (should fail)
    res = tool_delete_directory(self.sandbox_dir, "nested_dir/file.txt")
    self.assertIn("is a file", res)
    self.assertTrue(os.path.exists(file_path))

    # Test delete non-empty directory without recursive (should fail)
    res = tool_delete_directory(self.sandbox_dir, "nested_dir", recursive=False)
    self.assertIn("is not empty", res)
    self.assertTrue(os.path.exists(nested_dir_path))

    # Test delete empty directory
    res = tool_delete_directory(self.sandbox_dir, "empty_dir")
    self.assertIn("Successfully deleted directory", res)
    self.assertFalse(os.path.exists(dir_path))

    # Test delete non-empty directory with recursive=True
    res = tool_delete_directory(self.sandbox_dir, "nested_dir", recursive=True)
    self.assertIn("Successfully deleted directory", res)
    self.assertFalse(os.path.exists(nested_dir_path))

    # Test delete non-existent directory
    res = tool_delete_directory(self.sandbox_dir, "nonexistent_dir")
    self.assertIn("does not exist", res)

    # Test sandbox traversal
    res = tool_delete_directory(self.sandbox_dir, "../outside_dir")
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

  def test_search_grep(self):
    # Create test directory and files
    os.makedirs(os.path.join(self.sandbox_dir, "subdir"))
    
    file1 = "file1.txt"
    with open(os.path.join(self.sandbox_dir, file1), "w") as f:
      f.write("hello world\nline with pattern 0F10\nend of file\n")
      
    file2 = "subdir/file2.txt"
    with open(os.path.join(self.sandbox_dir, file2), "w") as f:
      f.write("another line\npattern FF00 here\n")

    # Test recursive search without line numbers (default)
    res = tool_search_grep(self.sandbox_dir, "0F10|FF00", ".")
    self.assertIn("file1.txt: line with pattern 0F10", res)
    self.assertIn("subdir/file2.txt: pattern FF00 here", res)

    # Test recursive search with line numbers
    res = tool_search_grep(self.sandbox_dir, "0F10|FF00", ".", line_numbers=True)
    self.assertIn("file1.txt:2: line with pattern 0F10", res)
    self.assertIn("subdir/file2.txt:2: pattern FF00 here", res)

    # Test search in specific file with line numbers
    res = tool_search_grep(self.sandbox_dir, "0F10|FF00", "file1.txt", line_numbers=True)
    self.assertIn("file1.txt:2: line with pattern 0F10", res)
    self.assertNotIn("subdir/file2.txt", res)

    # Test non-existent path
    res = tool_search_grep(self.sandbox_dir, "pattern", "nonexistent.txt")
    self.assertIn("Error: Path 'nonexistent.txt' does not exist.", res)

    # Test no matches
    res = tool_search_grep(self.sandbox_dir, "NOMATCH", "file1.txt")
    self.assertEqual(res, "No matches found.")

  def test_multi_patch(self):
    test_file = "test_multi.py"
    file_path = os.path.join(self.sandbox_dir, test_file)
    with open(file_path, "w") as f:
      f.write("def func_a():\n  x = 1\n  return x\n\ndef func_b():\n  y = 2\n  return y\n")

    # Happy path: non-overlapping, unique patches
    patches = [
      {"search": "  x = 1\n  return x", "replace": "  x = 10\n  return x + 1"},
      {"search": "  y = 2\n  return y", "replace": "  y = 20\n  return y + 2"}
    ]
    res = tool_multi_patch(self.sandbox_dir, test_file, patches)
    self.assertIn("Successfully updated file", res)
    with open(file_path, "r") as f:
      content = f.read()
    self.assertIn("x = 10", content)
    self.assertIn("return x + 1", content)
    self.assertIn("y = 20", content)
    self.assertIn("return y + 2", content)

    # Error path: non-existent file
    res = tool_multi_patch(self.sandbox_dir, "nonexistent.py", patches)
    self.assertIn("Error: File 'nonexistent.py' does not exist", res)

    # Error path: search block not found
    bad_patches = [{"search": "not found here", "replace": "something"}]
    res = tool_multi_patch(self.sandbox_dir, test_file, bad_patches)
    self.assertIn("Error: The search block in patch 1 was not found", res)

    # Error path: duplicate search blocks (non-unique)
    with open(file_path, "w") as f:
      f.write("duplicate\nduplicate\n")
    dup_patches = [{"search": "duplicate", "replace": "replaced"}]
    res = tool_multi_patch(self.sandbox_dir, test_file, dup_patches)
    self.assertIn("Error: Found 2 occurrences of the search block in patch 1", res)

    # Error path: overlapping patches
    with open(file_path, "w") as f:
      f.write("line 1\nline 2\nline 3\n")
    overlap_patches = [
      {"search": "line 1\nline 2", "replace": "replaced 1"},
      {"search": "line 2\nline 3", "replace": "replaced 2"}
    ]
    res = tool_multi_patch(self.sandbox_dir, test_file, overlap_patches)
    self.assertIn("Error: Overlapping patches detected", res)

  def test_make_file_preview_small(self):
    from chatty.tools import make_file_preview
    test_file = "preview_small.txt"
    file_path = os.path.join(self.sandbox_dir, test_file)
    content = "line1\nline2\nline3\n"
    with open(file_path, "w") as f:
      f.write(content)
    
    res = make_file_preview(file_path, [(1, 1)])
    self.assertIn("File 'preview_small.txt' now has 3 lines:", res)
    self.assertIn("1: line1", res)
    self.assertIn("2: line2", res)

  def test_make_file_preview_large(self):
    from chatty.tools import make_file_preview
    test_file = "preview_large.txt"
    file_path = os.path.join(self.sandbox_dir, test_file)
    content = "".join(f"line{i}\n" for i in range(1, 150))
    with open(file_path, "w") as f:
      f.write(content)
      
    res = make_file_preview(file_path, [(50, 52)], context_lines=2)
    self.assertIn("now has 149 lines", res)
    self.assertIn("... (lines 1-47 truncated) ...", res)
    self.assertIn("48: line48", res)
    self.assertIn("52: line52", res)
    self.assertIn("54: line54", res)
    self.assertIn("... (lines 55-149 truncated) ...", res)

  def test_tool_edit_lines_with_preview(self):
    from chatty.tools import tool_edit_lines
    test_file = "edit_preview.txt"
    file_path = os.path.join(self.sandbox_dir, test_file)
    with open(file_path, "w") as f:
      f.write("a\nb\nc\nd\ne\n")
      
    res = tool_edit_lines(self.sandbox_dir, test_file, 2, 4, "x\ny")
    self.assertIn("Successfully updated file", res)
    self.assertIn("now has 4 lines", res)
    self.assertIn("2: x\n3: y", res)

  def test_tool_patch_file_with_preview(self):
    from chatty.tools import tool_patch_file
    test_file = "patch_preview.txt"
    file_path = os.path.join(self.sandbox_dir, test_file)
    with open(file_path, "w") as f:
      f.write("a\nb\nc\nd\ne\n")
      
    res = tool_patch_file(self.sandbox_dir, test_file, "b\nc\nd", "x\ny")
    self.assertIn("Successfully updated file", res)
    self.assertIn("now has 4 lines", res)
    self.assertIn("2: x\n3: y", res)

  def test_tool_multi_edit_lines(self):
    from chatty.tools import tool_multi_edit_lines
    test_file = "multi_edit.txt"
    file_path = os.path.join(self.sandbox_dir, test_file)
    with open(file_path, "w") as f:
      f.write("line1\nline2\nline3\nline4\nline5\n")
      
    edits = [
      {"start_line": 2, "end_line": 2, "replacement": "new2"},
      {"start_line": 4, "end_line": 4, "replacement": "new4_a\nnew4_b"}
    ]
    res = tool_multi_edit_lines(self.sandbox_dir, test_file, edits)
    self.assertIn("Successfully updated file", res)
    with open(file_path, "r") as f:
      content = f.read()
    self.assertEqual(content, "line1\nnew2\nline3\nnew4_a\nnew4_b\nline5\n")

  def test_extract_tool_calls_from_text(self):
    from chatty.session import ChatbotSession
    # Create mock session
    session = ChatbotSession(
      provider="openrouter",
      model="mock",
      context_size=8192,
      sandbox=self.sandbox_dir
    )
    
    # Test text containing SystemVerilog code with curly braces and nested JSON with newlines
    sample_text = (
      "Let's fix always_comb begin rd_data = {a, b}; end.\n"
      "Here is the tool call:\n"
      "```json\n"
      "{\n"
      "  \"name\": \"multi_edit_lines\",\n"
      "  \"arguments\": {\n"
      "    \"path\": \"src/vector_regfile.sv\",\n"
      "    \"edits\": [\n"
      "      {\n"
      "        \"start_line\": 20,\n"
      "        \"end_line\": 25,\n"
      "        \"replacement\": \"assign a = {b, c};\"\n"
      "      }\n"
      "    ]\n"
      "  }\n"
      "}\n"
      "```"
    )
    
    parsed = session.extract_tool_calls_from_text(sample_text)
    self.assertEqual(len(parsed), 1)
    self.assertEqual(parsed[0]["function"]["name"], "multi_edit_lines")
    self.assertIn("src/vector_regfile.sv", parsed[0]["function"]["arguments"])


if __name__ == "__main__":
  unittest.main()


