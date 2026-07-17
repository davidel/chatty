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
  tool_patch_file
)

class TestSandboxOps(unittest.TestCase):
  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = tempfile.mkdtemp()

  def tearDown(self):
    os.chdir(self.old_cwd)
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

  def test_make_directory_multi(self):
    # Test multiple creation
    res = tool_make_directory(self.sandbox_dir, ["dir_a", "dir_b/dir_c"])
    self.assertIn("Successfully created directory 'dir_a'", res)
    self.assertIn("Successfully created directory 'dir_b/dir_c'", res)
    self.assertTrue(os.path.isdir(os.path.join(self.sandbox_dir, "dir_a")))
    self.assertTrue(os.path.isdir(os.path.join(self.sandbox_dir, "dir_b/dir_c")))

    # Test multi creation with mixed success/fail
    res = tool_make_directory(self.sandbox_dir, ["dir_a", "../outside_dir2"])
    self.assertIn("Directory 'dir_a' already exists.", res)
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

  def test_delete_file_multi(self):
    # Create multiple files
    f1 = os.path.join(self.sandbox_dir, "f1.txt")
    f2 = os.path.join(self.sandbox_dir, "f2.txt")
    with open(f1, "w") as f:
      f.write("1")
    with open(f2, "w") as f:
      f.write("2")

    # Delete multiple files
    res = tool_delete_file(self.sandbox_dir, ["f1.txt", "f2.txt"])
    self.assertIn("Successfully deleted file 'f1.txt'", res)
    self.assertIn("Successfully deleted file 'f2.txt'", res)
    self.assertFalse(os.path.exists(f1))
    self.assertFalse(os.path.exists(f2))

    # Test error propagation in list
    res = tool_delete_file(self.sandbox_dir, ["f1.txt", "../outside.txt"])
    self.assertIn("Error: Path 'f1.txt' does not exist.", res)
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

  def test_delete_directory_multi(self):
    # Create empty directories
    dir_x = os.path.join(self.sandbox_dir, "dir_x")
    dir_y = os.path.join(self.sandbox_dir, "dir_y")
    os.makedirs(dir_x)
    os.makedirs(dir_y)

    # Delete directories
    res = tool_delete_directory(self.sandbox_dir, ["dir_x", "dir_y"])
    self.assertIn("Successfully deleted directory 'dir_x'", res)
    self.assertIn("Successfully deleted directory 'dir_y'", res)
    self.assertFalse(os.path.exists(dir_x))
    self.assertFalse(os.path.exists(dir_y))

    # Test error propagation in list
    res = tool_delete_directory(self.sandbox_dir, ["dir_x", "../outside_dir"])
    self.assertIn("Error: Path 'dir_x' does not exist.", res)
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

    # Test binary file search directly
    bin_file = "binary.bin"
    with open(os.path.join(self.sandbox_dir, bin_file), "wb") as f:
      f.write(b"hello\x00pattern FF00 here\n")
    res = tool_search_grep(self.sandbox_dir, "FF00", "binary.bin")
    self.assertIn("is a binary file; searching is skipped", res)

    # Test recursive search ignores binary file
    res = tool_search_grep(self.sandbox_dir, "FF00", ".")
    self.assertIn("subdir/file2.txt: pattern FF00 here", res)
    self.assertNotIn("binary.bin", res)

  def test_tool_patch_file(self):
    test_file = "test_patch.py"
    file_path = os.path.join(self.sandbox_dir, test_file)
    with open(file_path, "w") as f:
      f.write("def func_a():\n  x = 1\n  return x\n\ndef func_b():\n  y = 2\n  return y\n")

    # Happy path: multiple Aider-style patches in one string
    patch = (
      "<<<<<<< SEARCH\n"
      "  x = 1\n"
      "  return x\n"
      "=======\n"
      "  x = 10\n"
      "  return x + 1\n"
      ">>>>>>> REPLACE\n"
      "<<<<<<< SEARCH\n"
      "  y = 2\n"
      "  return y\n"
      "=======\n"
      "  y = 20\n"
      "  return y + 2\n"
      ">>>>>>> REPLACE\n"
    )
    res = tool_patch_file(self.sandbox_dir, test_file, patch)
    self.assertIn("Successfully updated file", res)
    with open(file_path, "r") as f:
      content = f.read()
    self.assertIn("x = 10", content)
    self.assertIn("return x + 1", content)
    self.assertIn("y = 20", content)
    self.assertIn("return y + 2", content)

    # Indentation-shifting (fuzzy) matching
    with open(file_path, "w") as f:
      f.write("    def hello():\n        print(\"hi\")\n")
    
    # Search block has 2 spaces, file has 4/8 spaces
    fuzzy_patch = (
      "<<<<<<< SEARCH\n"
      "  def hello():\n"
      "      print(\"hi\")\n"
      "=======\n"
      "  def hello():\n"
      "      print(\"hello world\")\n"
      ">>>>>>> REPLACE\n"
    )
    res = tool_patch_file(self.sandbox_dir, test_file, fuzzy_patch)
    self.assertIn("Successfully updated file", res)
    with open(file_path, "r") as f:
      content_fuzzy = f.read()
    # It should have shifted the indent of the print statement to 8 spaces
    self.assertEqual(content_fuzzy, "    def hello():\n        print(\"hello world\")\n")

    # Error path: non-existent file
    res = tool_patch_file(self.sandbox_dir, "nonexistent.py", patch)
    self.assertIn("Error: File 'nonexistent.py' does not exist", res)

    # Error path: search block not found
    bad_patch = (
      "<<<<<<< SEARCH\n"
      "not found here\n"
      "=======\n"
      "something\n"
      ">>>>>>> REPLACE\n"
    )
    res = tool_patch_file(self.sandbox_dir, test_file, bad_patch)
    self.assertIn("SEARCH block not found in file", res)

    # Error path: duplicate search blocks (non-unique)
    with open(file_path, "w") as f:
      f.write("duplicate\nduplicate\n")
    dup_patch = (
      "<<<<<<< SEARCH\n"
      "duplicate\n"
      "=======\n"
      "replaced\n"
      ">>>>>>> REPLACE\n"
    )
    res = tool_patch_file(self.sandbox_dir, test_file, dup_patch)
    self.assertIn("SEARCH block is not unique", res)

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
      "  \"name\": \"patch_file\",\n"
      "  \"arguments\": {\n"
      "    \"path\": \"src/vector_regfile.sv\",\n"
      "    \"patch\": \"<<<<<<< SEARCH\\nrd_data = {a, b};\\n=======\\nassign a = {b, c};\\n>>>>>>> REPLACE\"\n"
      "  }\n"
      "}\n"
      "```"
    )
    
    parsed = session.extract_tool_calls_from_text(sample_text)
    self.assertEqual(len(parsed), 1)
    self.assertEqual(parsed[0]["function"]["name"], "patch_file")
    self.assertIn("src/vector_regfile.sv", parsed[0]["function"]["arguments"])

  def test_hex_dump(self):
    from chatty.tools.file_ops import tool_hex_dump
    
    bin_file = "test_dump.bin"
    # Write some bytes: hello\x00world\x01\x02\x03\x04\xff
    with open(os.path.join(self.sandbox_dir, bin_file), "wb") as f:
      f.write(b"hello\x00world\x01\x02\x03\x04\xff")
      
    # 1. Canonical format check (8-bit)
    res_canonical = tool_hex_dump(self.sandbox_dir, bin_file, start_offset=0, size=16, format="canonical")
    self.assertIn("68 65 6c 6c 6f 00 77 6f", res_canonical)
    self.assertIn("|hello.world.....|", res_canonical)
    self.assertTrue(res_canonical.startswith("00000000"))
    
    # 2. Hex format check (8-bit)
    res_hex = tool_hex_dump(self.sandbox_dir, bin_file, start_offset=0, size=16, format="hex")
    self.assertEqual(res_hex, "00000000: 0x68\n00000001: 0x65\n00000002: 0x6c\n00000003: 0x6c\n00000004: 0x6f\n00000005: 0x00\n00000006: 0x77\n00000007: 0x6f\n00000008: 0x72\n00000009: 0x6c\n0000000a: 0x64\n0000000b: 0x01\n0000000c: 0x02\n0000000d: 0x03\n0000000e: 0x04\n0000000f: 0xff")
    
    # 3. Raw format check (8-bit)
    res_raw = tool_hex_dump(self.sandbox_dir, bin_file, start_offset=0, size=16, format="raw")
    self.assertEqual(res_raw, "68656c6c6f00776f726c6401020304ff")
    
    # 4. Offset check
    res_offset = tool_hex_dump(self.sandbox_dir, bin_file, start_offset=6, size=5, format="raw")
    self.assertEqual(res_offset, "776f726c64")  # "world"

    # 5. 32-bit little endian unsigned (hello = 68 65 6c 6c -> 0x6c6c6568 = 1819043176)
    res_32_le = tool_hex_dump(self.sandbox_dir, bin_file, start_offset=0, size=8, format="dec", word_size=32, endian="little")
    self.assertIn("00000000: 1819043176", res_32_le)
    
    # 6. 32-bit big endian unsigned (hello = 68 65 6c 6c -> 0x68656c6c = 1751477356)
    res_32_be = tool_hex_dump(self.sandbox_dir, bin_file, start_offset=0, size=8, format="dec", word_size=32, endian="big")
    self.assertIn("00000000: 1751477356", res_32_be)

    # 7. Signed vs unsigned logic check
    # Write some bytes: \xff\xff\xff\xff (which is -1 signed, or 4294967295 unsigned)
    with open(os.path.join(self.sandbox_dir, "test_signed.bin"), "wb") as f:
      f.write(b"\xff\xff\xff\xff")
      
    res_unsigned = tool_hex_dump(self.sandbox_dir, "test_signed.bin", start_offset=0, size=4, format="dec", word_size=32, signed=False)
    self.assertIn("00000000: 4294967295", res_unsigned)

    res_signed = tool_hex_dump(self.sandbox_dir, "test_signed.bin", start_offset=0, size=4, format="dec", word_size=32, signed=True)
    self.assertIn("00000000: -1", res_signed)

  def test_tool_arguments_json_safety(self):
    import json
    from chatty.session import ChatbotSession
    session = ChatbotSession(
      provider="openrouter",
      model="mock",
      context_size=8192,
      sandbox=self.sandbox_dir
    )
    
    # 1. Test extract_tool_calls_from_text with list arguments
    sample_text = (
      "```json\n"
      "{\n"
      "  \"name\": \"run_command\",\n"
      "  \"arguments\": [\"git\", \"status\"]\n"
      "}\n"
      "```"
    )
    parsed = session.extract_tool_calls_from_text(sample_text)
    self.assertEqual(len(parsed), 1)
    args_str = parsed[0]["function"]["arguments"]
    # Ensure it's valid JSON
    parsed_args = json.loads(args_str)
    self.assertEqual(parsed_args, ["git", "status"])

    # 2. Test extract_tool_calls_from_text with a plain string that is not JSON
    sample_text_str = (
      "```json\n"
      "{\n"
      "  \"name\": \"run_command\",\n"
      "  \"arguments\": \"git status\"\n"
      "}\n"
      "```"
    )
    parsed_str = session.extract_tool_calls_from_text(sample_text_str)
    self.assertEqual(len(parsed_str), 1)
    args_str = parsed_str[0]["function"]["arguments"]
    parsed_args = json.loads(args_str)
    self.assertEqual(parsed_args, "git status")

    # 3. Test validation of invalid JSON arguments in tool_calls_accumulated list
    tool_calls = [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "run_command",
          "arguments": "{\"CommandLine\": \"git status\""  # invalid JSON (missing closing brace)
        }
      }
    ]
    
    # We can mock session run or just directly check the validation logic that we added:
    # Let's verify that when we sanitize the tool calls, it repairs it or falls back to valid JSON.
    for tc in tool_calls:
      func_obj = tc.get("function")
      if isinstance(func_obj, dict):
        t_args_raw = func_obj.get("arguments")
        if not isinstance(t_args_raw, str):
          func_obj["arguments"] = json.dumps(t_args_raw) if t_args_raw is not None else "{}"
        else:
          try:
            json.loads(t_args_raw)
          except Exception:
            try:
              from chatty.utils import repair_json
              repaired = repair_json(t_args_raw)
              json.loads(repaired)
              func_obj["arguments"] = repaired
            except Exception:
              func_obj["arguments"] = "{}"

    # Ensure the arguments were repaired to a valid JSON string
    self.assertNotEqual(tool_calls[0]["function"]["arguments"], "{\"CommandLine\": \"git status\"")
    parsed_repaired = json.loads(tool_calls[0]["function"]["arguments"])
    self.assertEqual(parsed_repaired.get("CommandLine"), "git status")


if __name__ == "__main__":
  unittest.main()


