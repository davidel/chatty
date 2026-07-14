import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.tools import (
  tool_write_file,
  tool_patch_file,
  tool_delete_file,
  tool_delete_directory
)
from chatty.backup import list_backups, restore_backup


class TestBackupAndRestore(unittest.TestCase):

  def setUp(self):
    self.sandbox_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.sandbox_dir)

  def test_write_and_patch_creates_backups(self):
    # 1. Write initial file (should not create backup as file did not exist yet)
    tool_write_file(self.sandbox_dir, "test.txt", "v1")
    backups_v1 = list_backups(self.sandbox_dir, "test.txt")
    self.assertEqual(len(backups_v1), 0)

    # 2. Write v2 (should backup v1)
    tool_write_file(self.sandbox_dir, "test.txt", "v2")
    backups_v2 = list_backups(self.sandbox_dir, "test.txt")
    self.assertEqual(len(backups_v2), 1)

    # 3. Patch file to v3 (should backup v2)
    patch_v3 = """<<<<<<< SEARCH
v2
=======
v3
>>>>>>> REPLACE"""
    tool_patch_file(self.sandbox_dir, "test.txt", patch_v3)
    backups_v3 = list_backups(self.sandbox_dir, "test.txt")
    self.assertEqual(len(backups_v3), 2)

    # 4. Restore v1
    # The oldest backup (index 1 in list backups, since it sorts newest first) is v1
    res = restore_backup(self.sandbox_dir, "test.txt", backups_v3[1][0])
    self.assertIn("Successfully restored", res)
    with open(os.path.join(self.sandbox_dir, "test.txt"), "r") as f:
      self.assertEqual(f.read(), "v1")

  def test_delete_file_creates_backup(self):
    tool_write_file(self.sandbox_dir, "to_delete.txt", "delete content")
    tool_delete_file(self.sandbox_dir, "to_delete.txt")
    backups = list_backups(self.sandbox_dir, "to_delete.txt")
    self.assertEqual(len(backups), 1)

    # Restore the deleted file
    restore_backup(self.sandbox_dir, "to_delete.txt")
    self.assertTrue(os.path.exists(os.path.join(self.sandbox_dir, "to_delete.txt")))
    with open(os.path.join(self.sandbox_dir, "to_delete.txt"), "r") as f:
      self.assertEqual(f.read(), "delete content")

  def test_delete_dir_creates_backups(self):
    os.makedirs(os.path.join(self.sandbox_dir, "subdir"))
    tool_write_file(self.sandbox_dir, "subdir/file1.txt", "content1")
    tool_write_file(self.sandbox_dir, "subdir/file2.txt", "content2")

    tool_delete_directory(self.sandbox_dir, "subdir", recursive=True)

    backups1 = list_backups(self.sandbox_dir, "subdir/file1.txt")
    backups2 = list_backups(self.sandbox_dir, "subdir/file2.txt")
    self.assertEqual(len(backups1), 1)
    self.assertEqual(len(backups2), 1)

  def test_gitignore_auto_ignores_chatty(self):
    tool_write_file(self.sandbox_dir, "file.txt", "v1")
    tool_write_file(self.sandbox_dir, "file.txt", "v2")
    gitignore_path = os.path.join(self.sandbox_dir, ".gitignore")
    self.assertTrue(os.path.exists(gitignore_path))
    with open(gitignore_path, "r", encoding="utf-8") as f:
      content = f.read()
    self.assertIn(".chatty/", content)


if __name__ == "__main__":
  unittest.main()
