import os
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.safety import is_path_ignored


class TestGitIgnoreMatching(unittest.TestCase):

  def test_basic_matching(self):
    # Pattern without slash matches file or dir at any level
    patterns = ["foo", "*.log"]
    self.assertTrue(is_path_ignored("foo", patterns))
    self.assertTrue(is_path_ignored("dir/foo", patterns))
    self.assertTrue(is_path_ignored("dir/subdir/foo", patterns))
    self.assertTrue(is_path_ignored("bar.log", patterns))
    self.assertTrue(is_path_ignored("dir/bar.log", patterns))
    self.assertFalse(is_path_ignored("foobar", patterns))
    self.assertFalse(is_path_ignored("log", patterns))

  def test_directory_only_matching(self):
    # Pattern with trailing slash matches directories only
    patterns = ["build/"]
    # If build is checked as a file, it shouldn't match
    self.assertFalse(is_path_ignored("build", patterns, is_dir=False))
    # If build is checked as a directory, it matches
    self.assertTrue(is_path_ignored("build", patterns, is_dir=True))
    # Any subpath of build matches (since parent directory 'build' is checked as a directory prefix)
    self.assertTrue(is_path_ignored("build/file.txt", patterns, is_dir=False))
    self.assertTrue(is_path_ignored("dir/build/file.txt", patterns, is_dir=False))
    self.assertTrue(is_path_ignored("dir/build", patterns, is_dir=True))
    self.assertFalse(is_path_ignored("dir/build", patterns, is_dir=False))

  def test_anchored_matching(self):
    # Pattern with leading slash matches relative to the root only
    patterns = ["/foo"]
    self.assertTrue(is_path_ignored("foo", patterns))
    self.assertTrue(is_path_ignored("foo/bar.txt", patterns))
    self.assertFalse(is_path_ignored("dir/foo", patterns))

  def test_inner_slash_anchoring(self):
    # Pattern with inner slash (but no leading slash) is also anchored to root
    patterns = ["dir/foo"]
    self.assertTrue(is_path_ignored("dir/foo", patterns))
    self.assertTrue(is_path_ignored("dir/foo/bar.txt", patterns))
    self.assertFalse(is_path_ignored("subdir/dir/foo", patterns))

  def test_double_asterisks(self):
    # **/foo matches foo at any level
    patterns1 = ["**/foo"]
    self.assertTrue(is_path_ignored("foo", patterns1))
    self.assertTrue(is_path_ignored("dir/foo", patterns1))
    self.assertTrue(is_path_ignored("dir/subdir/foo", patterns1))

    # foo/** matches contents of foo
    patterns2 = ["foo/**"]
    self.assertTrue(is_path_ignored("foo/bar", patterns2))
    self.assertTrue(is_path_ignored("foo/bar/baz.txt", patterns2))
    self.assertFalse(is_path_ignored("foo", patterns2))

    # foo/**/bar matches foo/bar, foo/a/bar, foo/a/b/bar, etc.
    patterns3 = ["foo/**/bar"]
    self.assertTrue(is_path_ignored("foo/bar", patterns3))
    self.assertTrue(is_path_ignored("foo/a/bar", patterns3))
    self.assertTrue(is_path_ignored("foo/a/b/bar", patterns3))
    self.assertTrue(is_path_ignored("foo/a/b/bar/file.txt", patterns3))
    self.assertFalse(is_path_ignored("foo/barry", patterns3))

  def test_negation(self):
    # Later negations override earlier exclusions
    patterns = ["*.log", "!important.log"]
    self.assertTrue(is_path_ignored("file.log", patterns))
    self.assertFalse(is_path_ignored("important.log", patterns))

    # Negation cannot re-include a file if its parent directory is excluded
    patterns2 = ["dir/", "!dir/file.txt"]
    self.assertTrue(is_path_ignored("dir/file.txt", patterns2))

    # Correct way to exclude dir except file.txt
    patterns3 = ["dir/*", "!dir/file.txt"]
    self.assertFalse(is_path_ignored("dir/file.txt", patterns3))
    self.assertTrue(is_path_ignored("dir/other.txt", patterns3))


if __name__ == "__main__":
  unittest.main()
