import unittest
import os
import shutil
import tempfile
from chatty.tools.code_intel import SymbolExtractor


class TestCodeIntel(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.extractor = SymbolExtractor(self.temp_dir)

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_python_ast_extractor(self):
    code = """
class MyClass:
  def my_method(self):
    pass

def my_func():
  pass
"""
    file_path = os.path.join(self.temp_dir, "test_file.py")
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(code)
    symbols = self.extractor.get_outline("test_file.py")
    self.assertEqual(len(symbols), 3)
    self.assertEqual(symbols[0]["name"], "MyClass")
    self.assertEqual(symbols[0]["type"], "class")
    self.assertEqual(symbols[1]["name"], "my_method")
    self.assertEqual(symbols[1]["type"], "method")
    self.assertEqual(symbols[1]["parent"], "MyClass")
    self.assertEqual(symbols[2]["name"], "my_func")
    self.assertEqual(symbols[2]["type"], "function")
    self.assertIsNone(symbols[2]["parent"])

  def test_regex_fallback(self):
    code = """
class SomeClass {
};
function go() {
}
"""
    file_path = os.path.join(self.temp_dir, "test_file.js")
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(code)
    symbols = self.extractor.get_outline("test_file.js")
    self.assertEqual(len(symbols), 2)
    self.assertEqual(symbols[0]["name"], "SomeClass")
    self.assertEqual(symbols[0]["type"], "class")
    self.assertEqual(symbols[1]["name"], "go")
    self.assertEqual(symbols[1]["type"], "function")

  def test_json_caching(self):
    code = "def check(): pass"
    file_path = os.path.join(self.temp_dir, "cached_file.py")
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(code)
    symbols = self.extractor.get_outline("cached_file.py")
    self.assertEqual(len(symbols), 1)
    cache_file = os.path.join(self.temp_dir, ".chatty", "symbol_cache.json")
    self.assertTrue(os.path.exists(cache_file))

  def test_find_symbol(self):
    code = "class SearchTarget:\n  def method(self):\n    pass"
    file_path = os.path.join(self.temp_dir, "search_file.py")
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(code)
    matches = self.extractor.find_symbol("SearchTarget")
    self.assertEqual(len(matches), 1)
    self.assertEqual(matches[0]["path"], "search_file.py")
    self.assertEqual(matches[0]["type"], "class")
