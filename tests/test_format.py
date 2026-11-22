import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.tools import tool_format_file


class TestFormatFile(unittest.TestCase):
  def setUp(self):
    self.sandbox_dir = self.enterContext(tempfile.TemporaryDirectory())

  def test_format_json(self):
    # Create unformatted JSON
    json_path = os.path.join(self.sandbox_dir, "test.json")
    with open(json_path, "w") as f:
      f.write('{"a":1,"b":   [2, 3]}')

    # Run formatter
    result = tool_format_file(self.sandbox_dir, "test.json")
    self.assertIn("Successfully formatted", result)

    # Verify formatted content
    with open(json_path, "r") as f:
      content = f.read()
    expected = '{\n  "a": 1,\n  "b": [\n    2,\n    3\n  ]\n}\n'
    self.assertEqual(content, expected)

  def test_format_yaml(self):
    # Create unformatted YAML
    yaml_path = os.path.join(self.sandbox_dir, "test.yaml")
    with open(yaml_path, "w") as f:
      f.write("a:   1\nb: [2, 3]")

    # Run formatter
    result = tool_format_file(self.sandbox_dir, "test.yaml")
    self.assertIn("Successfully formatted", result)

    with open(yaml_path, "r") as f:
      content = f.read()
    self.assertIn("a: 1", content)

  def test_format_cpp_using_clang_format(self):
    # Create unformatted C++
    cpp_path = os.path.join(self.sandbox_dir, "test.cpp")
    with open(cpp_path, "w") as f:
      f.write("int main() {int a=1+2;return 0;}")

    # Run formatter
    result = tool_format_file(self.sandbox_dir, "test.cpp")

    # Since clang-format is installed on the system, it should format
    if "is not installed" not in result:
      self.assertIn("Successfully formatted", result)
      with open(cpp_path, "r") as f:
        content = f.read()
      self.assertIn("int a = 1 + 2;", content)

  def test_format_sv_using_clang_format(self):
    # Create unformatted SystemVerilog
    sv_path = os.path.join(self.sandbox_dir, "test.sv")
    with open(sv_path, "w") as f:
      f.write("module test; logic clk; always_ff @(posedge clk) begin a<=b; end endmodule")

    # Run formatter
    result = tool_format_file(self.sandbox_dir, "test.sv")

    # Since clang-format is installed on the system, it should format
    if "is not installed" not in result:
      self.assertIn("Successfully formatted", result)
      with open(sv_path, "r") as f:
        content = f.read()
      self.assertIn("always_ff @(posedge clk) begin", content)

  def test_format_sv_with_config_path(self):
    # Create unformatted SystemVerilog
    sv_path = os.path.join(self.sandbox_dir, "test.sv")
    with open(sv_path, "w") as f:
      f.write("module test; logic clk; always_ff @(posedge clk) begin a<=b; end endmodule")

    # Create config file
    config_path = os.path.join(self.sandbox_dir, ".my-custom-format")
    with open(config_path, "w") as f:
      f.write("Language: Verilog\nIndentWidth: 4\n")

    # Run formatter with specified tool and config path
    result = tool_format_file(self.sandbox_dir, "test.sv", formatter="clang-format", config_path=".my-custom-format")

    # Since clang-format is installed on the system, it should format
    if "is not installed" not in result:
      self.assertIn("Successfully formatted", result)
      with open(sv_path, "r") as f:
        content = f.read()
      # It should have 4-space indentation for the body
      self.assertIn("    always_ff @(posedge clk)", content)

  def test_autodetect_style_yapf(self):
    # Create .style.yapf setting 2-space indentation
    style_path = os.path.join(self.sandbox_dir, ".style.yapf")
    with open(style_path, "w") as f:
      f.write("[style]\nbased_on_style = pep8\nindent_width = 2\n")

    # Create Python file with unformatted/4-space indentation
    py_path = os.path.join(self.sandbox_dir, "module.py")
    with open(py_path, "w") as f:
      f.write("def sample_fn():\n    val = 10\n    return val\n")

    # Call format_file WITHOUT specifying formatter or config_path
    result = tool_format_file(self.sandbox_dir, "module.py")

    if shutil.which("yapf"):
      self.assertIn("Successfully formatted", result)
      self.assertIn("yapf", result)
      self.assertIn(".style.yapf", result)
      with open(py_path, "r") as f:
        content = f.read()
      # Verify that 2-space indentation from .style.yapf was used
      self.assertIn("  val = 10\n  return val", content)

  def test_autodetect_style_yapf_in_parent_dir(self):
    # Create .style.yapf at root
    style_path = os.path.join(self.sandbox_dir, ".style.yapf")
    with open(style_path, "w") as f:
      f.write("[style]\nbased_on_style = pep8\nindent_width = 2\n")

    # Create nested subdirectory and file
    nested_dir = os.path.join(self.sandbox_dir, "pkg", "sub")
    os.makedirs(nested_dir, exist_ok=True)
    py_path = os.path.join(nested_dir, "nested.py")
    with open(py_path, "w") as f:
      f.write("def nested_fn():\n    val = 20\n    return val\n")

    # Call format_file on nested file without specifying formatter or config_path
    result = tool_format_file(self.sandbox_dir, "pkg/sub/nested.py")

    if shutil.which("yapf"):
      self.assertIn("Successfully formatted", result)
      self.assertIn("yapf", result)
      with open(py_path, "r") as f:
        content = f.read()
      self.assertIn("  val = 20\n  return val", content)

  def test_autodetect_clang_format(self):
    # Create .clang-format
    clang_config = os.path.join(self.sandbox_dir, ".clang-format")
    with open(clang_config, "w") as f:
      f.write("BasedOnStyle: LLVM\nIndentWidth: 4\n")

    cpp_path = os.path.join(self.sandbox_dir, "test.cpp")
    with open(cpp_path, "w") as f:
      f.write("int main() {int a=1+2;return 0;}")

    # Call without formatter or config_path
    result = tool_format_file(self.sandbox_dir, "test.cpp")

    if shutil.which("clang-format"):
      self.assertIn("Successfully formatted", result)
      self.assertIn("clang-format", result)
      self.assertIn(".clang-format", result)

  def test_formatter_specified_autodetects_config(self):
    # When formatter="yapf" is explicitly passed, config_path should be autodetected
    style_path = os.path.join(self.sandbox_dir, ".style.yapf")
    with open(style_path, "w") as f:
      f.write("[style]\nbased_on_style = pep8\nindent_width = 2\n")

    py_path = os.path.join(self.sandbox_dir, "explicit.py")
    with open(py_path, "w") as f:
      f.write("def explicit_fn():\n    x = 1\n    return x\n")

    result = tool_format_file(self.sandbox_dir, "explicit.py", formatter="yapf")
    if shutil.which("yapf"):
      self.assertIn("Successfully formatted", result)
      self.assertIn(".style.yapf", result)
      with open(py_path, "r") as f:
        content = f.read()
      self.assertIn("  x = 1\n  return x", content)

  def test_format_glob_json(self):
    p1 = os.path.join(self.sandbox_dir, "file1.json")
    p2 = os.path.join(self.sandbox_dir, "file2.json")
    with open(p1, "w") as f:
      f.write('{"k": 1}')
    with open(p2, "w") as f:
      f.write('{"k":   2}')

    result = tool_format_file(self.sandbox_dir, "*.json")
    self.assertIn("Formatted 2 of 2 file(s)", result)
    self.assertIn("file1.json", result)
    self.assertIn("file2.json", result)

    with open(p1, "r") as f:
      self.assertEqual(f.read(), '{\n  "k": 1\n}\n')
    with open(p2, "r") as f:
      self.assertEqual(f.read(), '{\n  "k": 2\n}\n')

  def test_format_recursive_glob(self):
    sub = os.path.join(self.sandbox_dir, "sub", "nested")
    os.makedirs(sub, exist_ok=True)
    p1 = os.path.join(self.sandbox_dir, "top.json")
    p2 = os.path.join(sub, "inner.json")
    with open(p1, "w") as f:
      f.write('{"top":1}')
    with open(p2, "w") as f:
      f.write('{"inner":2}')

    result = tool_format_file(self.sandbox_dir, "**/*.json")
    self.assertIn("Formatted 2 of 2 file(s)", result)
    self.assertIn("top.json", result)
    self.assertIn("sub/nested/inner.json", result)

  def test_format_directory(self):
    sub = os.path.join(self.sandbox_dir, "mydir")
    os.makedirs(sub, exist_ok=True)
    p1 = os.path.join(sub, "d1.json")
    p2 = os.path.join(sub, "d2.json")
    with open(p1, "w") as f:
      f.write('{"d1":1}')
    with open(p2, "w") as f:
      f.write('{"d2":2}')

    result = tool_format_file(self.sandbox_dir, "mydir")
    self.assertIn("Formatted 2 of 2 file(s)", result)
    self.assertIn("mydir/d1.json", result)
    self.assertIn("mydir/d2.json", result)

  def test_format_multiple_paths_list(self):
    p1 = os.path.join(self.sandbox_dir, "a.json")
    p2 = os.path.join(self.sandbox_dir, "b.json")
    with open(p1, "w") as f:
      f.write('{"a":1}')
    with open(p2, "w") as f:
      f.write('{"b":2}')

    result = tool_format_file(self.sandbox_dir, path=["a.json", "b.json"])
    self.assertIn("Formatted 2 of 2 file(s)", result)
    self.assertIn("a.json", result)
    self.assertIn("b.json", result)

  def test_format_paths_argument(self):
    p1 = os.path.join(self.sandbox_dir, "p1.json")
    p2 = os.path.join(self.sandbox_dir, "p2.json")
    with open(p1, "w") as f:
      f.write('{"p1":1}')
    with open(p2, "w") as f:
      f.write('{"p2":2}')

    result = tool_format_file(self.sandbox_dir, paths=["p1.json", "p2.json"])
    self.assertIn("Formatted 2 of 2 file(s)", result)
    self.assertIn("p1.json", result)
    self.assertIn("p2.json", result)

  def test_format_glob_skips_unsupported_for_explicit_formatter(self):
    style_path = os.path.join(self.sandbox_dir, ".style.yapf")
    with open(style_path, "w") as f:
      f.write("[style]\nbased_on_style = pep8\nindent_width = 2\n")

    py_path = os.path.join(self.sandbox_dir, "script.py")
    json_path = os.path.join(self.sandbox_dir, "data.json")
    with open(py_path, "w") as f:
      f.write("def fn():\n    return 42\n")
    with open(json_path, "w") as f:
      f.write('{"data": 1}')

    if shutil.which("yapf"):
      result = tool_format_file(self.sandbox_dir, "*", formatter="yapf")
      self.assertIn("script.py", result)
      self.assertIn("1 skipped", result)

  def test_format_glob_respects_gitignore(self):
    gitignore_path = os.path.join(self.sandbox_dir, ".gitignore")
    with open(gitignore_path, "w") as f:
      f.write("ignored/\n*.ignored.json\n")

    ignored_dir = os.path.join(self.sandbox_dir, "ignored")
    os.makedirs(ignored_dir, exist_ok=True)
    with open(os.path.join(ignored_dir, "bad.json"), "w") as f:
      f.write('{"ignored": 1}')
    with open(os.path.join(self.sandbox_dir, "skip.ignored.json"), "w") as f:
      f.write('{"ignored": 2}')

    good_path = os.path.join(self.sandbox_dir, "good.json")
    with open(good_path, "w") as f:
      f.write('{"good": 1}')

    result = tool_format_file(self.sandbox_dir, "**/*.json")
    self.assertIn("Formatted 1 of 1 file(s)", result)
    self.assertIn("good.json", result)
    self.assertNotIn("bad.json", result)
    self.assertNotIn("skip.ignored.json", result)

  def test_format_all_already_formatted_batch(self):
    p = os.path.join(self.sandbox_dir, "clean.json")
    with open(p, "w") as f:
      f.write('{\n  "clean": 1\n}\n')

    result = tool_format_file(self.sandbox_dir, path=["clean.json"])
    self.assertIn("already formatted correctly", result)

  def test_format_batch_partial_error(self):
    p1 = os.path.join(self.sandbox_dir, "valid.json")
    p2 = os.path.join(self.sandbox_dir, "broken.json")
    with open(p1, "w") as f:
      f.write('{"valid": 1}')
    with open(p2, "w") as f:
      f.write('{"broken": ')

    result = tool_format_file(self.sandbox_dir, paths=["valid.json", "broken.json"])
    self.assertIn("Formatted 1 of 2 file(s)", result)
    self.assertIn("1 error(s)", result)
    self.assertIn("valid.json", result)
    self.assertIn("broken.json", result)


if __name__ == "__main__":
  unittest.main()
