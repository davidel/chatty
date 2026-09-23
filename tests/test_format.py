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


if __name__ == "__main__":
  unittest.main()
