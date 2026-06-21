import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.cli import tool_format_file

class TestFormatFile(unittest.TestCase):
    def setUp(self):
        self.sandbox_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.sandbox_dir)

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

if __name__ == "__main__":
    unittest.main()
