import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.cli import validate_file_syntax

class TestSyntaxValidation(unittest.TestCase):
    def setUp(self):
        self.sandbox_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.sandbox_dir)

    def test_verilog_syntax_missing_module(self):
        # We write a file that instantiates a missing module
        sv_content = """
module top(input clk);
  missing_module inst(.clk(clk));
endmodule
"""
        sv_path = os.path.join(self.sandbox_dir, "top.sv")
        # Run validation
        is_valid, err_msg = validate_file_syntax(sv_path, sv_content)
        # Should succeed because MODMISSING is ignored/suppressed
        self.assertTrue(is_valid, f"Validation failed: {err_msg}")

    def test_verilog_syntax_with_other_module_in_dir(self):
        # Write another module to the same directory
        sub_content = """
module sub(input clk, input val);
endmodule
        """
        sub_path = os.path.join(self.sandbox_dir, "sub.sv")
        with open(sub_path, "w") as f:
            f.write(sub_content)

        # Write top module that instantiates sub
        top_content = """
module top(input clk);
  sub inst(.clk(clk), .val(1'b1));
endmodule
        """
        top_path = os.path.join(self.sandbox_dir, "top.sv")

        # Run validation
        is_valid, err_msg = validate_file_syntax(top_path, top_content)
        self.assertTrue(is_valid, f"Validation failed: {err_msg}")

    def test_cpp_syntax_with_compile_paths(self):
        # Create an include directory
        inc_dir = os.path.join(self.sandbox_dir, "my_includes")
        os.makedirs(inc_dir)
        
        # Write header file in it
        header_path = os.path.join(inc_dir, "helper.h")
        with open(header_path, "w") as f:
            f.write("int get_value() { return 42; }\n")
            
        # Write cpp file content that includes the header
        cpp_content = """
#include "helper.h"
int main() {
    return get_value();
}
"""
        cpp_path = os.path.join(self.sandbox_dir, "main.cpp")
        
        # Validation without compile_paths should fail (helper.h not found)
        is_valid_fail, _ = validate_file_syntax(cpp_path, cpp_content, self.sandbox_dir)
        self.assertFalse(is_valid_fail)
        
        # Validation with compile_paths should succeed
        is_valid_success, err_msg = validate_file_syntax(cpp_path, cpp_content, self.sandbox_dir, ["my_includes"])
        self.assertTrue(is_valid_success, f"Validation failed with compile_paths: {err_msg}")

    def test_verilog_syntax_with_explicit_compile_paths(self):
        # Create dependency file in a separate directory
        dep_dir = os.path.join(self.sandbox_dir, "deps")
        os.makedirs(dep_dir)
        dep_path = os.path.join(dep_dir, "my_dep.sv")
        with open(dep_path, "w") as f:
            f.write("module my_dep(input clk); endmodule\n")
            
        # Write top module that instantiates my_dep
        sv_content = """
module top(input clk);
  my_dep inst(.clk(clk));
endmodule
"""
        sv_path = os.path.join(self.sandbox_dir, "top.sv")
        
        # Validation with explicit file dependency passed in compile_paths
        is_valid, err_msg = validate_file_syntax(sv_path, sv_content, self.sandbox_dir, ["deps/my_dep.sv"])
        self.assertTrue(is_valid, f"Validation failed: {err_msg}")

if __name__ == "__main__":
    unittest.main()
