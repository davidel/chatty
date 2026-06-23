import os
import shutil
import tempfile
import unittest
import sys
import subprocess

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.cli import ChatbotSession, execute_tool

class TestToolStats(unittest.TestCase):
    def setUp(self):
        self.sandbox_dir = tempfile.mkdtemp()
        self.session = ChatbotSession(
            provider="ollama",
            model="mock-model",
            context_size=10000,
            sandbox=self.sandbox_dir
        )

    def tearDown(self):
        shutil.rmtree(self.sandbox_dir)

    def test_initial_stats(self):
        # Stats should be empty initially
        self.assertEqual(self.session.tool_calls_count, {})
        self.assertEqual(self.session.external_binaries_count, 0)
        self.assertEqual(self.session.external_binaries_breakdown, {})

    def test_tool_stats_tracking(self):
        # Call a tool via execute_tool
        execute_tool("make_directory", {"path": "test_dir"}, self.session)
        
        # Tool stats should reflect the call
        self.assertEqual(self.session.tool_calls_count.get("make_directory"), 1)
        self.assertEqual(self.session.tool_calls_count.get("read_file", 0), 0)

        # Call another tool
        execute_tool("make_directory", {"path": "test_dir_2"}, self.session)
        self.assertEqual(self.session.tool_calls_count.get("make_directory"), 2)

    def test_external_binary_tracking(self):
        # Run a subprocess
        subprocess.run(["echo", "hello test"])
        
        # Binary execution counts should reflect it
        self.assertEqual(self.session.external_binaries_count, 1)
        self.assertEqual(self.session.external_binaries_breakdown.get("echo"), 1)

        # Run another subprocess (e.g. ls / dir)
        # Check target OS or just execute a Python command
        subprocess.run([sys.executable, "-c", "print('test')"])
        self.assertEqual(self.session.external_binaries_count, 2)
        python_bin = os.path.basename(sys.executable)
        self.assertEqual(self.session.external_binaries_breakdown.get(python_bin), 1)

    def test_slash_command_execution(self):
        # Execute the slash command /tool_stats via handle_command
        # Should execute successfully without raising exceptions
        result = self.session.handle_command("/tool_stats")
        self.assertTrue(result)

    def test_pipeline_and_builtin_filtering(self):
        # Run a shell pipeline with redirection and builtins
        # cd dir && make || echo 'failed'
        # Both cd and echo should be tracked
        from chatty.cli import record_command_binaries
        record_command_binaries("(cd dir && make) || echo 'failed'")
        
        self.assertEqual(self.session.external_binaries_count, 3)
        self.assertEqual(self.session.external_binaries_breakdown.get("make"), 1)
        self.assertEqual(self.session.external_binaries_breakdown.get("cd"), 1)
        self.assertEqual(self.session.external_binaries_breakdown.get("echo"), 1)
