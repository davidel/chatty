import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.cli import (
    tool_list_dir,
    tool_read_file,
    tool_search_grep,
    tool_fetch_url,
    ChatbotSession,
    truncate_output
)

class TestCutoffs(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.sandbox_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.sandbox_dir)

    def test_truncate_output(self):
        text = "abcdefghij"
        # max_chars=4 -> half is 2, first 2 + middle notice + last 2
        truncated = truncate_output(text, max_chars=4)
        self.assertTrue(truncated.startswith("ab"))
        self.assertTrue(truncated.endswith("ij"))
        self.assertIn("TRUNCATED", truncated)

        # Should not truncate if length <= max_chars
        self.assertEqual(truncate_output(text, max_chars=20), text)

    def test_tool_list_dir_limit(self):
        # Create 3 files
        for i in range(3):
            with open(os.path.join(self.sandbox_dir, f"file_{i}.txt"), "w") as f:
                f.write("content")
        
        # list with limit 2
        res = tool_list_dir(self.sandbox_dir, ".", max_items=2)
        self.assertIn("WARNING: Directory listing truncated", res)
        self.assertIn("file_0.txt", res)
        self.assertIn("file_1.txt", res)
        self.assertNotIn("file_2.txt", res)

    def test_tool_read_file_limit(self):
        filepath = os.path.join(self.sandbox_dir, "large.txt")
        content = "a" * 100
        with open(filepath, "w") as f:
            f.write(content)

        # read with limit 40
        res = tool_read_file(self.sandbox_dir, "large.txt", max_chars=40)
        self.assertIn("WARNING: File 'large.txt' is too large", res)
        self.assertEqual(res[:40], "a" * 40)

    def test_tool_read_file_line_numbers(self):
        filepath = os.path.join(self.sandbox_dir, "test.txt")
        content = "line one\nline two\nline three\n"
        with open(filepath, "w") as f:
            f.write(content)

        res = tool_read_file(self.sandbox_dir, "test.txt", start_line=2, end_line=3, line_numbers=True)
        self.assertEqual(res, "2: line two\n3: line three\n")

    def test_tool_search_grep_limit(self):
        # Create a file with multiple matches
        filepath = os.path.join(self.sandbox_dir, "grep.txt")
        with open(filepath, "w") as f:
            f.write("apple\nbanana\napple\ncherry\napple\n")

        res = tool_search_grep(self.sandbox_dir, "apple", ".", max_results=2)
        self.assertIn("WARNING: Search results truncated to 2 matches", res)
        # Should contain 2 matches
        occurrences = res.count("grep.txt:")
        self.assertEqual(occurrences, 2)

    def test_prune_history_compression(self):
        # Initialize a mockup ChatbotSession
        session = ChatbotSession(
            provider="ollama",
            model="mock-model",
            context_size=10000,
            sandbox=self.sandbox_dir,
            max_history_tool_chars=50,
            history_keep_messages=2
        )
        
        # Add messages: system message is generated. Let's add tool output
        # Msg -1: assistant message defining call_1
        session.messages.append({
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}]
        })
        # Msg 0: tool call (historical) -> large output
        session.messages.append({
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "test_tool",
            "content": "x" * 200
        })
        # Msg 1: assistant message defining call_2
        session.messages.append({
            "role": "assistant",
            "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "another_tool", "arguments": "{}"}}]
        })
        # Msg 2: tool message (active, inside history_keep_messages window of 2)
        session.messages.append({
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "another_tool",
            "content": "y" * 200
        })

        # Prune history
        active_msgs = session.prune_history()
        
        # Find the first tool message in active_msgs (which corresponds to Msg 0)
        msg0_pruned = None
        msg2_pruned = None
        for m in active_msgs:
            if m.get("tool_call_id") == "call_1":
                msg0_pruned = m
            elif m.get("tool_call_id") == "call_2":
                msg2_pruned = m

        self.assertIsNotNone(msg0_pruned)
        self.assertIsNotNone(msg2_pruned)
        
        # Msg 0 (historical tool output) should be compressed/truncated
        self.assertIn("TRUNCATED", msg0_pruned["content"])
        self.assertLess(len(msg0_pruned["content"]), 200)

        # Msg 2 (within active keep window of last 2 messages) should remain raw/untruncated
        self.assertEqual(msg2_pruned["content"], "y" * 200)

    def test_openrouter_prompt_caching(self):
        # 1. Test that static_skills is enabled by default for openrouter, and disabled for ollama
        session_or = ChatbotSession(
            provider="openrouter",
            model="mock-model",
            context_size=10000,
            sandbox=self.sandbox_dir,
            prompt_caching=True
        )
        self.assertTrue(session_or.static_skills)

        session_ol = ChatbotSession(
            provider="ollama",
            model="mock-model",
            context_size=10000,
            sandbox=self.sandbox_dir
        )
        self.assertFalse(session_ol.static_skills)

        # 2. Test that system prompt and active messages contain cache_control when provider is openrouter
        session_or.messages.append({"role": "user", "content": "hello"})
        session_or.messages.append({"role": "assistant", "content": "hi there"})
        session_or.messages.append({"role": "user", "content": "how are you?"})

        active_msgs = session_or.prune_history()
        
        # System message (first element) should have cache_control
        self.assertEqual(active_msgs[0]["role"], "system")
        self.assertEqual(active_msgs[0].get("cache_control"), {"type": "ephemeral"})

        # Last two messages should have cache_control
        self.assertEqual(active_msgs[-1]["role"], "user")
        self.assertEqual(active_msgs[-1].get("cache_control"), {"type": "ephemeral"})

        self.assertEqual(active_msgs[-2]["role"], "assistant")
        self.assertEqual(active_msgs[-2].get("cache_control"), {"type": "ephemeral"})

        # 3. Test that get_tools returns tools schema annotated with cache_control for openrouter
        tools = session_or.get_tools()
        self.assertIsNotNone(tools)
        self.assertEqual(tools[-1].get("cache_control"), {"type": "ephemeral"})

        # Ollama should NOT have cache_control on anything
        session_ol.messages.append({"role": "user", "content": "hello"})
        active_msgs_ol = session_ol.prune_history()
        self.assertNotIn("cache_control", active_msgs_ol[0])
        self.assertNotIn("cache_control", active_msgs_ol[-1])
        
        tools_ol = session_ol.get_tools()
        if tools_ol:
            self.assertNotIn("cache_control", tools_ol[-1])

    def test_session_logging(self):
        import logging
        log_filepath = os.path.join(self.sandbox_dir, "test_chatty.log")
        
        # Manually configure a file handler for the 'chatty' logger
        chatty_logger = logging.getLogger("chatty")
        chatty_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_filepath, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        chatty_logger.addHandler(handler)
        
        try:
            # Instantiate session, which should write logs
            session = ChatbotSession(
                provider="ollama",
                model="mock-model-logging",
                context_size=10000,
                sandbox=self.sandbox_dir
            )
            
            # Close handler to flush contents to disk
            handler.close()
            
            # Verify log file exists and contains initialization info
            self.assertTrue(os.path.exists(log_filepath))
            with open(log_filepath, "r") as f:
                log_content = f.read()
            self.assertIn("ChatbotSession initialized", log_content)
            self.assertIn("mock-model-logging", log_content)
        finally:
            chatty_logger.removeHandler(handler)

    def test_loop_status_bar(self):
        session = ChatbotSession(
            provider="ollama",
            model="mock-model",
            context_size=10000,
            sandbox=self.sandbox_dir,
            max_loops=20
        )
        self.assertEqual(session.current_loop, 0)
        from rich.console import Console
        c = Console(width=200, record=True)
        c.print(session.get_rich_status_bar())
        rendered = c.export_text()
        rendered_clean = rendered.replace("\n", "").replace(" ", "")
        self.assertIn("Loops:0/20", rendered_clean)

    def test_context_size_scaling(self):
      session_or = ChatbotSession(
        provider="openrouter",
        model="mock-model",
        context_size=8192,
        sandbox=self.sandbox_dir
      )
      self.assertEqual(session_or.context_size, 100000)
      session_custom = ChatbotSession(
        provider="openrouter",
        model="mock-model",
        context_size=50000,
        sandbox=self.sandbox_dir
      )
      self.assertEqual(session_custom.context_size, 50000)
      session_switch = ChatbotSession(
        provider="ollama",
        model="mock-model",
        context_size=8192,
        sandbox=self.sandbox_dir
      )
      self.assertEqual(session_switch.context_size, 8192)
      session_switch.handle_command("/provider openrouter")
      self.assertEqual(session_switch.context_size, 100000)
      session_switch.handle_command("/provider ollama")
      self.assertEqual(session_switch.context_size, 8192)

if __name__ == "__main__":
    unittest.main()
