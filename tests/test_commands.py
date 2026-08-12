import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.commands import COMMANDS, cmd_provider, cmd_model, cmd_models, cmd_undo, cmd_pop, cmd_compress, cmd_skill
from chatty.utils import repair_json


class TestCommandsRegistry(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = tempfile.mkdtemp()
    self.session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir
    )

  def tearDown(self):
    os.chdir(self.old_cwd)
    shutil.rmtree(self.sandbox_dir)

  def test_registry_contains_commands(self):
    self.assertIn("/exit", COMMANDS)
    self.assertIn("/status", COMMANDS)
    self.assertIn("/provider", COMMANDS)
    self.assertIn("/model", COMMANDS)
    self.assertIn("/models", COMMANDS)
    self.assertIn("/undo", COMMANDS)
    self.assertIn("/pop", COMMANDS)

  def test_cmd_skill(self):
    self.session.ondemand_skills = {
      "greetings": {
        "metadata": {"name": "greetings", "description": "test"},
        "body": "greetings body"
      },
      "database": {
        "metadata": {"name": "database", "description": "test db"},
        "body": "db body"
      }
    }
    self.session.explicit_skills = []

    res = cmd_skill(self.session, "")
    self.assertTrue(res)

    res = cmd_skill(self.session, "greetings")
    self.assertTrue(res)
    self.assertIn("greetings", self.session.explicit_skills)

    res = cmd_skill(self.session, "database non_existent")
    self.assertTrue(res)
    self.assertIn("database", self.session.explicit_skills)
    self.assertNotIn("non_existent", self.session.explicit_skills)

    res = cmd_skill(self.session, "clear")
    self.assertTrue(res)
    self.assertEqual(len(self.session.explicit_skills), 0)

  def test_cmd_provider_and_model(self):
    self.assertEqual(self.session.provider, "ollama")
    res = cmd_provider(self.session, "openrouter")
    self.assertTrue(res)
    self.assertEqual(self.session.provider, "openrouter")

    self.assertEqual(self.session.model, "test-model")
    res = cmd_model(self.session, "custom-model")
    self.assertTrue(res)
    self.assertEqual(self.session.model, "custom-model")

  def test_cmd_models_management(self):
    self.assertIn("/models", COMMANDS)
    
    # Verify current model is listed (returns True)
    self.assertTrue(cmd_models(self.session, ""))
    
    # Add a model
    res = cmd_models(self.session, "add extra-model")
    self.assertTrue(res)
    self.assertIn("extra-model", self.session.models)
    
    # Add duplicate model (should be handled gracefully)
    self.assertTrue(cmd_models(self.session, "add extra-model"))
    self.assertEqual(self.session.models.count("extra-model"), 1)
    
    # Remove a model by name
    res = cmd_models(self.session, "remove extra-model")
    self.assertTrue(res)
    self.assertNotIn("extra-model", self.session.models)
    
    # Remove a model by ID (1-based index)
    cmd_models(self.session, "add model-a") # now self.session.models is ["test-model", "model-a"]
    self.assertEqual(len(self.session.models), 2)
    res = cmd_models(self.session, "remove 2")
    self.assertTrue(res)
    self.assertEqual(len(self.session.models), 1)
    self.assertNotIn("model-a", self.session.models)

  def test_cmd_model_switching_by_id(self):
    # Setup multiple models
    self.session.models = ["model-1", "model-2", "model-3"]
    self.session.model = "model-1"
    
    # Switch using ID (integer string)
    res = cmd_model(self.session, "2")
    self.assertTrue(res)
    self.assertEqual(self.session.model, "model-2")
    
    # Switch using model name that exists
    res = cmd_model(self.session, "model-3")
    self.assertTrue(res)
    self.assertEqual(self.session.model, "model-3")
    
    # Switch using model name that does not exist (should add and switch)
    res = cmd_model(self.session, "new-model")
    self.assertTrue(res)
    self.assertEqual(self.session.model, "new-model")
    self.assertIn("new-model", self.session.models)
    
    # Invalid ID
    res = cmd_model(self.session, "99")
    self.assertTrue(res) # Command handled, printed error
    self.assertEqual(self.session.model, "new-model") # unchanged

  def test_cmd_undo(self):
    self.session.messages = [
      {"role": "user", "content": "hello"},
      {"role": "assistant", "content": "hi", "tool_calls": []},
      {"role": "tool", "content": "result"},
      {"role": "user", "content": "prompt2"},
    ]
    # Popping prompt2 (count=1)
    res = cmd_undo(self.session, "")
    self.assertTrue(res)
    self.assertEqual(len(self.session.messages), 3)
    self.assertEqual(self.session.messages[0]["content"], "hello")

    # Popping the previous turn (hello, hi, result)
    res2 = cmd_undo(self.session, "")
    self.assertTrue(res2)
    self.assertEqual(len(self.session.messages), 0)

  def test_cmd_pop(self):
    self.session.messages = [
      {"role": "user", "content": "m1"},
      {"role": "assistant", "content": "m2"},
      {"role": "user", "content": "m3"},
    ]
    res = cmd_pop(self.session, "2")
    self.assertTrue(res)
    self.assertEqual(len(self.session.messages), 1)
    self.assertEqual(self.session.messages[0]["content"], "m1")

  def test_repair_json(self):
    # Truncated JSON
    self.assertEqual(repair_json('{"key": "val'), '{"key": "val"}')
    # Single quotes
    self.assertEqual(repair_json("{'key': 'val'}"), '{"key": "val"}')
    # Trailing comma
    self.assertEqual(repair_json('{"key": "val",}'), '{"key": "val"}')


  def test_chatty_completer(self):
    from prompt_toolkit.document import Document
    from chatty.session import ChattyCompleter

    completer = ChattyCompleter(["/exit", "/help", "/load"])
    
    # 1. Test completing a prefix of a slash command
    completions = list(completer.get_completions(Document("/he"), None))
    self.assertEqual(len(completions), 1)
    self.assertEqual(completions[0].text, "/help")
    self.assertEqual(completions[0].start_position, -3)

    # 2. Test completing all slash commands when typing just slash
    completions_all = list(completer.get_completions(Document("/"), None))
    self.assertEqual(len(completions_all), 3)
    self.assertEqual([c.text for c in completions_all], ["/exit", "/help", "/load"])

    # 3. Test that path completion is invoked for '/load '
    # Create a dummy file in the sandbox directory (which is the current working directory)
    dummy_file = os.path.join(self.sandbox_dir, "test_file_completion.txt")
    with open(dummy_file, "w") as f:
      f.write("")

    completions_path = list(completer.get_completions(Document("/load test_file_comp"), None))
    self.assertTrue(any(c.display_text == "test_file_completion.txt" for c in completions_path))

    # 4. Test inline path autocompletion in general sentences
    completions_general = list(completer.get_completions(Document("look at test_file_comp"), None))
    self.assertTrue(any(c.display_text == "test_file_completion.txt" for c in completions_general))

    # 5. Test that inline path autocompletion is not triggered for non-path-like words
    completions_normal = list(completer.get_completions(Document("look at non_existent_file"), None))
    self.assertEqual(len(completions_normal), 0)

    # 6. Test model name auto-completion under /models add and /model
    self.session.available_models = [
      {"id": "openai/gpt-4o", "name": "GPT-4o"},
      {"id": "google/gemini-flash-1.5", "name": "Gemini"},
    ]
    completer_with_session = ChattyCompleter(["/exit", "/models", "/model"], session=self.session)
    
    completions_model = list(completer_with_session.get_completions(Document("/models add openai/"), None))
    self.assertEqual(len(completions_model), 1)
    self.assertEqual(completions_model[0].text, "openai/gpt-4o")
    
    completions_model_short = list(completer_with_session.get_completions(Document("/model google/"), None))
    self.assertEqual(len(completions_model_short), 1)
    self.assertEqual(completions_model_short[0].text, "google/gemini-flash-1.5")


class TestCompressCommand(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = tempfile.mkdtemp()
    self.session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      history_keep_messages=3
    )

  def tearDown(self):
    os.chdir(self.old_cwd)
    shutil.rmtree(self.sandbox_dir)

  def test_compress_basic_and_rolling_window(self):
    import unittest.mock as mock

    # Mock the LLM stream call
    class MockDelta:
      def __init__(self, content=None):
        self.content = content

    class MockChoice:
      def __init__(self, delta):
        self.delta = delta

    class MockChunk:
      def __init__(self, choices):
        self.choices = choices

    mock_chunks = [
      MockChunk([MockChoice(MockDelta(content="Structured Summary Content"))])
    ]
    self.session.client = mock.Mock()
    self.session.client.chat.completions.create.return_value = mock_chunks

    # Populate session messages
    self.session.messages = [
      {"role": "user", "content": "msg 1"},
      {"role": "assistant", "content": "msg 2"},
      {"role": "user", "content": "msg 3"},
      {"role": "assistant", "content": "msg 4"},
      {"role": "user", "content": "msg 5"},
    ]

    # Call cmd_compress with no arguments (should default to history_keep_messages = 3)
    res = cmd_compress(self.session, "")
    self.assertTrue(res)

    # The new messages should be: [User_Summary_Prompt, Assistant_Summary, msg 3, msg 4, msg 5]
    self.assertEqual(len(self.session.messages), 5)
    self.assertEqual(self.session.messages[0]["role"], "user")
    self.assertEqual(self.session.messages[1]["role"], "assistant")
    self.assertEqual(self.session.messages[1]["content"], "Structured Summary Content")
    self.assertEqual(self.session.messages[2], {"role": "user", "content": "msg 3"})
    self.assertEqual(self.session.messages[3], {"role": "assistant", "content": "msg 4"})
    self.assertEqual(self.session.messages[4], {"role": "user", "content": "msg 5"})

    # Ensure the structured template was sent to LLM
    call_args = self.session.client.chat.completions.create.call_args[1]
    sent_messages = call_args["messages"]
    # Last message sent is the summary instruction
    self.assertEqual(sent_messages[-1]["role"], "user")
    self.assertIn("Goal & Task Context", sent_messages[-1]["content"])
    self.assertIn("Codebase Modifications", sent_messages[-1]["content"])

  def test_compress_with_explicit_n(self):
    import unittest.mock as mock
    # Mock LLM
    class MockDelta:
      def __init__(self, content=None):
        self.content = content
    class MockChoice:
      def __init__(self, delta):
        self.delta = delta
    class MockChunk:
      def __init__(self, choices):
        self.choices = choices
    self.session.client = mock.Mock()
    self.session.client.chat.completions.create.return_value = [
      MockChunk([MockChoice(MockDelta(content="Summary"))])
    ]

    self.session.messages = [
      {"role": "user", "content": "msg 1"},
      {"role": "assistant", "content": "msg 2"},
      {"role": "user", "content": "msg 3"},
    ]

    # Keep exactly 1 message
    cmd_compress(self.session, "1")
    # Result: [User_Summary, Assistant_Summary, msg 3]
    self.assertEqual(len(self.session.messages), 3)
    self.assertEqual(self.session.messages[-1], {"role": "user", "content": "msg 3"})

  def test_compress_cumulative_detection(self):
    import unittest.mock as mock
    # Mock LLM
    class MockDelta:
      def __init__(self, content=None):
        self.content = content
    class MockChoice:
      def __init__(self, delta):
        self.delta = delta
    class MockChunk:
      def __init__(self, choices):
        self.choices = choices
    self.session.client = mock.Mock()
    self.session.client.chat.completions.create.return_value = [
      MockChunk([MockChoice(MockDelta(content="Summary 2"))])
    ]

    self.session.messages = [
      {"role": "user", "content": "Summarize our progress and task context so far to optimize the context window."},
      {"role": "assistant", "content": "Previous Summary Details"},
      {"role": "user", "content": "new msg"},
    ]

    cmd_compress(self.session, "0")
    
    # Verify cumulative summarization instructions were sent to LLM
    call_args = self.session.client.chat.completions.create.call_args[1]
    sent_messages = call_args["messages"]
    self.assertIn("IMPORTANT: A previous summary is present in the history", sent_messages[-1]["content"])

  def test_compress_invalid_args(self):
    # Invalid integer
    res = cmd_compress(self.session, "abc")
    self.assertTrue(res)
    # Negative integer
    res = cmd_compress(self.session, "-1")
    self.assertTrue(res)

  def test_cmd_models_available_and_search(self):
    # Setup test models in session
    self.session.provider = "openrouter"
    self.session.available_models = [
      {"id": "openai/gpt-4o", "name": "GPT-4o", "context": 128000, "pricing_input": 5.0, "pricing_output": 15.0},
      {"id": "google/gemini-flash-1.5", "name": "Gemini Flash 1.5", "context": 1000000, "pricing_input": 0.075, "pricing_output": 0.3},
      {"id": "meta-llama/llama-3-8b", "name": "Llama 3 8b", "context": 8192, "pricing_input": 0.15, "pricing_output": 0.15},
    ]

    # Test openrouter available command
    self.assertTrue(cmd_models(self.session, "available"))

    # Test openrouter search command
    self.assertTrue(cmd_models(self.session, "search gemini"))
    self.assertTrue(cmd_models(self.session, "search non_existent_model"))

    # Test ollama available command
    self.session.provider = "ollama"
    self.session.available_models = [
      {"id": "llama3:latest", "name": "llama3", "size": int(4.7 * (1024**3)), "details": {"quantization_level": "Q4_K_M"}},
      {"id": "mistral:latest", "name": "mistral", "size": int(4.1 * (1024**3)), "details": {"quantization_level": "Q4_0"}},
    ]
    self.assertTrue(cmd_models(self.session, "available"))
    self.assertTrue(cmd_models(self.session, "search mistral"))

  def test_fetch_available_models(self):
    from chatty.llm import fetch_available_models
    import unittest.mock as mock

    # Mock urllib.request.urlopen
    with mock.patch("urllib.request.urlopen") as mock_urlopen:
      # Mock openrouter API response
      self.session.provider = "openrouter"
      self.session.api_key = "dummy_key"
      mock_response = mock.Mock()
      mock_response.read.return_value = b'{"data": [{"id": "model-1", "name": "Model 1", "context_length": 4096, "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]}'
      mock_urlopen.return_value.__enter__.return_value = mock_response

      models = fetch_available_models(self.session, force_refresh=True)
      self.assertEqual(len(models), 1)
      self.assertEqual(models[0]["id"], "model-1")
      self.assertEqual(models[0]["pricing_input"], 1.0)
      
      # Mock ollama API response
      self.session.provider = "ollama"
      self.session.url = "http://localhost:11434"
      mock_response_ollama = mock.Mock()
      mock_response_ollama.read.return_value = b'{"models": [{"name": "llama3:latest", "size": 4000000000, "details": {"quantization_level": "Q4"}}]}'
      mock_urlopen.return_value.__enter__.return_value = mock_response_ollama
      
      models_ollama = fetch_available_models(self.session)
      self.assertEqual(len(models_ollama), 1)
      self.assertEqual(models_ollama[0]["id"], "llama3:latest")

  def test_cmd_copy_and_write(self):
    from unittest.mock import patch

    # 1. No assistant messages
    with patch("chatty.commands.console.print") as mock_print:
      res = COMMANDS["/copy"](self.session, "")
      self.assertTrue(res)
      mock_print.assert_any_call("[bold red]No assistant response found in this session yet.[/bold red]")

    # 2. Add assistant message without code blocks
    self.session.messages.append({"role": "assistant", "content": "Hello! I am an assistant."})
    with patch("chatty.commands.console.print") as mock_print:
      res = COMMANDS["/copy"](self.session, "")
      self.assertTrue(res)
      mock_print.assert_any_call("[bold red]No code blocks found in the last assistant response.[/bold red]")

    # 3. Add assistant message with 1 code block
    self.session.messages.append({
      "role": "assistant",
      "content": "Here is the code:\n```python\nprint('hello')\n```\nEnjoy!"
    })

    # Test copy with 1 code block
    with patch("chatty.commands.console.print") as mock_print, \
         patch("chatty.utils.copy_to_clipboard") as mock_copy:
      mock_copy.return_value = True
      res = COMMANDS["/copy"](self.session, "")
      self.assertTrue(res)
      mock_copy.assert_called_once_with("print('hello')")
      mock_print.assert_any_call("[bold green]Copied code block 1 to clipboard.[/bold green]")

    # Test write with 1 code block
    with patch("chatty.commands.console.print") as mock_print:
      # Write block to a file
      res = COMMANDS["/write"](self.session, "my_file.py")
      self.assertTrue(res)
      # Check file contents
      file_path = os.path.join(self.sandbox_dir, "my_file.py")
      self.assertTrue(os.path.exists(file_path))
      with open(file_path, "r", encoding="utf-8") as f:
        self.assertEqual(f.read(), "print('hello')")

    # 4. Add assistant message with multiple code blocks
    self.session.messages.append({
      "role": "assistant",
      "content": "Here is Python:\n```python\nprint('one')\n```\nAnd Bash:\n```bash\necho 'two'\n```"
    })

    # Test copy multiple blocks without specifying index (shows choice menu)
    with patch("chatty.commands.console.print") as mock_print, \
         patch("chatty.utils.copy_to_clipboard") as mock_copy:
      res = COMMANDS["/copy"](self.session, "")
      self.assertTrue(res)
      mock_copy.assert_not_called()
      mock_print.assert_any_call("[bold yellow]Multiple code blocks found in the last response:[/bold yellow]")

    # Test copy multiple blocks specifying index 2
    with patch("chatty.commands.console.print") as mock_print, \
         patch("chatty.utils.copy_to_clipboard") as mock_copy:
      mock_copy.return_value = True
      res = COMMANDS["/copy"](self.session, "2")
      self.assertTrue(res)
      mock_copy.assert_called_once_with("echo 'two'")
      mock_print.assert_any_call("[bold green]Copied code block 2 to clipboard.[/bold green]")

    # Test write multiple blocks specifying index 1
    with patch("chatty.commands.console.print") as mock_print:
      res = COMMANDS["/write"](self.session, "first.py 1")
      self.assertTrue(res)
      first_file = os.path.join(self.sandbox_dir, "first.py")
      self.assertTrue(os.path.exists(first_file))
      with open(first_file, "r", encoding="utf-8") as f:
        self.assertEqual(f.read(), "print('one')")


if __name__ == "__main__":
  unittest.main()

