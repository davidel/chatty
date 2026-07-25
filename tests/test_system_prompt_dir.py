import os
import shutil
import tempfile
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession


class TestSystemPromptDir(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.temp_dir = tempfile.mkdtemp()
    os.chdir(self.temp_dir)
    self.sandbox_dir = os.path.join(self.temp_dir, "sandbox")
    os.makedirs(self.sandbox_dir, exist_ok=True)

  def tearDown(self):
    os.chdir(self.old_cwd)
    shutil.rmtree(self.temp_dir)

  def test_system_prompt_appends_from_cwd(self):
    # Create .chatty/system_prompt in the current working directory
    cwd_prompt_dir = os.path.join(self.temp_dir, ".chatty", "system_prompt")
    os.makedirs(cwd_prompt_dir, exist_ok=True)

    # Create two files that should be sorted and appended
    with open(os.path.join(cwd_prompt_dir, "02_second.txt"), "w", encoding="utf-8") as f:
      f.write("Second Prompt content.")

    with open(os.path.join(cwd_prompt_dir, "01_first.txt"), "w", encoding="utf-8") as f:
      f.write("First Prompt content.")

    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir
    )

    self.assertIn("First Prompt content.", session.system_prompt)
    self.assertIn("Second Prompt content.", session.system_prompt)
    # Ensure they are sorted: 01_first before 02_second
    first_idx = session.system_prompt.index("First Prompt content.")
    second_idx = session.system_prompt.index("Second Prompt content.")
    self.assertTrue(first_idx < second_idx)

  def test_system_prompt_appends_from_sandbox(self):
    # Create .chatty/system_prompt in the sandbox directory
    sandbox_prompt_dir = os.path.join(self.sandbox_dir, ".chatty", "system_prompt")
    os.makedirs(sandbox_prompt_dir, exist_ok=True)

    with open(os.path.join(sandbox_prompt_dir, "01_first.txt"), "w", encoding="utf-8") as f:
      f.write("Sandbox prompt content.")

    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir
    )

    self.assertIn("Sandbox prompt content.", session.system_prompt)
