import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from chatty.session import ChatbotSession
from chatty.commands import cmd_model, cmd_oracle, cmd_discovery_model
from chatty.models_config import (
  get_project_chatty_dir,
  get_saved_models,
  save_model_for_task,
  save_models_for_tasks
)


class TestModelsConfig(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.sandbox_dir = self.temp_dir.name

  def tearDown(self):
    try:
      os.chdir(self.old_cwd)
    except Exception:
      pass
    self.temp_dir.cleanup()

  def test_save_and_get_saved_models(self):
    save_model_for_task("main", "model-primary", sandbox=self.sandbox_dir)
    save_model_for_task("oracle", "model-oracle", sandbox=self.sandbox_dir)
    save_model_for_task("discovery", "model-scout", sandbox=self.sandbox_dir)

    saved = get_saved_models(sandbox=self.sandbox_dir)
    self.assertEqual(saved.get("main"), "model-primary")
    self.assertEqual(saved.get("oracle"), "model-oracle")
    self.assertEqual(saved.get("discovery"), "model-scout")

    # Verify JSON structure directly
    models_file = os.path.join(self.sandbox_dir, ".chatty", "models.json")
    self.assertTrue(os.path.exists(models_file))
    with open(models_file, "r", encoding="utf-8") as f:
      data = json.load(f)
    self.assertEqual(data["main"], "model-primary")
    self.assertEqual(data["oracle"], "model-oracle")
    self.assertEqual(data["discovery"], "model-scout")

  def test_aliases_handling(self):
    chatty_dir = os.path.join(self.sandbox_dir, ".chatty")
    os.makedirs(chatty_dir, exist_ok=True)
    models_file = os.path.join(chatty_dir, "models.json")
    with open(models_file, "w", encoding="utf-8") as f:
      json.dump({
        "chat": "alias-main",
        "oracle_model": "alias-oracle",
        "scout": "alias-discovery"
      }, f)

    saved = get_saved_models(sandbox=self.sandbox_dir)
    self.assertEqual(saved.get("main"), "alias-main")
    self.assertEqual(saved.get("oracle"), "alias-oracle")
    self.assertEqual(saved.get("discovery"), "alias-discovery")

  def test_session_attribute_assignment_persists(self):
    session = ChatbotSession(
      provider="ollama",
      model="init-model",
      sandbox=self.sandbox_dir
    )

    session.model = "switched-model"
    session.oracle_model = "switched-oracle"
    session.discovery_model = "switched-discovery"

    saved = get_saved_models(sandbox=self.sandbox_dir)
    self.assertEqual(saved.get("main"), "switched-model")
    self.assertEqual(saved.get("oracle"), "switched-oracle")
    self.assertEqual(saved.get("discovery"), "switched-discovery")

  def test_slash_commands_persist_models(self):
    session = ChatbotSession(
      provider="ollama",
      model="cmd-model-init",
      models=["cmd-model-init", "alt-model"],
      sandbox=self.sandbox_dir
    )

    self.assertTrue(cmd_model(session, "alt-model"))
    self.assertTrue(cmd_oracle(session, "expert-oracle"))
    self.assertTrue(cmd_discovery_model(session, "fast-discovery"))

    saved = get_saved_models(sandbox=self.sandbox_dir)
    self.assertEqual(saved.get("main"), "alt-model")
    self.assertEqual(saved.get("oracle"), "expert-oracle")
    self.assertEqual(saved.get("discovery"), "fast-discovery")

  @patch("chatty.cli.ChatbotSession")
  def test_cli_reuses_saved_models_when_flags_omitted(self, mock_session_class):
    mock_instance = MagicMock()
    mock_session_class.return_value.__enter__.return_value = mock_instance

    save_model_for_task("main", "saved-main", sandbox=self.sandbox_dir)
    save_model_for_task("oracle", "saved-oracle", sandbox=self.sandbox_dir)
    save_model_for_task("discovery", "saved-discovery", sandbox=self.sandbox_dir)

    from chatty.cli import main
    with patch("sys.argv", ["chatty", "--sandbox", self.sandbox_dir, "--headless"]):
      main()

    call_kwargs = mock_session_class.call_args[1]
    self.assertEqual(call_kwargs.get("model"), "saved-main")
    self.assertEqual(call_kwargs.get("oracle_model"), "saved-oracle")
    self.assertEqual(call_kwargs.get("discovery_model"), "saved-discovery")

  @patch("chatty.cli.ChatbotSession")
  def test_cli_flags_override_saved_models(self, mock_session_class):
    mock_instance = MagicMock()
    mock_session_class.return_value.__enter__.return_value = mock_instance

    save_model_for_task("main", "saved-main", sandbox=self.sandbox_dir)
    save_model_for_task("oracle", "saved-oracle", sandbox=self.sandbox_dir)
    save_model_for_task("discovery", "saved-discovery", sandbox=self.sandbox_dir)

    from chatty.cli import main
    with patch("sys.argv", [
      "chatty",
      "--sandbox", self.sandbox_dir,
      "--model", "cli-override-main",
      "--oracle-model", "cli-override-oracle",
      "--discovery-model", "cli-override-discovery",
      "--headless"
    ]):
      main()

    call_kwargs = mock_session_class.call_args[1]
    self.assertEqual(call_kwargs.get("model"), "cli-override-main")
    self.assertEqual(call_kwargs.get("oracle_model"), "cli-override-oracle")
    self.assertEqual(call_kwargs.get("discovery_model"), "cli-override-discovery")

    saved = get_saved_models(sandbox=self.sandbox_dir)
    self.assertEqual(saved.get("main"), "cli-override-main")
    self.assertEqual(saved.get("oracle"), "cli-override-oracle")
    self.assertEqual(saved.get("discovery"), "cli-override-discovery")
