import argparse
import unittest
from unittest.mock import MagicMock, patch

from chatty.commands import cmd_config
from chatty.config_specs import CONFIG_SPECS, ConfigSpec, register_config_arguments


class TestConfigSpecs(unittest.TestCase):
  def test_register_config_arguments(self):
    parser = argparse.ArgumentParser()
    register_config_arguments(parser)
    args = parser.parse_args([])
    for key, spec in CONFIG_SPECS.items():
      self.assertTrue(hasattr(args, key))
      self.assertEqual(getattr(args, key), spec.default)

  def test_parse_value_bool(self):
    spec = CONFIG_SPECS["auto_approve"]
    self.assertTrue(spec.parse_value("true"))
    self.assertTrue(spec.parse_value("yes"))
    self.assertTrue(spec.parse_value("1"))
    self.assertTrue(spec.parse_value("on"))
    self.assertFalse(spec.parse_value("false"))
    self.assertFalse(spec.parse_value("no"))
    self.assertFalse(spec.parse_value("0"))
    self.assertFalse(spec.parse_value("off"))
    with self.assertRaises(ValueError):
      spec.parse_value("maybe")

  def test_parse_value_int_and_float(self):
    int_spec = CONFIG_SPECS["max_loops"]
    self.assertEqual(int_spec.parse_value("42"), 42)
    with self.assertRaises(ValueError):
      int_spec.parse_value("not_a_number")

    float_spec = CONFIG_SPECS["api_delay"]
    self.assertEqual(float_spec.parse_value("3.5"), 3.5)
    with self.assertRaises(ValueError):
      float_spec.parse_value("not_a_float")

  def test_parse_value_choices(self):
    profile_spec = CONFIG_SPECS["profile"]
    self.assertEqual(profile_spec.parse_value("coder"), "coder")
    self.assertEqual(profile_spec.parse_value("general"), "general")
    with self.assertRaises(ValueError):
      profile_spec.parse_value("unknown_profile")

  def test_cmd_config_table_display(self):
    session = MagicMock()
    for key, spec in CONFIG_SPECS.items():
      setattr(session, key, spec.default)

    with patch("chatty.commands.console.print") as mock_print:
      res = cmd_config(session, "")
      self.assertTrue(res)
      self.assertEqual(mock_print.call_count, 1)
      table = mock_print.call_args[0][0]
      column_names = [col.header for col in table.columns]
      self.assertIn("Parameter", column_names)
      self.assertIn("Value", column_names)
      self.assertIn("Type", column_names)
      self.assertIn("Description", column_names)

  def test_cmd_config_single_param_query(self):
    session = MagicMock()
    session.max_loops = 25
    with patch("chatty.commands.console.print") as mock_print:
      res = cmd_config(session, "max_loops")
      self.assertTrue(res)
      printed = [str(call[0][0]) for call in mock_print.call_args_list if call[0]]
      self.assertTrue(any("max_loops" in p for p in printed))
      self.assertTrue(any(CONFIG_SPECS["max_loops"].description in p for p in printed))

  def test_cmd_config_update(self):
    session = MagicMock()
    session.max_loops = 20
    session.config = MagicMock()
    with patch("chatty.commands.console.print"):
      res = cmd_config(session, "max_loops=30")
      self.assertTrue(res)
      self.assertEqual(session.max_loops, 30)

  def test_cmd_config_invalid_param(self):
    session = MagicMock()
    with patch("chatty.commands.console.print") as mock_print:
      res = cmd_config(session, "invalid_setting")
      self.assertTrue(res)
      printed = [str(call[0][0]) for call in mock_print.call_args_list if call[0]]
      self.assertTrue(any("Error: 'invalid_setting' is not a configurable parameter" in p for p in printed))


if __name__ == "__main__":
  unittest.main()
