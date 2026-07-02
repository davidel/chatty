import os
import shutil
import tempfile
import unittest
import sys
from unittest.mock import patch, MagicMock

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.session import ChatbotSession
from chatty.safety import get_safe_path, active_session_var
from chatty.commands import cmd_whitelist


class TestWhitelistAndPermissions(unittest.TestCase):

  def setUp(self):
    self.sandbox_dir = tempfile.mkdtemp()
    self.outside_dir = tempfile.mkdtemp()
    self.session = ChatbotSession(
      provider="ollama",
      model="mock-model",
      context_size=8192,
      sandbox=self.sandbox_dir,
      headless=False
    )
    # Reset whitelists
    self.session.allowed_ro_paths.clear()
    self.session.allowed_rw_paths.clear()
    self.session.temp_allowed_ro_paths.clear()
    self.session.temp_allowed_rw_paths.clear()

  def tearDown(self):
    shutil.rmtree(self.sandbox_dir)
    shutil.rmtree(self.outside_dir)

  def test_inside_sandbox_always_allowed(self):
    # Any path inside the sandbox is allowed by default
    inside_path = os.path.join(self.sandbox_dir, "test.txt")
    resolved = get_safe_path(self.sandbox_dir, inside_path, write=False)
    self.assertEqual(os.path.realpath(inside_path), os.path.realpath(resolved))

  def test_outside_sandbox_blocked_without_session(self):
    # Outside paths are blocked if active_session_var is not set
    outside_path = os.path.join(self.outside_dir, "test.txt")
    with self.assertRaises(PermissionError):
      get_safe_path(self.sandbox_dir, outside_path, write=False)

  def test_whitelist_matching(self):
    # Paths (and sub-paths) matching the whitelist should be allowed
    parent_path = os.path.realpath(self.outside_dir)
    child_path = os.path.join(parent_path, "subfolder", "test.txt")

    # Add to RO whitelist
    self.session.allowed_ro_paths.add(parent_path)
    self.assertTrue(self.session.has_path_permission(child_path, write=False))
    self.assertFalse(self.session.has_path_permission(child_path, write=True))

    # Add to RW whitelist
    self.session.allowed_rw_paths.add(parent_path)
    self.assertTrue(self.session.has_path_permission(child_path, write=True))

  @patch('builtins.input', return_value='n')
  def test_get_safe_path_with_bound_session(self, mock_input):
    outside_path = os.path.realpath(os.path.join(self.outside_dir, "test.txt"))
    self.session.allowed_ro_paths.add(outside_path)

    token = active_session_var.set(self.session)
    try:
      # RO access should succeed without prompting because it's whitelisted
      resolved = get_safe_path(self.sandbox_dir, outside_path, write=False)
      self.assertEqual(outside_path, resolved)

      # RW access should raise PermissionError since it's only in RO whitelist and mock_input returns 'n' (deny)
      with self.assertRaises(PermissionError):
        get_safe_path(self.sandbox_dir, outside_path, write=True)
    finally:
      active_session_var.reset(token)

  @patch('builtins.input', return_value='y')
  def test_interactive_prompt_allow_once(self, mock_input):
    outside_path = os.path.realpath(os.path.join(self.outside_dir, "test.txt"))
    
    token = active_session_var.set(self.session)
    try:
      resolved = get_safe_path(self.sandbox_dir, outside_path, write=False)
      self.assertEqual(outside_path, resolved)
      # Allowed once should populate temp_allowed
      self.assertIn(outside_path, self.session.temp_allowed_ro_paths)
      self.assertNotIn(outside_path, self.session.allowed_ro_paths)
    finally:
      active_session_var.reset(token)

  @patch('builtins.input', return_value='a')
  def test_interactive_prompt_allow_always(self, mock_input):
    outside_path = os.path.realpath(os.path.join(self.outside_dir, "test.txt"))
    
    token = active_session_var.set(self.session)
    try:
      resolved = get_safe_path(self.sandbox_dir, outside_path, write=True)
      self.assertEqual(outside_path, resolved)
      # Allowed always should populate allowed_rw_paths
      self.assertIn(outside_path, self.session.allowed_rw_paths)
    finally:
      active_session_var.reset(token)

  @patch('builtins.input', side_effect=['p', '1'])
  def test_interactive_prompt_parents(self, mock_input):
    parent_path = os.path.realpath(self.outside_dir)
    outside_path = os.path.realpath(os.path.join(parent_path, "sub", "test.txt"))
    
    token = active_session_var.set(self.session)
    try:
      resolved = get_safe_path(self.sandbox_dir, outside_path, write=False)
      self.assertEqual(outside_path, resolved)
      # The immediate parent directory (choice 1) should be whitelisted
      self.assertIn(os.path.dirname(outside_path), self.session.allowed_ro_paths)
    finally:
      active_session_var.reset(token)

  def test_slash_commands(self):
    path1 = "/tmp/outside1"
    path2 = "/tmp/outside2"
    
    # Test add command
    cmd_whitelist(self.session, f"add {path1} ro")
    self.assertIn(os.path.realpath(path1), self.session.allowed_ro_paths)
    
    cmd_whitelist(self.session, f"add {path2} rw")
    self.assertIn(os.path.realpath(path2), self.session.allowed_rw_paths)

    # Test remove command
    cmd_whitelist(self.session, f"remove {path1}")
    self.assertNotIn(os.path.realpath(path1), self.session.allowed_ro_paths)

    # Test clear command
    cmd_whitelist(self.session, "clear")
    self.assertEqual(len(self.session.allowed_rw_paths), 0)

  def test_initial_whitelist_processing(self):
    # Test initialization of whitelist paths with modes from constructor
    path1 = "/tmp/initial_ro"
    path2 = "/tmp/initial_rw"
    path3 = "/tmp/initial_default"
    
    session = ChatbotSession(
      provider="ollama",
      model="mock-model",
      sandbox=self.sandbox_dir,
      headless=True,
      whitelist=[f"{path1}:ro", f"{path2}:rw", path3]
    )
    
    self.assertIn(os.path.realpath(path1), session.allowed_ro_paths)
    self.assertIn(os.path.realpath(path2), session.allowed_rw_paths)
    self.assertIn(os.path.realpath(path3), session.allowed_ro_paths)


if __name__ == '__main__':
  unittest.main()
