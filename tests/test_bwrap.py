import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from chatty.bwrap import (
  bootstrap_micromamba,
  ensure_micromamba,
  get_bwrap_sbox_dir,
  get_bwrap_sbox_status,
  get_cached_micromamba_path,
  init_bwrap_sbox_dirs,
  is_bwrap_overlay_supported,
  reset_bwrap_sbox,
  safe_rmtree,
  wrap_command_with_bwrap,
)


class TestBwrapHelpers(unittest.TestCase):

  def setUp(self):
    self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
    self.sandbox_dir = os.path.join(self.temp_dir, "sandbox")
    os.makedirs(self.sandbox_dir, exist_ok=True)

  def test_get_bwrap_sbox_dir(self):
    sbox_dir = get_bwrap_sbox_dir(self.sandbox_dir)
    self.assertEqual(sbox_dir, os.path.join(os.path.realpath(self.sandbox_dir), ".chatty", "sbox"))

  def test_init_bwrap_sbox_dirs(self):
    home, upper, work = init_bwrap_sbox_dirs(self.sandbox_dir)
    self.assertTrue(os.path.isdir(home))
    self.assertTrue(os.path.isdir(upper))
    self.assertTrue(os.path.isdir(work))
    self.assertEqual(os.path.basename(home), "home")
    self.assertEqual(os.path.basename(upper), "upper")
    self.assertEqual(os.path.basename(work), "work")

  def test_safe_rmtree_and_reset(self):
    home, upper, work = init_bwrap_sbox_dirs(self.sandbox_dir)
    test_file = os.path.join(home, "sample.txt")
    with open(test_file, "w") as f:
      f.write("hello")
    self.assertTrue(os.path.exists(test_file))

    reset_bwrap_sbox(self.sandbox_dir)
    self.assertFalse(os.path.exists(test_file))
    self.assertTrue(os.path.isdir(home))
    self.assertTrue(os.path.isdir(upper))
    self.assertTrue(os.path.isdir(work))

  def test_get_bwrap_sbox_status(self):
    home, upper, _ = init_bwrap_sbox_dirs(self.sandbox_dir)
    with open(os.path.join(home, "a.txt"), "w") as f:
      f.write("12345")
    with open(os.path.join(upper, "b.txt"), "w") as f:
      f.write("1234567")

    status = get_bwrap_sbox_status(self.sandbox_dir)
    self.assertEqual(status["home_bytes"], 5)
    self.assertEqual(status["home_files"], 1)
    self.assertEqual(status["upper_bytes"], 7)
    self.assertEqual(status["upper_files"], 1)
    self.assertEqual(status["total_bytes"], 12)
    self.assertEqual(status["total_files"], 2)

  def test_is_bwrap_overlay_supported(self):
    with patch("subprocess.run") as mock_run:
      mock_run.return_value = MagicMock(returncode=0)
      test_dir = os.path.join(self.temp_dir, "test_supported")
      supported = is_bwrap_overlay_supported(test_dir, bwrap_bin="/bin/true")
      self.assertTrue(supported)

    with patch("subprocess.run") as mock_run:
      mock_run.return_value = MagicMock(returncode=1)
      test_dir2 = os.path.join(self.temp_dir, "test_unsupported")
      supported2 = is_bwrap_overlay_supported(test_dir2, bwrap_bin="/bin/true")
      self.assertFalse(supported2)

  def test_wrap_command_without_overlay(self):
    cmd_args = wrap_command_with_bwrap(
      bwrap_bin="/usr/bin/bwrap",
      sandbox_dir=self.sandbox_dir,
      command="echo 1",
      enable_overlay=False,
      isolated_home=True,
    )
    self.assertNotIn("--overlay-src", cmd_args)
    self.assertNotIn("--overlay", cmd_args)
    self.assertIn("--bind", cmd_args)
    self.assertIn("MAMBA_ROOT_PREFIX", cmd_args)

  def test_ensure_micromamba_on_host(self):
    with patch("shutil.which", return_value="/usr/bin/micromamba"):
      result = ensure_micromamba(self.sandbox_dir)
      self.assertEqual(result, "/usr/bin/micromamba")

  def test_ensure_micromamba_download_and_link(self):
    fake_cache = os.path.join(self.temp_dir, "fake_cache_micromamba")
    with open(fake_cache, "w") as f:
      f.write("#!/bin/sh\necho 1.0\n")
    os.chmod(fake_cache, 0o755)

    with patch("shutil.which", return_value=None):
      with patch("chatty.bwrap.bootstrap_micromamba", return_value=fake_cache):
        result = ensure_micromamba(self.sandbox_dir)
        expected_target = os.path.join(
          get_bwrap_sbox_dir(self.sandbox_dir),
          "home",
          ".local",
          "bin",
          "micromamba"
        )
        self.assertEqual(result, expected_target)
        self.assertTrue(os.path.exists(expected_target))
        self.assertTrue(os.access(expected_target, os.X_OK))
