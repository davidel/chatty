import logging
import os
import re
import unittest
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.logging_setup import GlogFormatter


class TestGlogFormatter(unittest.TestCase):

  def test_formatter_layout(self):
    # Setup standard logger and a custom handler with GlogFormatter
    logger = logging.getLogger("test_glog")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Use StringIO to capture logs in-memory
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(GlogFormatter())
    logger.addHandler(handler)

    try:
      logger.info("This is an info message.")
      log_output = stream.getvalue().strip()

      # Log format: Lyyyymmdd hh:mm:ss.uuuuuu process file:line] msg
      # e.g., I20260621 08:54:00.123456 12345 test_logging.py:24] This is an info message.
      pattern = r"^I\d{8} \d{2}:\d{2}:\d{2}\.\d{6} \d+ test_logging\.py:\d+\] This is an info message\.$"
      self.assertTrue(
        re.match(pattern, log_output),
        f"Log output '{log_output}' did not match pattern '{pattern}'"
      )
    finally:
      logger.removeHandler(handler)

  def test_formatter_levels(self):
    # Verify mapping of level letters:
    # DEBUG -> D, INFO -> I, WARNING -> W, ERROR -> E, CRITICAL -> F
    logger = logging.getLogger("test_glog_levels")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(GlogFormatter())
    logger.addHandler(handler)

    try:
      logger.debug("debug message")
      logger.warning("warning message")
      logger.error("error message")
      logger.critical("critical message")

      lines = stream.getvalue().strip().split("\n")
      self.assertEqual(len(lines), 4)

      self.assertTrue(lines[0].startswith("D"))
      self.assertTrue(lines[1].startswith("W"))
      self.assertTrue(lines[2].startswith("E"))
      self.assertTrue(lines[3].startswith("F"))
    finally:
      logger.removeHandler(handler)
