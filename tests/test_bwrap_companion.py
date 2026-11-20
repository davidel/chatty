import json
import os
import socket
import struct
import subprocess
import threading
import unittest
from unittest.mock import MagicMock, patch

from chatty.bwrap_companion import (
  handle_client,
  recv_exact,
  recv_framed_msg,
  send_framed_msg,
)


class TestBwrapCompanion(unittest.TestCase):

  def test_framed_msg_roundtrip_small(self):
    sock_a, sock_b = socket.socketpair()
    try:
      payload = b"hello world"
      send_framed_msg(sock_a, payload)
      received = recv_framed_msg(sock_b, max_bytes=1024 * 1024)
      self.assertEqual(received, payload)
    finally:
      sock_a.close()
      sock_b.close()

  def test_framed_msg_roundtrip_large_over_64kb(self):
    sock_a, sock_b = socket.socketpair()
    try:
      # 256 KB payload in separate thread to prevent single-thread socket buffer deadlock
      large_payload = b"A" * (256 * 1024)
      t = threading.Thread(target=send_framed_msg, args=(sock_a, large_payload))
      t.start()
      received = recv_framed_msg(sock_b, max_bytes=1024 * 1024)
      t.join()
      self.assertEqual(received, large_payload)
    finally:
      sock_a.close()
      sock_b.close()

  def test_framed_msg_exceeds_max_bytes_raises(self):
    sock_a, sock_b = socket.socketpair()
    try:
      payload = b"1234567890"
      send_framed_msg(sock_a, payload)
      with self.assertRaises(ValueError) as ctx:
        recv_framed_msg(sock_b, max_bytes=5)
      self.assertIn("exceeds configured limit", str(ctx.exception))
    finally:
      sock_a.close()
      sock_b.close()

  @patch("subprocess.run")
  def test_handle_client_success(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
      args=["bwrap"],
      returncode=0,
      stdout=b"output data",
      stderr=b""
    )
    sock_a, sock_b = socket.socketpair()
    try:
      req_payload = json.dumps({"cmd": ["bwrap", "echo", "hi"]}).encode("utf-8")
      send_framed_msg(sock_a, req_payload)

      handle_client(sock_b, bwrap_bin="/usr/bin/bwrap", default_timeout=10.0, max_payload_bytes=1024 * 1024)

      raw_resp = recv_framed_msg(sock_a, max_bytes=1024 * 1024)
      resp = json.loads(raw_resp.decode("utf-8"))
      self.assertEqual(resp["returncode"], 0)
      self.assertEqual(resp["stdout"], "output data")
    finally:
      sock_a.close()

  @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["bwrap"], timeout=5.0))
  def test_handle_client_timeout(self, mock_run):
    sock_a, sock_b = socket.socketpair()
    try:
      req_payload = json.dumps({"cmd": ["bwrap", "sleep", "10"], "timeout": 5.0}).encode("utf-8")
      send_framed_msg(sock_a, req_payload)

      handle_client(sock_b, bwrap_bin="/usr/bin/bwrap", default_timeout=5.0, max_payload_bytes=1024 * 1024)

      raw_resp = recv_framed_msg(sock_a, max_bytes=1024 * 1024)
      resp = json.loads(raw_resp.decode("utf-8"))
      self.assertEqual(resp["returncode"], 124)
      self.assertIn("timed out", resp["stderr"])
    finally:
      sock_a.close()


if __name__ == "__main__":
  unittest.main()
