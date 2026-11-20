#!/usr/bin/env python3
"""Host-side companion daemon for executing Bubblewrap sandboxes outside container boundaries."""

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
from typing import Optional

logger = logging.getLogger("bwrap_companion")


def send_framed_msg(sock: socket.socket, data: bytes) -> None:
  """Sends length-prefixed payload (4-byte big-endian header)."""
  header = struct.pack(">I", len(data))
  sock.sendall(header + data)


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
  """Reads exactly num_bytes from the socket stream."""
  buf = bytearray()
  while len(buf) < num_bytes:
    chunk = sock.recv(min(num_bytes - len(buf), 65536))
    if not chunk:
      raise ConnectionError("Socket closed prematurely by peer.")
    buf.extend(chunk)
  return bytes(buf)


def recv_framed_msg(sock: socket.socket, max_bytes: int) -> Optional[bytes]:
  """Receives a length-prefixed payload, enforcing max_bytes limit."""
  try:
    header = recv_exact(sock, 4)
  except ConnectionError:
    return None

  (payload_len,) = struct.unpack(">I", header)
  if payload_len > max_bytes:
    raise ValueError(f"Payload size {payload_len} bytes exceeds configured limit of {max_bytes} bytes.")

  return recv_exact(sock, payload_len)


def handle_client(
  conn: socket.socket,
  bwrap_bin: str,
  default_timeout: Optional[float],
  max_payload_bytes: int
) -> None:
  """Handles a single client connection, executing the requested command."""
  with conn:
    try:
      raw_data = recv_framed_msg(conn, max_bytes=max_payload_bytes)
      if not raw_data:
        return

      payload = json.loads(raw_data.decode("utf-8"))
      cmd = payload.get("cmd", [])
      cwd = payload.get("cwd")
      env = payload.get("env")
      timeout = payload.get("timeout", default_timeout)

      if not cmd:
        response = {"returncode": 1, "stdout": "", "stderr": "Error: Empty command received."}
      else:
        if cmd[0] == "bwrap":
          cmd[0] = bwrap_bin

        logger.info(f"Executing: {' '.join(cmd[:6])}... (total args: {len(cmd)})")

        proc = subprocess.run(
          cmd,
          cwd=cwd,
          env=env,
          timeout=timeout,
          capture_output=True
        )
        response = {
          "returncode": proc.returncode,
          "stdout": proc.stdout.decode("utf-8", errors="replace"),
          "stderr": proc.stderr.decode("utf-8", errors="replace"),
        }
    except subprocess.TimeoutExpired:
      response = {"returncode": 124, "stdout": "", "stderr": f"Command timed out after {timeout} seconds."}
    except Exception as e:
      logger.exception("Error processing command request")
      response = {"returncode": 1, "stdout": "", "stderr": f"Companion daemon error: {str(e)}"}

    send_framed_msg(conn, json.dumps(response).encode("utf-8"))


def main():
  parser = argparse.ArgumentParser(
    description="Host companion daemon for running Bubblewrap from inside nested/unprivileged environments."
  )
  parser.add_argument(
    "--socket-path", "-s",
    default=os.environ.get("CHATTY_BWRAP_SOCK", "/tmp/chatty_bwrap.sock"),
    help="Path to the UNIX domain socket (default: /tmp/chatty_bwrap.sock or CHATTY_BWRAP_SOCK)."
  )
  parser.add_argument(
    "--socket-mode", "-m",
    type=lambda x: int(x, 8),
    default=0o666,
    help="Octal permissions for the socket file (default: 0666)."
  )
  parser.add_argument(
    "--bwrap-bin", "-b",
    default=shutil.which("bwrap") or "/usr/bin/bwrap",
    help="Path to the host 'bwrap' executable (default: finds bwrap in PATH)."
  )
  parser.add_argument(
    "--max-payload-mb",
    type=int,
    default=64,
    help="Maximum allowed payload length in megabytes (default: 64MB)."
  )
  parser.add_argument(
    "--timeout", "-t",
    type=float,
    default=None,
    help="Default timeout in seconds for command execution (default: None)."
  )
  parser.add_argument(
    "--verbose", "-v",
    action="store_true",
    help="Enable debug logging."
  )

  args = parser.parse_args()

  logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
  )

  if not os.path.exists(args.bwrap_bin):
    logger.error(f"Bubblewrap binary not found at '{args.bwrap_bin}'")
    sys.exit(1)

  if os.path.exists(args.socket_path):
    os.unlink(args.socket_path)

  server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  server.bind(args.socket_path)
  os.chmod(args.socket_path, args.socket_mode)
  server.listen(16)

  max_payload_bytes = args.max_payload_mb * 1024 * 1024
  logger.info(f"Bwrap companion listening on {args.socket_path} (mode: {oct(args.socket_mode)}, max: {args.max_payload_mb}MB)")

  def cleanup_exit(sig, frame):
    logger.info("Shutting down companion daemon...")
    if os.path.exists(args.socket_path):
      os.unlink(args.socket_path)
    sys.exit(0)

  signal.signal(signal.SIGINT, cleanup_exit)
  signal.signal(signal.SIGTERM, cleanup_exit)

  try:
    while True:
      conn, _ = server.accept()
      handle_client(conn, args.bwrap_bin, args.timeout, max_payload_bytes)
  finally:
    if os.path.exists(args.socket_path):
      os.unlink(args.socket_path)


if __name__ == "__main__":
  main()
