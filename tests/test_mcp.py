import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from chatty.mcp import MCPClient, MCPManager
from chatty.tools import execute_tool
from chatty.session import ChatbotSession
from chatty.commands import cmd_mcp

# Inline helper script for a compliant MCP test server over stdio
MOCK_SERVER_SCRIPT = """
import sys, json

for line in sys.stdin:
  line = line.strip()
  if not line:
    continue
  msg = json.loads(line)
  req_id = msg.get("id")
  method = msg.get("method")
  params = msg.get("params", {})

  if method == "initialize":
    res = {
      "jsonrpc": "2.0",
      "id": req_id,
      "result": {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "mock-mcp-server", "version": "1.0.0"}
      }
    }
    sys.stdout.write(json.dumps(res) + "\\n")
    sys.stdout.flush()
  elif method == "notifications/initialized":
    pass
  elif method == "tools/list":
    res = {
      "jsonrpc": "2.0",
      "id": req_id,
      "result": {
        "tools": [
          {
            "name": "echo",
            "description": "Echo input message",
            "inputSchema": {
              "type": "object",
              "properties": {"msg": {"type": "string"}},
              "required": ["msg"]
            }
          },
          {
            "name": "fail_tool",
            "description": "Tool that returns error",
            "inputSchema": {"type": "object", "properties": {}}
          }
        ]
      }
    }
    sys.stdout.write(json.dumps(res) + "\\n")
    sys.stdout.flush()
  elif method == "tools/call":
    tool_name = params.get("name")
    args = params.get("arguments", {})
    if tool_name == "echo":
      res = {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
          "content": [{"type": "text", "text": "Echo: " + str(args.get("msg", ""))}],
          "isError": False
        }
      }
    else:
      res = {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
          "content": [{"type": "text", "text": "Intentional tool failure"}],
          "isError": True
        }
      }
    sys.stdout.write(json.dumps(res) + "\\n")
    sys.stdout.flush()
"""


class TestMCPIntegration(unittest.TestCase):

  def setUp(self):
    self.old_cwd = os.getcwd()
    self.sandbox_dir = self.enterContext(tempfile.TemporaryDirectory())
    self.server_script_path = os.path.join(self.sandbox_dir, "mock_mcp_server.py")
    with open(self.server_script_path, "w", encoding="utf-8") as f:
      f.write(MOCK_SERVER_SCRIPT)

  def tearDown(self):
    os.chdir(self.old_cwd)

  def test_mcp_client_handshake_and_tool_call(self):
    client = MCPClient(
      name="test-server",
      command=sys.executable,
      args=[self.server_script_path],
      cwd=self.sandbox_dir,
      timeout=5.0
    )
    try:
      connected = client.connect()
      self.assertTrue(connected)
      self.assertTrue(client.is_connected)
      self.assertEqual(client.server_info.get("name"), "mock-mcp-server")
      self.assertEqual(len(client.tools), 2)
      self.assertEqual(client.tools[0]["name"], "echo")

      # Call echo tool
      result = client.call_tool("echo", {"msg": "hello from test"})
      self.assertEqual(result, "Echo: hello from test")

      # Call failing tool
      err_result = client.call_tool("fail_tool", {})
      self.assertIn("Error: Intentional tool failure", err_result)
    finally:
      client.close()
      self.assertFalse(client.is_connected)

  def test_mcp_manager_config_file_discovery_and_execution(self):
    # Create .chatty/mcp.json in sandbox
    chatty_dir = os.path.join(self.sandbox_dir, ".chatty")
    os.makedirs(chatty_dir, exist_ok=True)
    mcp_json_path = os.path.join(chatty_dir, "mcp.json")
    config_data = {
      "mcpServers": {
        "local_mock": {
          "command": sys.executable,
          "args": [self.server_script_path]
        }
      }
    }
    with open(mcp_json_path, "w", encoding="utf-8") as f:
      json.dump(config_data, f)

    manager = MCPManager(sandbox=self.sandbox_dir)
    try:
      manager.load_and_connect()
      self.assertIn("local_mock", manager.clients)
      client = manager.clients["local_mock"]
      self.assertTrue(client.is_connected)

      # Verify tool discovery and mapping
      self.assertTrue(manager.has_tool("echo"))
      self.assertTrue(manager.has_tool("mcp__local_mock__echo"))

      # Verify tool execution
      out = manager.call_tool("echo", {"msg": "world"})
      self.assertEqual(out, "Echo: world")

      # Verify schemas
      schemas = manager.get_tools_schema()
      self.assertTrue(any(s["function"]["name"] == "echo" for s in schemas))

      # Test status report
      status_list = manager.get_servers_status()
      self.assertEqual(len(status_list), 1)
      self.assertEqual(status_list[0]["name"], "local_mock")
      self.assertTrue(status_list[0]["connected"])
      self.assertIn("echo", status_list[0]["tools"])
    finally:
      manager.close()

  def test_chatbot_session_mcp_integration(self):
    mcp_config_file = os.path.join(self.sandbox_dir, "custom_mcp.json")
    config_data = {
      "mcpServers": {
        "test_srv": {
          "command": sys.executable,
          "args": [self.server_script_path]
        }
      }
    }
    with open(mcp_config_file, "w", encoding="utf-8") as f:
      json.dump(config_data, f)

    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True,
      mcp_config=mcp_config_file
    )
    with session:
      self.assertIsNotNone(session.mcp_manager)
      self.assertIn("test_srv", session.mcp_manager.clients)

      # Check get_tools includes the MCP tool
      tools = session.get_tools()
      self.assertIsNotNone(tools)
      tool_names = [t["function"]["name"] for t in tools]
      self.assertIn("echo", tool_names)

      # Test execute_tool delegates to MCP tool
      res = execute_tool("echo", {"msg": "chatty mcp test"}, session)
      self.assertEqual(res, "Echo: chatty mcp test")

      # Check tool stats tracking
      self.assertEqual(session.tool_calls_count.get("echo"), 1)

  def test_cmd_mcp(self):
    mcp_config_file = os.path.join(self.sandbox_dir, "custom_mcp.json")
    config_data = {
      "mcpServers": {
        "srv1": {
          "command": sys.executable,
          "args": [self.server_script_path]
        }
      }
    }
    with open(mcp_config_file, "w", encoding="utf-8") as f:
      json.dump(config_data, f)

    session = ChatbotSession(
      provider="ollama",
      model="test-model",
      sandbox=self.sandbox_dir,
      headless=True,
      mcp_config=mcp_config_file
    )
    with session:
      # Test /mcp table output
      self.assertTrue(cmd_mcp(session, ""))
      # Test /mcp reload
      self.assertTrue(cmd_mcp(session, "reload"))
      self.assertTrue(session.mcp_manager.clients["srv1"].is_connected)


if __name__ == "__main__":
  unittest.main()
