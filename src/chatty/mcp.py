import os
import sys
import json
import logging
import threading
import subprocess
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("chatty")


class MCPClient:
  """Client for communicating with an MCP (Model Context Protocol) server over stdio."""

  def __init__(
    self,
    name: str,
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: float = 30.0
  ):
    self.name = name
    self.command = command
    self.args = args or []
    self.env = env or {}
    self.cwd = cwd
    self.timeout = timeout

    self.process: Optional[subprocess.Popen] = None
    self.server_info: Dict[str, Any] = {}
    self.tools: List[Dict[str, Any]] = []
    self.is_connected = False
    self.last_error: Optional[str] = None

    self._req_counter = 0
    self._lock = threading.Lock()
    self._pending_requests: Dict[int, Tuple[threading.Event, Dict[str, Any]]] = {}
    self._reader_thread: Optional[threading.Thread] = None

  def connect(self) -> bool:
    """Spawns the MCP server subprocess and performs the initialize handshake."""
    full_cmd = [self.command] + self.args
    merged_env = os.environ.copy()
    merged_env.update(self.env)

    try:
      self.process = subprocess.Popen(
        full_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=self.cwd,
        env=merged_env,
        text=True,
        bufsize=1,
        encoding="utf-8"
      )
    except Exception as e:
      self.last_error = f"Failed to start MCP server '{self.name}': {str(e)}"
      logger.warning(self.last_error)
      return False

    self._reader_thread = threading.Thread(
      target=self._read_stdout_loop,
      daemon=True,
      name=f"mcp-reader-{self.name}"
    )
    self._reader_thread.start()

    # Perform MCP initialize handshake
    init_params = {
      "protocolVersion": "2024-11-05",
      "capabilities": {
        "tools": {}
      },
      "clientInfo": {
        "name": "chatty",
        "version": "0.5.3"
      }
    }

    try:
      init_res = self._send_request("initialize", init_params)
      if isinstance(init_res, dict):
        self.server_info = init_res.get("serverInfo", {})
    except Exception as e:
      self.last_error = f"MCP initialize failed for '{self.name}': {str(e)}"
      logger.warning(self.last_error)
      self.close()
      return False

    # Send notifications/initialized
    try:
      self._send_notification("notifications/initialized")
    except Exception as e:
      logger.debug(f"Failed to send initialized notification: {e}")

    # Discover available tools
    try:
      tools_res = self._send_request("tools/list", {})
      if isinstance(tools_res, dict):
        self.tools = tools_res.get("tools", [])
    except Exception as e:
      logger.warning(f"tools/list failed for MCP server '{self.name}': {e}")
      self.tools = []

    self.is_connected = True
    self.last_error = None
    logger.info(f"Connected to MCP server '{self.name}' ({len(self.tools)} tool(s) registered).")
    return True

  def _read_stdout_loop(self):
    """Continuously reads newline-delimited JSON-RPC messages from server stdout."""
    if not self.process or not self.process.stdout:
      return

    while True:
      try:
        line = self.process.stdout.readline()
        if not line:
          break
        line = line.strip()
        if not line:
          continue

        try:
          msg = json.loads(line)
        except Exception:
          continue

        req_id = msg.get("id")
        if req_id is not None:
          with self._lock:
            pending = self._pending_requests.get(req_id)
          if pending:
            event, result_box = pending
            result_box["message"] = msg
            event.set()
      except Exception:
        break

    self.is_connected = False

  def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Sends a JSON-RPC request and synchronously awaits the response."""
    if not self.process or not self.process.stdin or self.process.poll() is not None:
      raise RuntimeError(f"MCP server '{self.name}' is not running.")

    with self._lock:
      self._req_counter += 1
      req_id = self._req_counter
      event = threading.Event()
      result_box: Dict[str, Any] = {}
      self._pending_requests[req_id] = (event, result_box)

    req = {
      "jsonrpc": "2.0",
      "id": req_id,
      "method": method,
      "params": params or {}
    }

    payload = json.dumps(req, ensure_ascii=False) + "\n"
    try:
      self.process.stdin.write(payload)
      self.process.stdin.flush()
    except Exception as e:
      with self._lock:
        self._pending_requests.pop(req_id, None)
      raise RuntimeError(f"Error writing to MCP server '{self.name}': {e}")

    signaled = event.wait(timeout=self.timeout)
    with self._lock:
      self._pending_requests.pop(req_id, None)

    if not signaled:
      raise TimeoutError(f"Timed out waiting for response from MCP server '{self.name}' ({method}).")

    resp_msg = result_box.get("message", {})
    if "error" in resp_msg:
      err = resp_msg["error"]
      err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
      raise RuntimeError(f"MCP Error ({self.name}): {err_msg}")

    return resp_msg.get("result")

  def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
    """Sends a JSON-RPC notification (no response expected)."""
    if not self.process or not self.process.stdin or self.process.poll() is not None:
      return

    notif = {
      "jsonrpc": "2.0",
      "method": method,
      "params": params or {}
    }
    payload = json.dumps(notif, ensure_ascii=False) + "\n"
    try:
      self.process.stdin.write(payload)
      self.process.stdin.flush()
    except Exception:
      pass

  def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Calls a tool on the MCP server and formats the returned content."""
    params = {
      "name": tool_name,
      "arguments": arguments or {}
    }
    try:
      result = self._send_request("tools/call", params)
    except Exception as e:
      return f"Error executing MCP tool '{tool_name}' on '{self.name}': {str(e)}"

    if not isinstance(result, dict):
      return str(result)

    is_error = bool(result.get("isError", False))
    content_items = result.get("content", [])

    text_parts = []
    for item in content_items:
      if isinstance(item, dict):
        item_type = item.get("type", "text")
        if item_type == "text":
          text_parts.append(item.get("text", ""))
        elif item_type == "image":
          mime = item.get("mimeType", "image/png")
          text_parts.append(f"[Binary Image: {mime}]")
        else:
          text_parts.append(str(item))
      else:
        text_parts.append(str(item))

    output = "\n".join(text_parts) if text_parts else "(No output)"
    if is_error:
      return f"Error: {output}"
    return output

  def close(self):
    """Cleanly terminates the MCP server subprocess."""
    self.is_connected = False
    if self.process:
      proc = self.process
      self.process = None
      try:
        if proc.stdin and not proc.stdin.closed:
          proc.stdin.close()
      except Exception:
        pass
      try:
        proc.terminate()
        proc.wait(timeout=1.5)
      except Exception:
        try:
          proc.kill()
          proc.wait(timeout=1.0)
        except Exception:
          pass
      try:
        if proc.stdout and not proc.stdout.closed:
          proc.stdout.close()
      except Exception:
        pass
      try:
        if proc.stderr and not proc.stderr.closed:
          proc.stderr.close()
      except Exception:
        pass


class MCPManager:
  """Discovers MCP configurations and manages multiple MCPClient instances."""

  def __init__(self, sandbox: Optional[str] = None, config_path: Optional[str] = None):
    self.sandbox = sandbox or os.getcwd()
    self.config_path = config_path
    self.clients: Dict[str, MCPClient] = {}
    self.tool_to_client: Dict[str, Tuple[MCPClient, str]] = {}

  def find_config_files(self) -> List[str]:
    """Finds candidate MCP JSON configuration files."""
    paths = []
    if self.config_path:
      if os.path.exists(self.config_path):
        paths.append(os.path.abspath(self.config_path))
      return paths

    candidates = [
      os.path.join(self.sandbox, ".chatty", "mcp.json"),
      os.path.join(self.sandbox, "mcp.json"),
      os.path.join(self.sandbox, ".mcp.json"),
      os.path.join(os.getcwd(), ".chatty", "mcp.json"),
      os.path.join(os.getcwd(), "mcp.json"),
      os.path.join(os.getcwd(), ".mcp.json"),
      os.path.expanduser("~/.config/chatty/mcp.json"),
      os.path.expanduser("~/.chatty/mcp.json")
    ]

    for p in candidates:
      abs_p = os.path.abspath(p)
      if os.path.isfile(abs_p) and abs_p not in paths:
        paths.append(abs_p)

    return paths

  def load_and_connect(self):
    """Loads MCP configurations and connects to configured servers."""
    config_files = self.find_config_files()
    if not config_files:
      return

    configs = {}
    for p in config_files:
      try:
        with open(p, "r", encoding="utf-8") as f:
          data = json.load(f)
        if isinstance(data, dict):
          servers = data.get("mcpServers", data)
          if isinstance(servers, dict):
            for s_name, s_conf in servers.items():
              if isinstance(s_conf, dict) and s_name not in configs:
                configs[s_name] = s_conf
      except Exception as e:
        logger.warning(f"Error parsing MCP config file '{p}': {e}")

    for s_name, s_conf in configs.items():
      command = s_conf.get("command")
      if not command:
        continue
      args = s_conf.get("args", [])
      env = s_conf.get("env", {})
      cwd = s_conf.get("cwd") or self.sandbox
      timeout = float(s_conf.get("timeout", 30.0))

      client = MCPClient(
        name=s_name,
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        timeout=timeout
      )
      client.connect()
      self.clients[s_name] = client

    self._rebuild_tool_index()

  def _rebuild_tool_index(self):
    """Builds a lookup mapping from tool name to (client, server_tool_name)."""
    self.tool_to_client.clear()
    name_counts: Dict[str, int] = {}

    for client in self.clients.values():
      if not client.is_connected:
        continue
      for tool in client.tools:
        raw_name = tool.get("name")
        if raw_name:
          name_counts[raw_name] = name_counts.get(raw_name, 0) + 1

    for client in self.clients.values():
      if not client.is_connected:
        continue
      for tool in client.tools:
        raw_name = tool.get("name")
        if not raw_name:
          continue

        namespaced_name = f"mcp__{client.name}__{raw_name}"
        self.tool_to_client[namespaced_name] = (client, raw_name)

        # If name is unique across all servers, allow direct short name
        if name_counts.get(raw_name, 0) == 1:
          self.tool_to_client[raw_name] = (client, raw_name)

  def has_tool(self, name: str) -> bool:
    """Returns True if the specified tool is provided by any connected MCP server."""
    return name in self.tool_to_client

  def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
    """Executes a tool on the appropriate MCP server."""
    entry = self.tool_to_client.get(name)
    if not entry:
      return f"Error: MCP tool '{name}' not found."
    client, raw_name = entry
    return client.call_tool(raw_name, arguments)

  def get_tools_schema(self) -> List[Dict[str, Any]]:
    """Returns OpenAI/LiteLLM function tool schemas for all connected MCP tools."""
    schemas = []
    seen_names = set()

    for exposed_name, (client, raw_name) in self.tool_to_client.items():
      # Only emit the preferred name for the tool in the LLM tool list
      # (Prefer direct short name if unique, otherwise namespaced)
      is_namespaced = exposed_name.startswith(f"mcp__{client.name}__")
      if not is_namespaced and exposed_name in seen_names:
        continue
      if is_namespaced and raw_name in self.tool_to_client and self.tool_to_client[raw_name][0] == client:
        # Direct short name is already available and preferred
        continue

      seen_names.add(exposed_name)
      target_tool = next((t for t in client.tools if t.get("name") == raw_name), None)
      desc = target_tool.get("description", "") if target_tool else ""
      input_schema = target_tool.get("inputSchema") if target_tool else None
      if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}}

      server_tag = f" [MCP: {client.name}]"
      schemas.append({
        "type": "function",
        "function": {
          "name": exposed_name,
          "description": (desc + server_tag).strip(),
          "parameters": input_schema
        }
      })

    return schemas

  def get_servers_status(self) -> List[Dict[str, Any]]:
    """Returns a list of status dictionaries for all configured servers."""
    status_list = []
    for name, client in self.clients.items():
      status_list.append({
        "name": name,
        "command": client.command,
        "args": client.args,
        "connected": client.is_connected,
        "tools_count": len(client.tools),
        "tools": [t.get("name") for t in client.tools if t.get("name")],
        "error": client.last_error
      })
    return status_list

  def reload(self):
    """Closes all server connections and reloads configuration."""
    self.close()
    self.clients.clear()
    self.tool_to_client.clear()
    self.load_and_connect()

  def close(self):
    """Closes all connected MCP server subprocesses."""
    for client in self.clients.values():
      client.close()
