"""
MCP Client for Singapore Travel Assistant
Manages connections to MCP servers (weather and currency) via subprocess stdio.
Provides tool invocation with error handling.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional


MCP_SERVERS_DIR = Path(__file__).parent.parent / "mcp_servers"


class MCPClient:
    """
    A simple MCP client that communicates with an MCP server via stdio (subprocess).
    """

    def __init__(self, server_script: str, server_name: str):
        self.server_script = server_script
        self.server_name = server_name
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self._lock = threading.Lock()
        self.available_tools: list = []
        self._connected = False

    def connect(self) -> bool:
        """Start the MCP server subprocess and initialize the connection."""
        try:
            self.process = subprocess.Popen(
                [sys.executable, self.server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Send initialize request
            init_response = self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "singapore-travel-assistant", "version": "1.0.0"},
                },
            )

            if "result" not in init_response:
                return False

            # Send initialized notification
            self._send_notification("notifications/initialized", {})

            # Get available tools
            tools_response = self._send_request("tools/list", {})
            if "result" in tools_response:
                self.available_tools = tools_response["result"].get("tools", [])

            self._connected = True
            return True

        except Exception as e:
            print(f"⚠️ Failed to connect to MCP server '{self.server_name}': {e}")
            return False

    def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and read the response."""
        with self._lock:
            self.request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params,
            }

            if self.process is None or self.process.poll() is not None:
                return {"error": "Server process not running"}

            try:
                line = json.dumps(request) + "\n"
                self.process.stdin.write(line)
                self.process.stdin.flush()

                response_line = self.process.stdout.readline()
                if not response_line:
                    return {"error": "No response from server"}

                return json.loads(response_line.strip())

            except Exception as e:
                return {"error": str(e)}

    def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if self.process is None or self.process.poll() is not None:
            return
        try:
            notification = {"jsonrpc": "2.0", "method": method, "params": params}
            self.process.stdin.write(json.dumps(notification) + "\n")
            self.process.stdin.flush()
        except Exception:
            pass

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call an MCP tool and return the result.
        Returns a dict with 'success', 'content', and optionally 'error'.
        """
        if not self._connected:
            return {
                "success": False,
                "error": f"MCP server '{self.server_name}' is not connected.",
                "content": "",
            }

        response = self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

        if "error" in response and "result" not in response:
            return {
                "success": False,
                "error": response["error"].get("message", str(response["error"])),
                "content": "",
            }

        result = response.get("result", {})
        is_error = result.get("isError", False)
        content_list = result.get("content", [])
        content_text = "\n".join(
            item.get("text", "") for item in content_list if item.get("type") == "text"
        )

        return {
            "success": not is_error,
            "content": content_text,
            "error": content_text if is_error else None,
        }

    def get_tool_names(self) -> list:
        """Return list of available tool names."""
        return [t["name"] for t in self.available_tools]

    def disconnect(self) -> None:
        """Terminate the MCP server subprocess."""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self._connected = False


class MCPToolManager:
    """
    Manages multiple MCP server connections and provides a unified interface
    for tool invocation across weather and currency servers.
    """

    def __init__(self):
        self.clients: dict[str, MCPClient] = {}
        self.tool_to_client: dict[str, str] = {}  # tool_name -> client_name
        self._initialized = False

    def initialize(self) -> dict:
        """
        Start all MCP servers and register their tools.
        Returns status dict with connection results.
        """
        status = {}

        # Weather server
        weather_script = str(MCP_SERVERS_DIR / "weather_server.py")
        if Path(weather_script).exists():
            weather_client = MCPClient(weather_script, "weather")
            connected = weather_client.connect()
            status["weather"] = connected
            if connected:
                self.clients["weather"] = weather_client
                for tool_name in weather_client.get_tool_names():
                    self.tool_to_client[tool_name] = "weather"
                print(f"✅ Weather MCP server connected. Tools: {weather_client.get_tool_names()}")
            else:
                print("⚠️ Weather MCP server failed to connect.")
        else:
            status["weather"] = False
            print(f"⚠️ Weather server script not found: {weather_script}")

        # Currency server
        currency_script = str(MCP_SERVERS_DIR / "currency_server.py")
        if Path(currency_script).exists():
            currency_client = MCPClient(currency_script, "currency")
            connected = currency_client.connect()
            status["currency"] = connected
            if connected:
                self.clients["currency"] = currency_client
                for tool_name in currency_client.get_tool_names():
                    self.tool_to_client[tool_name] = "currency"
                print(f"✅ Currency MCP server connected. Tools: {currency_client.get_tool_names()}")
            else:
                print("⚠️ Currency MCP server failed to connect.")
        else:
            status["currency"] = False
            print(f"⚠️ Currency server script not found: {currency_script}")

        self._initialized = True
        return status

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Route a tool call to the appropriate MCP client.
        """
        if tool_name not in self.tool_to_client:
            return {
                "success": False,
                "error": f"Unknown tool '{tool_name}'. Available tools: {list(self.tool_to_client.keys())}",
                "content": "",
            }

        client_name = self.tool_to_client[tool_name]
        client = self.clients.get(client_name)

        if client is None:
            return {
                "success": False,
                "error": f"MCP server for '{tool_name}' is not available.",
                "content": "",
            }

        return client.call_tool(tool_name, arguments)

    def get_all_tools(self) -> list:
        """Return all available tools across all connected servers."""
        tools = []
        for client in self.clients.values():
            tools.extend(client.available_tools)
        return tools

    def get_all_tool_names(self) -> list:
        """Return all available tool names."""
        return list(self.tool_to_client.keys())

    def get_tools_description(self) -> str:
        """Return a formatted description of all available tools for the LLM prompt."""
        lines = []
        for client_name, client in self.clients.items():
            for tool in client.available_tools:
                name = tool["name"]
                desc = tool.get("description", "")
                props = tool.get("inputSchema", {}).get("properties", {})
                param_str = ", ".join(
                    f"{k} ({v.get('type', 'any')}): {v.get('description', '')}"
                    for k, v in props.items()
                )
                lines.append(f"- **{name}**: {desc}")
                if param_str:
                    lines.append(f"  Parameters: {param_str}")
        return "\n".join(lines) if lines else "No MCP tools available."

    def is_connected(self) -> bool:
        """Check if at least one MCP server is connected."""
        return bool(self.clients)

    def shutdown(self) -> None:
        """Disconnect all MCP servers."""
        for client in self.clients.values():
            client.disconnect()
        self.clients.clear()
        self.tool_to_client.clear()
