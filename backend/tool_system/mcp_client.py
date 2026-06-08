"""MCP Client — Model Context Protocol 客户端。

v3: 修复 SSE 传输——维护持久 HTTP 会话，支持工具发现和调用。
支持两种传输:
  - stdio: 子进程通信（本地 MCP 服务器）
  - SSE (HTTP): 远程 MCP 服务器
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP 工具描述。"""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolResult:
    """MCP 工具调用结果。"""
    success: bool
    content: str = ""
    error: Optional[str] = None


class MCPClient:
    """MCP 协议客户端。

    支持 stdio（子进程）和 HTTP SSE 两种传输。
    自动选择 _discover_tools 和 call_tool 的传输方式。
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._http_session = None       # aiohttp.ClientSession（SSE 传输）
        self._server_url: str = ""      # SSE 服务器 URL
        self._session_id: Optional[str] = None
        self._connected = False
        self._transport: str = ""       # "stdio" | "sse"
        self._tools: Dict[str, MCPTool] = {}
        self._request_id: int = 0

    # ── stdio 传输 ──

    async def connect_stdio(self, command: str, args: Optional[List[str]] = None) -> bool:
        """启动本地 MCP 服务器子进程，通过 stdin/stdout 通信。"""
        try:
            self._process = subprocess.Popen(
                [command] + (args or []),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._session_id = f"stdio:{command}"
            self._transport = "stdio"

            response = await self._stdio_request("initialize", {
                "protocolVersion": "0.1.0",
                "clientInfo": {"name": "SuperMew", "version": "0.1.0"},
            })

            if response and response.get("result"):
                self._connected = True
                await self._discover_tools()
                return True
            return False
        except Exception:
            self._connected = False
            return False

    # ── SSE (HTTP) 传输 ──

    async def connect_sse(self, server_url: str) -> bool:
        """通过 HTTP JSON-RPC 连接远程 MCP 服务器。

        维护持久 aiohttp.ClientSession，支持多轮工具调用。
        """
        try:
            import aiohttp
            self._server_url = server_url
            self._session_id = f"sse:{server_url}"
            self._transport = "sse"

            # 创建持久 HTTP 会话
            self._http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30, connect=10))

            # 发送 initialize 请求
            resp = await self._sse_request("initialize", {
                "protocolVersion": "0.1.0",
                "clientInfo": {"name": "SuperMew", "version": "0.1.0"},
            })

            if resp and resp.get("result"):
                self._connected = True
                await self._discover_tools()
                return True

            await self._http_session.close()
            self._http_session = None
            return False
        except ImportError:
            self._connected = False
            return False
        except Exception:
            if self._http_session:
                await self._http_session.close()
                self._http_session = None
            self._connected = False
            return False

    # ── 工具发现 ──

    async def list_tools(self) -> List[MCPTool]:
        """列出可用的 MCP 工具。"""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """按名称获取 MCP 工具。"""
        return self._tools.get(name)

    # ── 工具调用 ──

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """调用 MCP 工具（自动选择传输方式）。"""
        if not self._connected:
            return MCPToolResult(success=False, error="MCP 客户端未连接")

        if tool_name not in self._tools:
            return MCPToolResult(success=False, error=f"工具 {tool_name} 未找到")

        try:
            if self._transport == "sse":
                response = await self._sse_request("tools/call", {
                    "name": tool_name,
                    "arguments": arguments,
                })
            elif self._transport == "stdio":
                response = await self._stdio_request("tools/call", {
                    "name": tool_name,
                    "arguments": arguments,
                })
            else:
                return MCPToolResult(success=False, error="未知传输方式")

            if response and response.get("result"):
                content = response["result"].get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        c.get("text", str(c)) for c in content if isinstance(c, dict)
                    )
                return MCPToolResult(success=True, content=str(content))
            else:
                error = (response.get("error") or {}).get("message", "未知错误") if response else "无响应"
                return MCPToolResult(success=False, error=error)
        except Exception as e:
            return MCPToolResult(success=False, error=str(e))

    # ── 生命周期 ──

    async def disconnect(self):
        """断开 MCP 连接。"""
        if self._transport == "stdio" and self._process:
            try:
                self._process.stdin.close()
                self._process.stdout.close()
                self._process.terminate()
                await asyncio.get_running_loop().run_in_executor(None, self._process.wait, 5)
            except Exception:
                self._process.kill()
                try:
                    await asyncio.get_running_loop().run_in_executor(None, self._process.wait, 5)
                except Exception:
                    pass
            finally:
                self._process = None
        elif self._transport == "sse" and self._http_session:
            await self._http_session.close()
            self._http_session = None

        self._connected = False
        self._session_id = None
        self._transport = ""
        self._tools.clear()
        await asyncio.sleep(0)

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── 内部方法 ──

    async def _stdio_request(self, method: str, params: dict) -> Optional[dict]:
        """通过 stdio 发送 JSON-RPC 请求。"""
        if not self._process or not self._process.stdin:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id,
        }

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: (self._process.stdin.write(json.dumps(request) + "\n"), self._process.stdin.flush())
            )
            line = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None, self._process.stdout.readline
                ),
                timeout=30,
            )
            if line:
                return json.loads(line.strip())
        except asyncio.TimeoutError:
            logger.warning("MCP stdio request timed out: method=%s id=%d", method, self._request_id)
        except Exception:
            pass
        return None

    async def _sse_request(self, method: str, params: dict) -> Optional[dict]:
        """通过 HTTP 发送 JSON-RPC 请求（SSE 模式）。"""
        if not self._http_session:
            return None

        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._request_id,
        }

        try:
            async with self._http_session.post(
                self._server_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status >= 400:
                    return None
                return await resp.json()
        except Exception:
            return None

    async def _discover_tools(self):
        """发现 MCP 服务器提供的工具列表。"""
        if self._transport == "stdio":
            response = await self._stdio_request("tools/list", {})
        elif self._transport == "sse":
            response = await self._sse_request("tools/list", {})
        else:
            return

        if response and response.get("result"):
            tools_data = response["result"].get("tools", [])
            for td in tools_data:
                tool = MCPTool(
                    name=td.get("name", ""),
                    description=td.get("description", ""),
                    parameters=td.get("inputSchema", td.get("parameters", {})),
                )
                self._tools[tool.name] = tool
