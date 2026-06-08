"""工具系统 — ToolRegistry 自注册 + MCP 客户端扩展。

不与 backend/tools.py 冲突（独立包名）。
"""

from tool_system.registry import ToolRegistry, RegisteredTool, tool_registry
from tool_system.mcp_client import MCPClient, MCPTool, MCPToolResult

__all__ = [
    "ToolRegistry", "RegisteredTool", "tool_registry",
    "MCPClient", "MCPTool", "MCPToolResult",
]
