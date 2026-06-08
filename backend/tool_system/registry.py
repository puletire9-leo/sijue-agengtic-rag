"""ToolRegistry — 工具自注册与发现。

支持:
  - 装饰器 @register_tool 自动注册
  - 按名称查找工具
  - 获取所有工具的 LangChain tool spec
  - 启用/禁用工具

用法:
    registry = ToolRegistry()

    @registry.register("search_knowledge_base", description="搜索知识库")
    def search(query: str) -> str: ...

    tool = registry.get("search_knowledge_base")
    result = tool.fn("什么是 RAG？")
"""

from typing import Any, Callable, Dict, List, Optional


class RegisteredTool:
    """已注册的工具。"""

    def __init__(self, name: str, fn: Callable, description: str = "",
                 enabled: bool = True, category: str = "general",
                 tags: Optional[List[str]] = None):
        self.name = name
        self.fn = fn
        self.description = description
        self.enabled = enabled
        self.category = category
        self.tags = tags or []
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        """调用工具函数。"""
        self.call_count += 1
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return f"RegisteredTool({self.name}, enabled={self.enabled}, category={self.category})"


class ToolRegistry:
    """工具注册中心。

    集中的工具管理：注册、查找、启用/禁用、统计。
    """

    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, name: str, description: str = "",
                 category: str = "general", enabled: bool = True,
                 tags: Optional[List[str]] = None):
        """装饰器：注册一个工具。

        Usage:
            @registry.register("get_weather", description="获取天气")
            def get_weather(location: str) -> str: ...
        """
        def wrapper(fn: Callable):
            self._tools[name] = RegisteredTool(
                name=name, fn=fn, description=description,
                enabled=enabled, category=category, tags=tags,
            )
            return self._tools[name]
        return wrapper

    def register_direct(self, name: str, fn: Callable, **kwargs):
        """直接注册（非装饰器）。"""
        self._tools[name] = RegisteredTool(name=name, fn=fn, **kwargs)

    def get(self, name: str) -> Optional[RegisteredTool]:
        """获取工具。"""
        return self._tools.get(name)

    def list_all(self) -> List[RegisteredTool]:
        """列出所有工具。"""
        return list(self._tools.values())

    def list_enabled(self) -> List[RegisteredTool]:
        """列出所有启用的工具。"""
        return [t for t in self._tools.values() if t.enabled]

    def list_by_category(self, category: str) -> List[RegisteredTool]:
        """按分类列出工具。"""
        return [t for t in self._tools.values() if t.category == category]

    def enable(self, name: str):
        """启用工具。"""
        if name in self._tools:
            self._tools[name].enabled = True

    def disable(self, name: str):
        """禁用工具。"""
        if name in self._tools:
            self._tools[name].enabled = False

    def get_stats(self) -> Dict[str, Any]:
        """获取注册统计。"""
        total = len(self._tools)
        enabled = len(self.list_enabled())
        categories = {}
        for t in self._tools.values():
            categories[t.category] = categories.get(t.category, 0) + 1
        return {
            "total_tools": total,
            "enabled_tools": enabled,
            "disabled_tools": total - enabled,
            "categories": categories,
            "call_counts": {name: t.call_count for name, t in self._tools.items()},
        }


# 全局注册表
tool_registry = ToolRegistry()
