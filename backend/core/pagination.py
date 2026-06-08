"""消息分页加载 — 替代当前全量加载会话消息的策略。

当前问题 (agent.py ConversationStorage.load):
- 一次加载会话的所有消息
- 大会话可能导致 OOM 或超高延迟

优化方案:
- 按窗口加载最近 N 条消息
- 支持按偏移量加载历史
- 预取策略：加载当前窗口后，后台预取前一个窗口
"""

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class PageWindow:
    """消息窗口。"""
    messages: list
    total_count: int
    offset: int
    limit: int
    has_more: bool


class MessagePagination:
    """消息分页加载器。

    用法:
        pagination = MessagePagination(load_fn=storage.load_range)
        window = pagination.load_recent("session_1", window_size=50)
        older = pagination.load_older("session_1", from_offset=50)
    """

    def __init__(
        self,
        load_fn: Optional[Callable] = None,
        count_fn: Optional[Callable] = None,
        default_window: int = 50,
        prefetch_enabled: bool = False,
    ):
        self._load_fn = load_fn
        self._count_fn = count_fn
        self.default_window = default_window
        self.prefetch_enabled = prefetch_enabled
        self._cache: dict = {}

    def set_load_fn(self, load_fn: Callable):
        """设置加载函数。"""
        self._load_fn = load_fn

    def set_count_fn(self, count_fn: Callable):
        """设置计数函数。"""
        self._count_fn = count_fn

    def load_recent(
        self,
        session_id: str,
        window_size: Optional[int] = None,
    ) -> PageWindow:
        """加载最近的消息窗口。"""
        size = window_size or self.default_window
        total = self._get_total_count(session_id)

        if self._load_fn:
            messages = self._load_fn(session_id, offset=max(0, total - size), limit=size)
        else:
            messages = []

        self._cache[session_id] = {
            "loaded_offset": max(0, total - size),
            "loaded_limit": size,
        }

        return PageWindow(
            messages=messages,
            total_count=total,
            offset=max(0, total - size),
            limit=size,
            has_more=total > size,
        )

    def load_older(
        self,
        session_id: str,
        from_offset: int,
        window_size: Optional[int] = None,
    ) -> PageWindow:
        """加载更早的消息。"""
        size = window_size or self.default_window
        total = self._get_total_count(session_id)

        offset = max(0, from_offset - size)

        if self._load_fn:
            messages = self._load_fn(session_id, offset=offset, limit=size)
        else:
            messages = []

        return PageWindow(
            messages=messages,
            total_count=total,
            offset=offset,
            limit=size,
            has_more=offset > 0,
        )

    def _get_total_count(self, session_id: str) -> int:
        """获取会话总消息数。通过 count_fn 外部注入查询。"""
        if self._count_fn:
            return self._count_fn(session_id)
        if self._load_fn:
            raise RuntimeError(
                "MessagePagination requires a count_fn for accurate pagination. "
                "Set one via set_count_fn() or the constructor."
            )
        return 0

    def clear_cache(self, session_id: str):
        """清除会话的缓存状态。"""
        self._cache.pop(session_id, None)
