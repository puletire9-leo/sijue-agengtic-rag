"""MemoryManager — 三层记忆协调器。

协调 Layer 0 (会话) / Layer 1 (短期) / Layer 2 (长期) 三层记忆的读写。
"""

from typing import Any, Dict, List, Optional

from memory.providers.base import MemoryItem, MemoryProvider


class MemoryManager:
    """三层记忆协调器。

    用法:
        mgr = MemoryManager(session_provider, short_term_provider, long_term_provider)
        memories = await mgr.recall("user1", "用户偏好")
        await mgr.save("user1", MemoryItem(type="preference", content="喜欢简洁回答"))
    """

    def __init__(
        self,
        session_provider: Optional[MemoryProvider] = None,
        short_term_provider: Optional[MemoryProvider] = None,
        long_term_provider: Optional[MemoryProvider] = None,
    ):
        self.session = session_provider      # Layer 0: Redis
        self.short_term = short_term_provider  # Layer 1: PostgreSQL FTS
        self.long_term = long_term_provider    # Layer 2: PostgreSQL + 向量

    async def save(self, user_id: str, item: MemoryItem, layer: Optional[str] = None) -> str:
        """保存记忆到指定层。

        Args:
            user_id: 用户 ID
            item: 记忆条目
            layer: 目标层 (session/short_term/long_term)，None 自动判断

        Returns:
            记忆 ID
        """
        provider, _ = self._resolve_provider(item, layer)
        if provider:
            return await provider.save(user_id, item)
        return ""

    async def save_batch(self, user_id: str, items: List[MemoryItem]) -> List[str]:
        """批量保存记忆。按 provider 分组后调用 save_batch()。"""
        from collections import defaultdict
        grouped: Dict[str, List[MemoryItem]] = defaultdict(list)
        for item in items:
            _, layer = self._resolve_provider(item)
            grouped[layer].append(item)

        ids: List[str] = []
        for layer, group_items in grouped.items():
            provider = self._get_provider(layer)
            if provider:
                batch_ids = await provider.save_batch(user_id, group_items)
                ids.extend(batch_ids)
        return ids

    async def recall(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        include_layers: Optional[List[str]] = None,
    ) -> List[MemoryItem]:
        """跨层召回记忆。

        Args:
            user_id: 用户 ID
            query: 查询文本
            limit: 返回上限
            include_layers: 包含的层 (默认全部)

        Returns:
            按相关度排序的记忆列表
        """
        layers = include_layers or ["session", "short_term", "long_term"]
        results: List[MemoryItem] = []

        for layer in layers:
            provider = self._get_provider(layer)
            if provider:
                items = await provider.recall(user_id, query, limit)
                results.extend(items)

        # 按重要性排序
        results.sort(key=lambda x: x.importance, reverse=True)
        return results[:limit]

    async def get_recent(
        self,
        user_id: str,
        limit: int = 20,
        layer: str = "session",
    ) -> List[MemoryItem]:
        """获取最近的记忆。"""
        provider = self._get_provider(layer)
        if provider:
            return await provider.get_recent(user_id, limit)
        return []

    async def forget(self, user_id: str, memory_id: str) -> bool:
        """删除指定记忆。"""
        for provider in [self.session, self.short_term, self.long_term]:
            if provider:
                try:
                    if await provider.forget(user_id, memory_id):
                        return True
                except Exception:
                    continue
        return False

    async def clear_user(self, user_id: str) -> int:
        """清除用户所有记忆。"""
        total = 0
        for provider in [self.session, self.short_term, self.long_term]:
            if provider:
                total += await provider.clear_user(user_id)
        return total

    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取记忆统计。"""
        stats: Dict[str, Any] = {}
        for layer_name, provider in [
            ("session", self.session),
            ("short_term", self.short_term),
            ("long_term", self.long_term),
        ]:
            if provider:
                stats[layer_name] = await provider.get_stats(user_id)
            else:
                stats[layer_name] = {"status": "not_configured"}
        return stats

    def _resolve_provider(self, item: MemoryItem, layer: Optional[str] = None):
        """解析记忆条目应保存到哪一层。"""
        if layer:
            provider = self._get_provider(layer)
            if provider is None:
                provider = self.session
                layer = "session"
            return provider, layer

        # 自动判断
        if item.importance >= 0.8:
            if self.long_term:
                return self.long_term, "long_term"
            return self.short_term, "short_term"
        elif item.importance >= 0.4:
            if self.short_term:
                return self.short_term, "short_term"
            return self.session, "session"
        else:
            return self.session, "session"

    def _get_provider(self, layer: str) -> Optional[MemoryProvider]:
        """根据层名称获取 provider。"""
        return {
            "session": self.session,
            "short_term": self.short_term,
            "long_term": self.long_term,
        }.get(layer)
