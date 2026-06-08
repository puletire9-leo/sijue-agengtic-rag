"""增量保存优化 — 替代当前全量删除再插入的会话保存策略。

当前问题 (agent.py ConversationStorage.save):
- 每次保存先 DELETE 再 INSERT 全部消息
- 会话长时效率低，DB 产生大量 WAL

优化方案:
- 追踪已保存的消息 ID，只插入新增消息
- 不删除旧消息，避免全表扫描
"""

import threading
from collections import OrderedDict
from typing import Dict, Set


class IncrementalSaveTracker:
    """增量保存追踪器 — 记录已保存的消息摘要，避免重复保存。"""

    def __init__(self, max_sessions: int = 1000):
        self._saved_hashes: Dict[str, Set[str]] = {}  # session_id → {content_sha256, ...}
        self._access_order: OrderedDict[str, None] = OrderedDict()  # LRU tracking: most recent at end
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def _touch_session(self, session_id: str) -> None:
        """Update LRU access order for a session. Must be called when session is accessed."""
        if session_id in self._access_order:
            self._access_order.move_to_end(session_id)
        else:
            self._access_order[session_id] = None

    def _evict_if_needed(self) -> None:
        """Evict oldest sessions when the dict exceeds max size."""
        while len(self._saved_hashes) > self._max_sessions and self._access_order:
            oldest, _ = self._access_order.popitem(last=False)
            self._saved_hashes.pop(oldest, None)

    @staticmethod
    def _base_hash(msg, index: int) -> str:
        """Compute base hash from index + type + content (position-aware)."""
        import hashlib
        content = f"{index}:{getattr(msg, 'type', '')}:{getattr(msg, 'content', '')}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_unsaved_messages(self, session_id: str, messages: list) -> tuple[list[int], list]:
        """返回尚未保存的新消息及其在原列表中的索引。

        不修改 _saved_hashes，由 mark_saved 在 commit 成功后统一标记。
        返回: (unsaved_indices, unsaved_messages)
        """
        with self._lock:
            self._touch_session(session_id)
            saved = self._saved_hashes.get(session_id, set())
            unsaved_indices: list[int] = []
            unsaved: list = []

            for idx, msg in enumerate(messages):
                content_hash = self._base_hash(msg, idx)
                if content_hash not in saved:
                    unsaved_indices.append(idx)
                    unsaved.append(msg)

            self._evict_if_needed()
            return unsaved_indices, unsaved

    def mark_saved(self, session_id: str, messages: list, indices: list[int] = None):
        """将消息标记为已保存。indices 提供每条消息在原始列表中的位置；为 None 时按 enumerate 自动编号。"""
        with self._lock:
            self._touch_session(session_id)
            saved = self._saved_hashes.get(session_id, set())
            new_hashes: Set[str] = set()
            for idx, msg in enumerate(messages):
                real_index = indices[idx] if indices and idx < len(indices) else idx
                content_hash = self._base_hash(msg, real_index)
                new_hashes.add(content_hash)
            self._saved_hashes[session_id] = saved | new_hashes
            self._evict_if_needed()

    def reset_session(self, session_id: str):
        """重置会话的追踪状态。"""
        with self._lock:
            self._saved_hashes.pop(session_id, None)
            self._access_order.pop(session_id, None)
