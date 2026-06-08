"""MemoryProvider 抽象基类 — 定义记忆系统的统一接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    """记忆条目。"""
    id: Optional[str] = None
    user_id: str = ""
    type: str = ""           # "preference" | "fact" | "task" | "entity"
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # 0~1
    timestamp: Optional[str] = None
    source: Optional[str] = None  # "chat" | "extract" | "manual"


class MemoryProvider(ABC):
    """记忆存储提供者抽象基类。"""

    @abstractmethod
    async def save(self, user_id: str, item: MemoryItem) -> str:
        """保存一条记忆。返回记忆 ID。"""
        ...

    @abstractmethod
    async def save_batch(self, user_id: str, items: List[MemoryItem]) -> List[str]:
        """批量保存记忆。返回记忆 ID 列表。"""
        ...

    @abstractmethod
    async def recall(self, user_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        """根据查询召回相关记忆。"""
        ...

    @abstractmethod
    async def get_recent(self, user_id: str, limit: int = 20) -> List[MemoryItem]:
        """获取最近的记忆。"""
        ...

    @abstractmethod
    async def forget(self, user_id: str, memory_id: str) -> bool:
        """删除一条记忆。"""
        ...

    @abstractmethod
    async def clear_user(self, user_id: str) -> int:
        """清除用户的所有记忆。返回删除条数。"""
        ...

    @abstractmethod
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取记忆统计。"""
        ...
