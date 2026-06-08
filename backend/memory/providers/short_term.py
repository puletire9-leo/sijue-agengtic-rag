"""ShortTermMemoryProvider — PostgreSQL FTS 短期记忆后端。

使用 PostgreSQL 全文搜索实现关键词记忆召回。
默认保留 30 天。
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Index, create_engine, func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import JSON

from database import Base as _Base, SessionLocal
from memory.providers.base import MemoryItem, MemoryProvider


class ShortTermMemoryRow(_Base):
    __tablename__ = "short_term_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(String(256), nullable=False, index=True)
    type = Column(String(64), default="")
    content = Column(Text, default="")
    metadata_json = Column(JSON, default=dict)
    importance = Column(Float, default=0.5)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    source = Column(String(64), default="")

    __table_args__ = (
        Index("idx_stm_user_time", "user_id", "timestamp"),
    )


class ShortTermMemoryProvider(MemoryProvider):
    """PostgreSQL FTS 短期记忆。

    依赖 PostgreSQL tsvector 进行全文检索。
    需要在数据库中创建表，可通过 create_tables() 初始化。
    """

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days

    @staticmethod
    def create_tables(engine=None):
        """创建记忆表。"""
        _Base.metadata.create_all(engine or create_engine("sqlite:///:memory:"))

    async def save(self, user_id: str, item: MemoryItem) -> str:
        if not item.id:
            item.id = str(uuid.uuid4())[:12]
        if not item.timestamp:
            item.timestamp = datetime.now(timezone.utc).isoformat()

        db = SessionLocal()
        try:
            row = ShortTermMemoryRow(
                memory_id=item.id,
                user_id=user_id,
                type=item.type,
                content=item.content,
                metadata_json=item.metadata,
                importance=item.importance,
                timestamp=self._parse_timestamp(item.timestamp),
                source=item.source,
            )
            db.add(row)
            db.commit()
            return item.id
        finally:
            db.close()

    async def save_batch(self, user_id: str, items: List[MemoryItem]) -> List[str]:
        db = SessionLocal()
        try:
            ids = []
            for item in items:
                if not item.id:
                    item.id = str(uuid.uuid4())[:12]
                if not item.timestamp:
                    item.timestamp = datetime.now(timezone.utc).isoformat()
                ts = self._parse_timestamp(item.timestamp)
                row = ShortTermMemoryRow(
                    memory_id=item.id,
                    user_id=user_id,
                    type=item.type,
                    content=item.content,
                    metadata_json=item.metadata,
                    importance=item.importance,
                    timestamp=ts,
                    source=item.source,
                )
                db.add(row)
                ids.append(item.id)
            db.commit()
            return ids
        finally:
            db.close()

    async def recall(self, user_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        """使用 PostgreSQL LIKE 近似搜索（降级方案）。

        完整 FTS 需要数据库 init_db 时创建 tsvector 列和触发器。
        此处先用 ILIKE 提供基础功能。
        """
        db = SessionLocal()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            escaped_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped_query}%"
            rows = (
                db.query(ShortTermMemoryRow)
                .filter(
                    ShortTermMemoryRow.user_id == user_id,
                    ShortTermMemoryRow.timestamp >= cutoff,
                    (ShortTermMemoryRow.content.ilike(pattern, escape="\\"))
                    | (ShortTermMemoryRow.type.ilike(pattern, escape="\\")),
                )
                .order_by(ShortTermMemoryRow.importance.desc(), ShortTermMemoryRow.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [self._to_memory_item(row) for row in rows]
        finally:
            db.close()

    async def get_recent(self, user_id: str, limit: int = 20) -> List[MemoryItem]:
        db = SessionLocal()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            rows = (
                db.query(ShortTermMemoryRow)
                .filter(
                    ShortTermMemoryRow.user_id == user_id,
                    ShortTermMemoryRow.timestamp >= cutoff,
                )
                .order_by(ShortTermMemoryRow.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [self._to_memory_item(row) for row in rows]
        finally:
            db.close()

    async def forget(self, user_id: str, memory_id: str) -> bool:
        db = SessionLocal()
        try:
            row = (
                db.query(ShortTermMemoryRow)
                .filter(
                    ShortTermMemoryRow.memory_id == memory_id,
                    ShortTermMemoryRow.user_id == user_id,
                )
                .first()
            )
            if row:
                db.delete(row)
                db.commit()
                return True
            return False
        finally:
            db.close()

    async def clear_user(self, user_id: str) -> int:
        db = SessionLocal()
        try:
            count = (
                db.query(ShortTermMemoryRow)
                .filter(ShortTermMemoryRow.user_id == user_id)
                .count()
            )
            db.query(ShortTermMemoryRow).filter(
                ShortTermMemoryRow.user_id == user_id
            ).delete()
            db.commit()
            return count
        finally:
            db.close()

    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            count = (
                db.query(ShortTermMemoryRow)
                .filter(ShortTermMemoryRow.user_id == user_id)
                .count()
            )
            return {
                "provider": "short_term",
                "layer": "postgresql_fts",
                "count": count,
                "retention_days": self.retention_days,
            }
        finally:
            db.close()

    @staticmethod
    def _parse_timestamp(ts_str: str) -> datetime:
        """Parse ISO timestamp preserving timezone, then convert to naive UTC for DB storage."""
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        parsed = datetime.fromisoformat(ts_str)
        if parsed.tzinfo:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def _to_memory_item(self, row: ShortTermMemoryRow) -> MemoryItem:
        return MemoryItem(
            id=row.memory_id,
            user_id=row.user_id,
            type=row.type or "",
            content=row.content or "",
            metadata=row.metadata_json or {},
            importance=row.importance or 0.5,
            timestamp=row.timestamp.isoformat() if row.timestamp else None,
            source=row.source or "",
        )
