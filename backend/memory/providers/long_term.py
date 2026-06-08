"""LongTermMemoryProvider — PostgreSQL + 向量相似度长期记忆后端。

永久存储用户偏好和关键事实，支持语义检索。
"""

import logging
import math
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_MEMORIES_PER_USER = int(os.getenv("LONG_TERM_MEMORY_MAX", "10000"))

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Index, func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import JSON

from database import Base as _LongBase, SessionLocal
from memory.providers.base import MemoryItem, MemoryProvider


def _get_embedding(text: str) -> Optional[List[float]]:
    """使用 BGE-M3 生成文本嵌入向量。"""
    try:
        from embedding import embedding_service
        embeddings = embedding_service.get_embeddings([text])
        return embeddings[0] if embeddings else None
    except Exception:
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LongTermMemoryProvider(MemoryProvider):
    """PostgreSQL + BGE-M3 语义向量长期记忆。

    v3: 保存时生成嵌入向量（存储为 JSON），召回时计算余弦相似度。
    不使用 pgvector——在应用层做向量比对，适合用户级数据量。
    """

    def __init__(self, model=None):
        self.model = model

    async def save(self, user_id: str, item: MemoryItem) -> str:
        if not item.id:
            item.id = str(uuid.uuid4())[:12]
        if not item.timestamp:
            item.timestamp = datetime.now(timezone.utc).isoformat()

        # 生成嵌入向量
        embedding = _get_embedding(item.content)

        db = SessionLocal()
        try:
            self._evict_if_needed(user_id, db)
            row = LongTermMemoryRow(
                memory_id=item.id,
                user_id=user_id,
                type=item.type,
                content=item.content,
                metadata_json=item.metadata,
                importance=item.importance,
                timestamp=self._parse_timestamp(item.timestamp),
                source=item.source,
                embedding_json=embedding,
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
            # Pre-assign IDs and timestamps
            for item in items:
                if not item.id:
                    item.id = str(uuid.uuid4())[:12]
                if not item.timestamp:
                    item.timestamp = datetime.now(timezone.utc).isoformat()
                ids.append(item.id)
            # Batch embed all content in a single API call
            contents = [item.content for item in items]
            try:
                from embedding import embedding_service
                embeddings = embedding_service.get_embeddings(contents)
            except Exception:
                embeddings = [None] * len(items)
            for item, embedding in zip(items, embeddings):
                ts = self._parse_timestamp(item.timestamp)
                row = LongTermMemoryRow(
                    memory_id=item.id,
                    user_id=user_id,
                    type=item.type,
                    content=item.content,
                    metadata_json=item.metadata,
                    importance=item.importance,
                    timestamp=ts,
                    source=item.source,
                    embedding_json=embedding,
                )
                db.add(row)
            self._evict_if_needed(user_id, db)
            db.commit()
            return ids
        finally:
            db.close()

    async def recall(self, user_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        """语义向量检索 + 重要性加权排序。

        v3: 使用 BGE-M3 生成查询嵌入，与存储的记忆向量做余弦相似度比对。
        降级路径: embedding 不可用 → ILIKE 关键词匹配 → 重要性排序。
        """
        db = SessionLocal()
        try:
            # 加载用户所有长期记忆（用户级数据量通常 < 1000，可全量比对）
            rows = (
                db.query(LongTermMemoryRow)
                .filter(LongTermMemoryRow.user_id == user_id)
                .order_by(LongTermMemoryRow.timestamp.desc())
                .limit(500)  # 安全上限
                .all()
            )

            if not rows:
                return []

            # 尝试向量检索
            query_embedding = _get_embedding(query)

            if query_embedding:
                # 向量语义检索 + 时间衰减 + 重要性加权
                now = datetime.now(timezone.utc)
                scored = []
                for row in rows:
                    emb = row.embedding_json
                    if emb and isinstance(emb, list) and len(emb) > 0:
                        final_score = self._compute_recall_score(row, query_embedding, now)
                    else:
                        # 无嵌入的旧数据用关键词降级
                        sim = 0.3 if query.lower() in (row.content or "").lower() else 0.0
                        days_passed = (now - (row.timestamp or now)).days
                        time_decay = math.exp(-0.008 * days_passed)
                        final_score = sim * time_decay * 0.6 + (row.importance or 0.5) * 0.4
                    scored.append((final_score, row))

                scored.sort(key=lambda x: x[0], reverse=True)
                result = [self._to_memory_item(row) for _, row in scored[:limit]]
                # Update last_accessed for recalled memories
                recalled_ids = [row.memory_id for _, row in scored[:limit]]
                if recalled_ids:
                    try:
                        self._touch_memories(recalled_ids, db)
                    except Exception:
                        logger.warning("Failed to update last_accessed for memories: %s", recalled_ids, exc_info=True)
                return result

            # 降级: ILIKE 关键词匹配
            pattern = f"%{query}%"
            matched = [
                row for row in rows
                if pattern.replace("%", "").lower() in (row.content or "").lower()
                   or pattern.replace("%", "").lower() in (row.type or "").lower()
            ]
            if matched:
                matched.sort(key=lambda r: (r.importance or 0), reverse=True)
                return [self._to_memory_item(row) for row in matched[:limit]]

            # 最终降级: 最重要的记忆
            rows_by_imp = sorted(rows, key=lambda r: (r.importance or 0), reverse=True)
            return [self._to_memory_item(row) for row in rows_by_imp[:limit]]
        finally:
            db.close()

    async def get_recent(self, user_id: str, limit: int = 20) -> List[MemoryItem]:
        db = SessionLocal()
        try:
            rows = (
                db.query(LongTermMemoryRow)
                .filter(LongTermMemoryRow.user_id == user_id)
                .order_by(LongTermMemoryRow.timestamp.desc())
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
                db.query(LongTermMemoryRow)
                .filter(
                    LongTermMemoryRow.memory_id == memory_id,
                    LongTermMemoryRow.user_id == user_id,
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
                db.query(LongTermMemoryRow)
                .filter(LongTermMemoryRow.user_id == user_id)
                .count()
            )
            db.query(LongTermMemoryRow).filter(
                LongTermMemoryRow.user_id == user_id
            ).delete()
            db.commit()
            return count
        finally:
            db.close()

    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            count = (
                db.query(LongTermMemoryRow)
                .filter(LongTermMemoryRow.user_id == user_id)
                .count()
            )
            return {
                "provider": "long_term",
                "layer": "postgresql_vector",
                "count": count,
                "retention": "permanent",
            }
        finally:
            db.close()

    def _to_memory_item(self, row) -> MemoryItem:
        from memory.providers.base import MemoryItem as MI
        return MI(
            id=row.memory_id,
            user_id=row.user_id,
            type=row.type or "",
            content=row.content or "",
            metadata=row.metadata_json or {},
            importance=row.importance or 0.5,
            timestamp=row.timestamp.isoformat() if row.timestamp else None,
            source=row.source or "",
        )

    @staticmethod
    def _parse_timestamp(ts_str: str) -> datetime:
        """Parse ISO timestamp preserving timezone, then convert to naive UTC for DB storage."""
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        parsed = datetime.fromisoformat(ts_str)
        if parsed.tzinfo:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def _evict_if_needed(self, user_id: str, db):
        """Evict lowest-value memories if user exceeds capacity.

        Uses a DB subquery to avoid loading all rows into Python.
        """
        count = db.query(func.count()).filter(
            LongTermMemoryRow.user_id == user_id
        ).scalar()

        if count <= MAX_MEMORIES_PER_USER:
            return

        # Eviction score computed in SQL: importance * 0.7 + recency * 0.3
        days_since = func.extract(
            'epoch',
            func.now() - func.coalesce(LongTermMemoryRow.last_accessed, LongTermMemoryRow.timestamp),
        ) / 86400
        recency = 1.0 / (1.0 + days_since / 30.0)
        eviction_score = LongTermMemoryRow.importance * 0.7 + recency * 0.3

        # Subquery: memory_ids to KEEP (top N by eviction score)
        keep_subq = (
            db.query(LongTermMemoryRow.memory_id)
            .filter(LongTermMemoryRow.user_id == user_id)
            .order_by(eviction_score.desc())
            .limit(MAX_MEMORIES_PER_USER)
            .subquery()
        )

        # Delete rows NOT in the keep set
        db.query(LongTermMemoryRow).filter(
            LongTermMemoryRow.user_id == user_id,
            ~LongTermMemoryRow.memory_id.in_(db.query(keep_subq.c.memory_id)),
        ).delete(synchronize_session='fetch')

    def _compute_recall_score(self, memory, query_embedding, current_time):
        """Compute recall score with time decay."""
        # Semantic similarity (cosine)
        emb = memory.embedding_json
        if emb and isinstance(emb, list) and len(emb) > 0:
            semantic_score = _cosine_similarity(memory.embedding_json, query_embedding)
        else:
            semantic_score = 0.0

        # Time decay: half-life of 90 days
        days_passed = (current_time - (memory.timestamp or current_time)).days
        time_decay = math.exp(-0.008 * days_passed)  # ln(2)/90 ≈ 0.0077

        # Importance boost
        importance = memory.importance or 0.5

        # Combined score: 60% semantic * decay + 40% importance
        final_score = semantic_score * time_decay * 0.6 + importance * 0.4

        return final_score

    def _touch_memories(self, memory_ids: list, db):
        """Update last_accessed and access_count for recalled memories."""
        now = datetime.now(timezone.utc)
        db.query(LongTermMemoryRow).filter(
            LongTermMemoryRow.memory_id.in_(memory_ids)
        ).update({
            LongTermMemoryRow.last_accessed: now,
            LongTermMemoryRow.access_count: LongTermMemoryRow.access_count + 1,
        }, synchronize_session=False)
        db.commit()


class LongTermMemoryRow(_LongBase):
    __tablename__ = "long_term_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(String(256), nullable=False, index=True)
    type = Column(String(64), default="")
    content = Column(Text, default="")
    metadata_json = Column(JSON, default=dict)
    importance = Column(Float, default=0.5)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    source = Column(String(64), default="")
    embedding_json = Column(JSON, nullable=True)  # v3: BGE-M3 嵌入向量
    last_accessed = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    access_count = Column(Integer, default=0)

    __table_args__ = (
        Index("idx_ltm_user_imp", "user_id", "importance"),
    )
