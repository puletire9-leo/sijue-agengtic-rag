"""SessionMemoryProvider — Redis 会话记忆后端。

存储在 Redis 中，默认 TTL 24h，适合当次会话上下文。
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis

from memory.providers.base import MemoryItem, MemoryProvider


class SessionMemoryProvider(MemoryProvider):
    """Redis 会话记忆。

    用法:
        provider = SessionMemoryProvider(redis.Redis.from_url("redis://..."))
        await provider.save("user1", MemoryItem(type="fact", content="用户喜欢简洁回答"))
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None, ttl: int = 86400):
        if redis_client:
            self.redis = redis_client
        else:
            self.redis = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
            )
        self.ttl = ttl

    async def save(self, user_id: str, item: MemoryItem) -> str:
        if not item.id:
            item.id = str(uuid.uuid4())[:12]
        if not item.timestamp:
            item.timestamp = datetime.now(timezone.utc).isoformat()

        key = f"mem:session:{user_id}:{item.id}"
        data = {
            "id": item.id,
            "user_id": user_id,
            "type": item.type,
            "content": item.content,
            "metadata": item.metadata,
            "importance": item.importance,
            "timestamp": item.timestamp,
            "source": item.source,
        }

        def _do_save():
            self.redis.setex(key, self.ttl, json.dumps(data, ensure_ascii=False))
            idx_key = f"mem:session:{user_id}:idx"
            self.redis.sadd(idx_key, item.id)
            self.redis.expire(idx_key, self.ttl)

        await asyncio.to_thread(_do_save)
        return item.id

    async def save_batch(self, user_id: str, items: List[MemoryItem]) -> List[str]:
        ids = []
        for item in items:
            if not item.id:
                item.id = str(uuid.uuid4())[:12]
            if not item.timestamp:
                item.timestamp = datetime.now(timezone.utc).isoformat()
            ids.append(item.id)

        def _do_save_batch():
            pipe = self.redis.pipeline()
            for item in items:
                key = f"mem:session:{user_id}:{item.id}"
                data = {
                    "id": item.id, "user_id": user_id, "type": item.type,
                    "content": item.content, "metadata": item.metadata,
                    "importance": item.importance,
                    "timestamp": item.timestamp, "source": item.source,
                }
                pipe.setex(key, self.ttl, json.dumps(data, ensure_ascii=False))
                pipe.sadd(f"mem:session:{user_id}:idx", item.id)
            pipe.expire(f"mem:session:{user_id}:idx", self.ttl)
            pipe.execute()

        await asyncio.to_thread(_do_save_batch)
        return ids

    async def recall(self, user_id: str, query: str, limit: int = 10) -> List[MemoryItem]:
        """基于字符 bigram 重叠率的召回（兼容中文）。

        优化: 先按重要性排序取 top 50，再对候选集做 bigram 过滤，避免全量扫描。
        """
        def _do_recall():
            idx_key = f"mem:session:{user_id}:idx"
            ids = list(self.redis.smembers(idx_key) or set())

            # Batch-fetch with mget to avoid N+1 queries
            if not ids:
                return []
            keys = [f"mem:session:{user_id}:{mid}" for mid in ids]
            raw_values = self.redis.mget(keys)

            candidates = []
            for raw in raw_values:
                if raw:
                    data = json.loads(raw)
                    candidates.append((float(data.get("importance", 0.5)), data))
            candidates.sort(key=lambda x: x[0], reverse=True)

            MAX_MEMORIES_TO_SCAN = 50
            items = []
            query_lower = query.lower()
            q_bigrams = {query_lower[i:i+2] for i in range(len(query_lower)-1)}
            for _, data in candidates[:MAX_MEMORIES_TO_SCAN]:
                content = data.get("content", "").lower()
                meta_str = json.dumps(data.get("metadata", {})).lower()
                c_bigrams = {content[i:i+2] for i in range(len(content)-1)}
                overlap = len(q_bigrams & c_bigrams) / len(q_bigrams) if q_bigrams else 0.0
                if overlap > 0.1:
                    items.append(self._to_memory_item(data))
                else:
                    m_bigrams = {meta_str[i:i+2] for i in range(len(meta_str)-1)}
                    meta_overlap = len(q_bigrams & m_bigrams) / len(q_bigrams) if q_bigrams else 0.0
                    if meta_overlap > 0.1:
                        items.append(self._to_memory_item(data))
            items.sort(key=lambda x: x.importance, reverse=True)
            return items[:limit]

        return await asyncio.to_thread(_do_recall)

    async def get_recent(self, user_id: str, limit: int = 20) -> List[MemoryItem]:
        def _do_get_recent():
            idx_key = f"mem:session:{user_id}:idx"
            ids = list(self.redis.smembers(idx_key) or [])
            if not ids:
                return []
            keys = [f"mem:session:{user_id}:{mid}" for mid in ids]
            raw_values = self.redis.mget(keys)
            items = []
            for raw in raw_values:
                if raw:
                    items.append(self._to_memory_item(json.loads(raw)))

            def _sort_key(item: MemoryItem):
                ts = item.timestamp
                if not ts:
                    return datetime.min.replace(tzinfo=timezone.utc)
                try:
                    if ts.endswith("Z"):
                        ts = ts[:-1] + "+00:00"
                    return datetime.fromisoformat(ts)
                except (ValueError, AttributeError):
                    return datetime.min.replace(tzinfo=timezone.utc)

            items.sort(key=_sort_key, reverse=True)
            return items[:limit]

        return await asyncio.to_thread(_do_get_recent)

    async def forget(self, user_id: str, memory_id: str) -> bool:
        def _do_forget():
            key = f"mem:session:{user_id}:{memory_id}"
            existed = self.redis.exists(key)
            self.redis.delete(key)
            self.redis.srem(f"mem:session:{user_id}:idx", memory_id)
            return existed > 0

        return await asyncio.to_thread(_do_forget)

    async def clear_user(self, user_id: str) -> int:
        def _do_clear_user():
            idx_key = f"mem:session:{user_id}:idx"
            ids = self.redis.smembers(idx_key) or set()
            if not ids:
                return 0
            keys = [f"mem:session:{user_id}:{mid}" for mid in ids]
            pipe = self.redis.pipeline()
            for k in keys:
                pipe.delete(k)
            pipe.delete(idx_key)
            pipe.execute()
            return len(ids)

        return await asyncio.to_thread(_do_clear_user)

    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        def _do_get_stats():
            idx_key = f"mem:session:{user_id}:idx"
            count = self.redis.scard(idx_key) or 0
            return {"provider": "session", "layer": "redis", "count": count, "ttl": self.ttl}

        return await asyncio.to_thread(_do_get_stats)

    def _to_memory_item(self, data: dict) -> MemoryItem:
        return MemoryItem(
            id=data.get("id"),
            user_id=data.get("user_id", ""),
            type=data.get("type", ""),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 0.5),
            timestamp=data.get("timestamp"),
            source=data.get("source"),
        )
