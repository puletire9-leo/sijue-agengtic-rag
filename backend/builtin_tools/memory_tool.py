"""记忆工具 — Agent 可以主动保存和召回用户偏好与关键事实。

对应 hermes-agent 的 memory_tool.py。
"""

import os
import threading

import redis as _redis

from langchain_core.tools import tool

# Module-level lazy Redis client — avoids creating a new connection per tool call.
_mem_redis_client: _redis.Redis | None = None
_mem_redis_lock = threading.Lock()


def _get_mem_redis() -> _redis.Redis:
    global _mem_redis_client
    if _mem_redis_client is None:
        with _mem_redis_lock:
            if _mem_redis_client is None:
                _mem_redis_client = _redis.Redis.from_url(
                    os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
                    decode_responses=True,
                )
    return _mem_redis_client


@tool("remember_preference")
def remember_preference(content: str, importance: float = 0.7) -> str:
    """Save information about the user to persistent memory.

    Use this when the user explicitly asks you to remember something,
    or when you detect important user preferences/facts worth saving
    (e.g. "I prefer short answers", "My name is Zhang San", "I work at Company X").

    Args:
        content: What to remember — a clear, concise statement
        importance: How important this is (0-1). Use 0.9+ for critical facts,
                   0.5-0.8 for preferences, <0.5 for transient info.

    Returns:
        Confirmation message
    """
    try:
        from tools import get_current_user
        user_id = get_current_user()

        import uuid
        import json
        from datetime import datetime, timezone

        item_id = str(uuid.uuid4())[:12]
        timestamp = datetime.now(timezone.utc).isoformat()

        r = _get_mem_redis()
        key = f"mem:session:{user_id}:{item_id}"
        data = {
            "id": item_id,
            "user_id": user_id,
            "type": "preference" if importance < 0.8 else "fact",
            "content": content,
            "metadata": {},
            "importance": importance,
            "timestamp": timestamp,
            "source": "agent_tool",
        }
        ttl = 86400  # 24h
        r.setex(key, ttl, json.dumps(data, ensure_ascii=False))
        idx_key = f"mem:session:{user_id}:idx"
        r.sadd(idx_key, item_id)
        r.expire(idx_key, ttl)

        return f"已记住: {content} (重要性: {importance:.0%})"
    except Exception as e:
        return f"记忆保存失败: {e}"


@tool("recall_memory")
def recall_memory(query: str) -> str:
    """Search the user's saved memories for relevant information.

    Use this when the user asks "what do you remember about me"
    or references something that might be in their saved preferences.

    Args:
        query: What to search for in memories

    Returns:
        Matching memories
    """
    try:
        from tools import get_current_user
        user_id = get_current_user()

        import json

        r = _get_mem_redis()
        idx_key = f"mem:session:{user_id}:idx"
        ids = r.smembers(idx_key) or set()

        query_lower = query.lower()
        matching = []
        id_list = list(ids)[:200]
        if id_list:
            pipe = r.pipeline()
            for mid in id_list:
                pipe.get(f"mem:session:{user_id}:{mid}")
            raw_values = pipe.execute()
            for raw in raw_values:
                if raw:
                    data = json.loads(raw)
                    content = data.get("content", "").lower()
                    if query_lower in content:
                        matching.append(data)
                    elif query_lower in json.dumps(data.get("metadata", {})).lower():
                        matching.append(data)

        if not matching:
            return "未找到相关记忆。"

        matching.sort(key=lambda x: x.get("importance", 0.5), reverse=True)
        matching = matching[:5]

        lines = ["找到以下相关记忆:"]
        for item in matching:
            lines.append(f"- [{item.get('type', '')}] {item.get('content', '')} (重要性: {float(item.get('importance', 0.5)):.0%})")
        return "\n".join(lines)
    except Exception as e:
        return f"记忆召回失败: {e}"
