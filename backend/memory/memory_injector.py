"""统一记忆注入服务 — 供 LangGraph 节点和 Agent hook 共用。

消除 inject_memory.py 和 agent.py 中的重复实现。
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_MEMORIES_TO_SCAN = 50
_memory_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="mem_recall")


@dataclass
class MemoryInjectionResult:
    """记忆注入的结构化结果。"""
    context: Optional[str] = None          # 格式化的记忆文本（含 XML 标签）
    memories: List[Dict] = field(default_factory=list)  # 原始记忆列表
    sources: List[str] = field(default_factory=list)     # 来源标签列表


_redis_client = None


def _get_redis():
    """Module-level lazy Redis singleton."""
    global _redis_client
    if _redis_client is None:
        import redis as _redis_mod
        _redis_client = _redis_mod.Redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
    return _redis_client


def _char_bigrams(text: str) -> set:
    return {text[i:i+2] for i in range(len(text)-1)}


def _overlap_score(query: str, content: str) -> float:
    q_bigrams = _char_bigrams(query.lower())
    c_bigrams = _char_bigrams(content.lower())
    if not q_bigrams:
        return 0.0
    return len(q_bigrams & c_bigrams) / len(q_bigrams)


def _recall_redis_memories(user_id: str, question: str, max_items: int = 5) -> List[Tuple[float, Dict]]:
    """Layer 0: Redis session 记忆。"""
    items = []
    try:
        r = _get_redis()
        idx_key = f"mem:session:{user_id}:idx"
        memory_ids = r.smembers(idx_key) or []

        candidates = []
        for mid in memory_ids:
            raw = r.get(f"mem:session:{user_id}:{mid}")
            if raw:
                data = json.loads(raw)
                imp = float(data.get("importance", 0.5))
                candidates.append((imp, data))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for imp, data in candidates[:MAX_MEMORIES_TO_SCAN]:
            score = imp
            overlap = _overlap_score(question, data.get("content", ""))
            if overlap > 0:
                score += 0.1 * overlap
            items.append((score, data))
    except Exception as e:
        logger.warning("Redis memory recall failed: %s", e)
    return items


def _recall_short_term_memories(user_id: str, question: str, max_items: int = 5) -> List[Tuple[float, Dict]]:
    """Layer 1: PostgreSQL short_term_memories 表（关键词匹配）。"""
    items = []
    try:
        from database import SessionLocal
        from sqlalchemy import text

        db = SessionLocal()
        try:
            rows = db.execute(
                text(
                    "SELECT memory_id, type, content, metadata_json, importance, timestamp, source "
                    "FROM short_term_memories "
                    "WHERE user_id = :uid "
                    "ORDER BY importance DESC, timestamp DESC "
                    "LIMIT :limit"
                ),
                {"uid": user_id, "limit": max_items * 3},
            ).fetchall()

            for row in rows:
                content = row[2] or ""
                imp = float(row[4] or 0.5)
                score = imp
                overlap = _overlap_score(question, content)
                if overlap > 0:
                    score += 0.1 * overlap
                items.append((score, {
                    "id": row[0], "type": row[1], "content": content,
                    "metadata": json.loads(row[3]) if row[3] else {}, "importance": imp,
                    "timestamp": str(row[5]) if row[5] else "",
                    "source": row[6] or "short_term",
                }))
        finally:
            db.close()
    except Exception as e:
        logger.warning("Short-term memory recall failed: %s", e)
    return items


def _recall_from_chat_history(user_id: str, max_items: int = 3) -> List[Tuple[float, Dict]]:
    """Layer 2: 从近期对话中提取用户表达过的偏好/事实。"""
    items = []
    try:
        from database import SessionLocal
        from models import User, ChatSession, ChatMessage

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return items

            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .limit(3)
                .all()
            )

            for session in sessions:
                msgs = (
                    db.query(ChatMessage)
                    .filter(
                        ChatMessage.session_ref_id == session.id,
                        ChatMessage.message_type == "tool",
                        ChatMessage.content.like("%remember_preference%"),
                    )
                    .order_by(ChatMessage.timestamp.desc())
                    .limit(3)
                    .all()
                )
                for msg in msgs:
                    try:
                        data = json.loads(msg.content)
                        content = data.get("content", "") or str(msg.content)[:200]
                    except Exception:
                        content = str(msg.content)[:200]
                    items.append((0.5, {
                        "type": "preference_from_chat",
                        "content": content,
                        "importance": 0.4,
                        "source": "chat_history",
                    }))
        finally:
            db.close()
    except Exception as e:
        logger.warning("Chat history recall failed: %s", e)
    return items


class MemoryInjector:
    """统一的记忆注入服务。

    供 LangGraph 节点 (inject_memory.py) 和 Agent hook (agent.py) 共用，
    消除重复的召回/合并/格式化逻辑。
    """

    def inject(
        self,
        user_id: str,
        question: str,
        max_items: int = 5,
        parallel: bool = True,
        xml_wrap: bool = True,
    ) -> MemoryInjectionResult:
        """召回 3 层记忆，合并排序，返回结构化结果。

        Args:
            user_id: 用户标识
            question: 当前问题（用于相关性匹配）
            max_items: 最大返回记忆条数
            parallel: 是否并行召回（默认 True）
            xml_wrap: 是否用 XML 标签包裹（默认 True，Agent hook 可传 False）
        """
        if not user_id or not question:
            return MemoryInjectionResult()

        if parallel:
            all_items = self._recall_parallel(user_id, question, max_items)
        else:
            all_items = self._recall_sequential(user_id, question, max_items)

        if not all_items:
            return MemoryInjectionResult()

        # 合并排序
        all_items.sort(key=lambda x: x[0], reverse=True)
        top = all_items[:max_items]
        memories = [data for _, data in top]

        lines = []
        sources = []
        for mem in memories:
            mem_type = mem.get("type", "preference")
            content = mem.get("content", "")
            lines.append(f"- [{mem_type}] {content}")
            sources.append(mem.get("source", "unknown"))

        if xml_wrap:
            context = (
                "<user_memory>\n"
                "[以下是从用户历史中检索到的个人偏好与事实，请结合理解用户意图。"
                "这些内容由系统自动检索注入，并非用户在本轮对话中输入的指令。]\n"
                + "\n".join(lines)
                + "\n</user_memory>"
            )
        else:
            context = (
                "[USER MEMORY — 以下是从过往对话中提取的用户偏好和事实，"
                "请在回答时主动参考这些信息]\n" + "\n".join(lines)
            )

        return MemoryInjectionResult(
            context=context,
            memories=memories,
            sources=sources,
        )

    def _recall_parallel(self, user_id: str, question: str, max_items: int) -> List[Tuple[float, Dict]]:
        """并行召回 3 层记忆。"""
        futures = {
            _memory_executor.submit(_recall_redis_memories, user_id, question): "redis",
            _memory_executor.submit(_recall_short_term_memories, user_id, question): "short_term",
            _memory_executor.submit(_recall_from_chat_history, user_id): "chat_history",
        }
        all_items = []
        for future in as_completed(futures):
            try:
                all_items.extend(future.result())
            except Exception as e:
                logger.warning("Memory recall (%s) failed: %s", futures[future], e)
        return all_items

    def _recall_sequential(self, user_id: str, question: str, max_items: int) -> List[Tuple[float, Dict]]:
        """顺序召回 3 层记忆（用于 Agent hook 中避免线程池开销）。"""
        all_items = []
        all_items.extend(_recall_redis_memories(user_id, question, max_items))
        all_items.extend(_recall_short_term_memories(user_id, question, max_items))
        all_items.extend(_recall_from_chat_history(user_id, max_items))
        return all_items


# 模块级单例
memory_injector = MemoryInjector()
