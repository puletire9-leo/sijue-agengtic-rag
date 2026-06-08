from datetime import datetime, timezone
from sqlalchemy import func
from cache import cache
from database import SessionLocal
from models import User, ChatSession, ChatMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class ConversationStorage:
    """对话存储（PostgreSQL + Redis）。"""

    def __init__(self, save_tracker, pagination):
        self.save_tracker = save_tracker
        self.pagination = pagination

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            reasoning_content = msg_data.get("reasoning_content")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                msg = AIMessage(content=content)
                if reasoning_content:
                    msg.additional_kwargs["reasoning_content"] = reasoning_content
                messages.append(msg)
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        """保存对话"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()
            else:
                if metadata is not None:
                    session.metadata_json = metadata

            # 增量保存：只写新消息，不做全量 DELETE + INSERT
            # get_unsaved_messages 不再修改 _saved_hashes，commit 成功后才通过 mark_saved 标记
            unsaved_indices, new_messages = self.save_tracker.get_unsaved_messages(session_id, messages)

            if not new_messages:
                session.updated_at = datetime.now(timezone.utc)
                db.commit()
                return

            now = datetime.now(timezone.utc)
            # 构建 Redis 缓存（全量消息）
            serialized = []
            for idx, msg in enumerate(messages):
                rag_trace = None
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")
                rc = None
                if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                    rc = msg.additional_kwargs.get('reasoning_content')
                serialized.append({
                    "type": msg.type,
                    "content": str(msg.content),
                    "timestamp": now.isoformat(),
                    "rag_trace": rag_trace,
                    "reasoning_content": rc,
                })

            # DB 只插入新增消息
            last_msg_rag_trace = serialized[-1].get("rag_trace") if serialized else None
            for idx, msg in enumerate(new_messages):
                is_last = (idx == len(new_messages) - 1)
                rc = None
                if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                    rc = msg.additional_kwargs.get('reasoning_content')
                db.add(ChatMessage(
                    session_ref_id=session.id,
                    message_type=msg.type,
                    content=str(msg.content),
                    reasoning_content=rc,
                    timestamp=datetime.now(timezone.utc),
                    rag_trace=last_msg_rag_trace if is_last else None,
                ))

            session.updated_at = now

            db.commit()
            # mark_saved 在 commit 之后，避免 commit 失败时追踪器误认为已保存
            self.save_tracker.mark_saved(session_id, new_messages, indices=unsaved_indices)
            # 写缓存在 commit 之后，保证 DB 先落盘再更新缓存
            cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    def load(self, user_id: str, session_id: str, use_pagination: bool = True) -> list:
        """加载对话。

        Args:
            use_pagination: True 时分页加载最近消息（默认50条），False 时全量加载。
        """
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            messages = self._to_langchain_messages(cached)
            # Bug 1 fix: populate tracker from cached messages to avoid re-insertion after restart
            self.save_tracker.mark_saved(session_id, messages)
            return messages

        if use_pagination:
            # Create a local pagination instance to avoid race conditions
            # with concurrent requests mutating the shared self.pagination
            from core.pagination import MessagePagination
            local_pagination = MessagePagination(default_window=50)

            def _load_range(session_id: str, offset: int, limit: int) -> list:
                db2 = SessionLocal()
                try:
                    user = db2.query(User).filter(User.username == user_id).first()
                    if not user:
                        return []
                    session = (
                        db2.query(ChatSession)
                        .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                        .first()
                    )
                    if not session:
                        return []
                    rows = (
                        db2.query(ChatMessage)
                        .filter(ChatMessage.session_ref_id == session.id)
                        .order_by(ChatMessage.id.asc())
                        .offset(offset)
                        .limit(limit)
                        .all()
                    )
                    return [
                        {"type": row.message_type, "content": row.content,
                         "reasoning_content": row.reasoning_content,
                         "timestamp": row.timestamp.isoformat() if row.timestamp else "",
                         "rag_trace": row.rag_trace}
                        for row in rows
                    ]
                finally:
                    db2.close()

            def _count_range(session_id: str) -> int:
                db2 = SessionLocal()
                try:
                    user = db2.query(User).filter(User.username == user_id).first()
                    if not user:
                        return 0
                    session = (
                        db2.query(ChatSession)
                        .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                        .first()
                    )
                    if not session:
                        return 0
                    return (
                        db2.query(ChatMessage)
                        .filter(ChatMessage.session_ref_id == session.id)
                        .count()
                    )
                finally:
                    db2.close()

            local_pagination.set_load_fn(_load_range)
            local_pagination.set_count_fn(_count_range)
            window = local_pagination.load_recent(session_id)
            records = window.messages if hasattr(window, 'messages') else []
            if records:
                cache.set_json(self._messages_cache_key(user_id, session_id), records)
            messages = self._to_langchain_messages(records)
            # Bug 1 fix: populate tracker from DB-loaded messages to avoid re-insertion after restart
            self.save_tracker.mark_saved(session_id, messages)
            return messages

        # 全量加载（兼容模式）
        records = self.get_session_messages(user_id, session_id)
        cache.set_json(self._messages_cache_key(user_id, session_id), records)
        messages = self._to_langchain_messages(records)
        # Bug 1 fix: populate tracker from DB-loaded messages to avoid re-insertion after restart
        self.save_tracker.mark_saved(session_id, messages)
        return messages

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []

            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            session_ids = [s.id for s in sessions]

            # Single query for all message counts, filtered by current user's sessions
            msg_counts = (
                db.query(ChatMessage.session_ref_id, func.count(ChatMessage.id))
                .filter(ChatMessage.session_ref_id.in_(session_ids))
                .group_by(ChatMessage.session_ref_id)
                .all()
            ) if session_ids else []
            count_map = {sid: cnt for sid, cnt in msg_counts}

            # Single query for first user message per session (title)
            title_map: dict = {}
            if session_ids:
                # Single query: first human message per session (for title)
                first_msgs = (
                    db.query(
                        ChatMessage.session_ref_id,
                        ChatMessage.content,
                    )
                    .filter(
                        ChatMessage.session_ref_id.in_(session_ids),
                        ChatMessage.message_type == "human",
                    )
                    .order_by(ChatMessage.session_ref_id, ChatMessage.id.asc())
                    .all()
                )
                seen: set = set()
                for ref_id, content in first_msgs:
                    if ref_id not in seen:
                        seen.add(ref_id)
                        title_map[ref_id] = content[:30] + ("..." if len(content) > 30 else "")

            result = []
            for s in sessions:
                result.append(
                    {
                        "session_id": s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count_map.get(s.id, 0),
                        "title": title_map.get(s.id),
                    }
                )
            cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        finally:
            db.close()

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "reasoning_content": row.reasoning_content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                }
                for row in rows
            ]
            cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def get_session_messages_paginated(self, user_id: str, session_id: str, offset: int = 0, limit: int = 50) -> dict:
        """获取分页消息，返回 {"messages": [...], "total": int}。"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return {"messages": [], "total": 0}
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return {"messages": [], "total": 0}

            total = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .count()
            )
            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            messages = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "reasoning_content": row.reasoning_content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                }
                for row in rows
            ]
            return {"messages": messages, "total": total}
        finally:
            db.close()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话，返回是否删除成功"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
            self.save_tracker.reset_session(session_id)
            return True
        finally:
            db.close()
