import logging

from fastapi import APIRouter, Depends, HTTPException

from agent import storage
from auth import get_current_user

logger = logging.getLogger(__name__)
from models import User
from schemas import (
    MessageInfo,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)

router = APIRouter()


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,
    offset: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
):
    """获取指定会话的消息（分页）"""
    try:
        limit = min(max(limit, 1), 200)
        offset = max(offset, 0)
        result = storage.get_session_messages_paginated(
            current_user.username, session_id, offset=offset, limit=limit
        )
        messages = [
            MessageInfo(
                type=msg["type"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                rag_trace=msg.get("rag_trace"),
            )
            for msg in result["messages"]
        ]
        return SessionMessagesResponse(messages=messages, total=result["total"])
    except Exception as e:
        logger.exception("Failed to get session messages")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    """获取当前用户的所有会话列表"""
    try:
        sessions = [SessionInfo(**item) for item in storage.list_session_infos(current_user.username)]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        logger.exception("Failed to list sessions")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    """删除当前用户的指定会话"""
    try:
        deleted = storage.delete_session(current_user.username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete session")
        raise HTTPException(status_code=500, detail="Internal server error")
