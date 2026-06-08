import asyncio
import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from agent import chat_with_agent, chat_with_agent_stream
from auth import get_current_user
from metrics import ACTIVE_STREAMS
from models import User
from schemas import ChatRequest, ChatResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        session_id = request.session_id or "default_session"
        resp = await asyncio.to_thread(chat_with_agent, request.message, current_user.username, session_id)
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        logger.exception("Chat endpoint error")
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。",
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail="认证失败")
            raise HTTPException(status_code=code, detail="上游服务错误")
        raise HTTPException(status_code=500, detail="服务器内部错误")


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """跟 Agent 对话 (流式)"""

    async def event_generator():
        ACTIVE_STREAMS.inc()
        try:
            session_id = request.session_id or "default_session"
            async for chunk in chat_with_agent_stream(request.message, current_user.username, session_id):
                yield chunk
        except Exception as e:
            logger.exception("Chat stream error")
            error_data = {"type": "error", "content": "服务器内部错误"}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            ACTIVE_STREAMS.dec()
            try:
                yield "data: [DONE]\n\n"
            except GeneratorExit:
                raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
