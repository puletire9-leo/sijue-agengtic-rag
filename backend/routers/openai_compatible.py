"""OpenAI 兼容路由 — 让 Open WebUI 对接 SuperMew 后端。

实现 /v1/models 和 /v1/chat/completions 端点，
将 Open WebUI 的请求转换为 SuperMew Agent 管线调用。
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from pydantic import BaseModel

from agent import (
    agent,
    compressor,
    loop_detector,
    storage,
    _inject_memory_context,
)
from agentic_rag.config import agent as agent_cfg
from agentic_rag.token_estimator import estimate_message_tokens as _estimate_msg_tokens, estimate_text_tokens as _estimate_text_tokens
from events import set_rag_step_queue
from metrics import ACTIVE_STREAMS, LLM_CALLS
from openai_adapter import (
    extract_last_user_message,
    format_citations_as_sources,
    format_citations_as_text,
    make_chunk_id,
    openai_messages_to_langchain,
    to_openai_chunk,
    to_openai_done,
    to_openai_error,
    to_openai_error_sse,
    to_openai_finish,
    to_openai_reasoning_chunk,
    to_openai_response,
    to_openai_sources_event,
)
from core.rate_limiter import LocalRateLimiter
from tools import get_last_rag_context, reset_tool_call_guards, set_current_user, set_current_kb_ids

log = logging.getLogger(__name__)

router = APIRouter(tags=["openai_compatible"])

# Per-user rate limiter for chat completions (30 requests per 60 seconds)
_chat_rate_limiter = LocalRateLimiter(max_requests=30, window_seconds=60)


class _RagStepProxy:
    """代理：捕获 rag_context 事件和 RAG 步骤，用于 OpenAI 格式响应。

    - rag_context: 最终的 RAG 上下文（包含 citations）
    - output_queue: 实时推送 RAG 步骤到 SSE 输出队列
    - _loop: 事件循环引用（供 emit_rag_step 跨线程使用）
    """
    def __init__(self, output_queue: asyncio.Queue = None):
        self.rag_context = None
        self.output_queue = output_queue
        self._loop = asyncio.get_running_loop()

    def put_nowait(self, step):
        if isinstance(step, dict):
            if step.get("type") == "rag_context":
                self.rag_context = step.get("context")
            elif step.get("type") == "rag_step":
                # 实时推送 RAG 步骤到输出队列
                if self.output_queue is not None:
                    try:
                        self.output_queue.put_nowait({"type": "rag_step", "step": step.get("step", {})})
                    except asyncio.QueueFull:
                        pass

# ── API Key 配置（必须通过环境变量设置）──
# NOTE: No RuntimeError at import time — the check is deferred to verify_api_key()
# so that missing env var does not crash the entire process at import.
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY")


# ── 请求模型 ──


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None

    class Config:
        # Limit individual message content size
        str_max_length = 50000


class ChatCompletionRequest(BaseModel):
    model: str = "supermew-rag"
    messages: list[ChatMessage]  # Limited to 100 messages via validator
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    tools: Optional[list] = None
    tool_choice: Optional[str | dict] = None
    starlink_kb_ids: Optional[list[str]] = None


# ── 认证 ──


async def verify_api_key(request: Request) -> str:
    """从 Authorization 头提取并验证 API Key。返回 user_id。"""
    if not OPENAI_COMPATIBLE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Service is not configured",
        )

    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header format")
    token = auth[7:].strip()

    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")

    # 常量时间比较，防止时序攻击
    api_key_match = hmac.compare_digest(token, OPENAI_COMPATIBLE_API_KEY)

    if not api_key_match:
        # 尝试作为 SuperMew JWT token 验证
        try:
            from jose import jwt
            from auth import ALGORITHM, SECRET_KEY

            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload.get("sub", "openwebui_user")
        except Exception:
            # 统一错误消息，不暴露认证策略
            raise HTTPException(status_code=401, detail="Invalid credentials")

    # 尝试从 Open WebUI 用户信息头提取 user_id
    user_name = request.headers.get("x-openwebui-user-name", "")
    return user_name or "openwebui_user"


async def get_user_info(request: Request) -> tuple[str, str]:
    """返回 (username, role)。role 从 JWT 或 header 提取。"""
    if not OPENAI_COMPATIBLE_API_KEY:
        raise HTTPException(status_code=503, detail="Service is not configured")

    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header format")
    token = auth[7:].strip()

    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")

    api_key_match = hmac.compare_digest(token, OPENAI_COMPATIBLE_API_KEY)

    if not api_key_match:
        try:
            from jose import jwt
            from auth import ALGORITHM, SECRET_KEY
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload.get("sub", "openwebui_user"), payload.get("role", "user")
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    user_name = request.headers.get("x-openwebui-user-name", "") or "openwebui_user"
    user_role = request.headers.get("x-openwebui-user-role", "user")
    return user_name, user_role


# ── 端点 ──


@router.get("/v1/models")
async def list_models(request: Request):
    """返回可用模型列表（Open WebUI 启动时调用）。"""
    # 当 API Key 已配置时要求认证
    if OPENAI_COMPATIBLE_API_KEY:
        await verify_api_key(request)
    return {
        "object": "list",
        "data": [
            {
                "id": "supermew-rag",
                "object": "model",
                "created": 1234567890,
                "owned_by": "supermew",
                "name": "SuperMew RAG",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
):
    """OpenAI 兼容的 Chat Completions 端点。

    支持流式（SSE）和非流式响应。
    """
    # 认证
    user_id = await verify_api_key(request)

    # 速率限制
    if not _chat_rate_limiter.is_allowed(f"openai:{user_id}"):
        return JSONResponse(
            status_code=429,
            content=to_openai_error("Rate limit exceeded. Please try again later.", 429),
        )

    # 提取最后一条 user 消息
    messages_dict = [msg.model_dump(exclude_none=True) for msg in body.messages]
    user_text = extract_last_user_message(messages_dict)
    if not user_text:
        return JSONResponse(
            status_code=400,
            content=to_openai_error("At least one user message is required", 400),
        )

    if body.stream:
        return StreamingResponse(
            _stream_response(messages_dict, user_text, user_id, body.model, body.starlink_kb_ids),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _non_stream_response(messages_dict, user_text, user_id, body.model, body.starlink_kb_ids)


# ── 流式响应 ──


async def _stream_response(
    messages_dict: list[dict],
    user_text: str,
    user_id: str,
    model: str,
    starlink_kb_ids: Optional[list[str]] = None,
):
    """流式生成 OpenAI 格式的 SSE 响应。"""
    # P0-1 修复: 在生成器内部设置 ContextVar，避免并发污染
    set_current_user(user_id)
    # 如果前端指定了 starlink_kb_ids，使用指定的；否则由 agent.py 加载用户可访问的
    if starlink_kb_ids:
        set_current_kb_ids(starlink_kb_ids)
    ACTIVE_STREAMS.inc()
    LLM_CALLS.labels(model="openai_compat", purpose="chat").inc()

    chunk_id = make_chunk_id()
    created_ts = int(time.time())  # P2-1: 同一请求共享时间戳
    first_chunk = True
    full_response = ""
    citations_text = ""

    try:
        # 构建 LangChain 消息列表（跳过 storage.load）
        lc_messages = openai_messages_to_langchain(messages_dict)

        # 清理残留 RAG 上下文
        get_last_rag_context(clear=True)
        reset_tool_call_guards()

        # 统一输出队列
        output_queue: asyncio.Queue = asyncio.Queue()

        rag_proxy = _RagStepProxy(output_queue=output_queue)
        set_rag_step_queue(rag_proxy)

        # 上下文压缩
        if sum(_estimate_msg_tokens(m) for m in lc_messages) > agent_cfg.COMPRESSION_THRESHOLD_AGENT:
            result = compressor.compress(lc_messages)
            lc_messages = result.messages

        # 记忆注入
        memory_ctx = await asyncio.to_thread(_inject_memory_context, user_id, user_text)
        if memory_ctx:
            lc_messages = [SystemMessage(content=memory_ctx)] + lc_messages

        # 循环检测
        loop_state = loop_detector.detect(lc_messages)
        if loop_state.is_stuck:
            lc_messages = loop_detector.recover(lc_messages, loop_state)

        # Agent 异步执行
        async def _agent_worker():
            nonlocal full_response
            try:
                async for msg, metadata in agent.astream(
                    {"messages": lc_messages},
                    stream_mode="messages",
                    config={"recursion_limit": int(os.environ.get("RECURSION_LIMIT", "100"))},
                ):
                    if not isinstance(msg, AIMessageChunk):
                        continue

                    # 提取 content（即使消息同时包含 tool_calls 也照常发送）
                    content = ""
                    if isinstance(msg.content, str):
                        content = msg.content
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, str):
                                content += block
                            elif isinstance(block, dict) and block.get("type") == "text":
                                content += block.get("text", "")

                    if content:
                        full_response += content
                        await output_queue.put({"type": "content", "content": content})
            except Exception as e:
                log.exception("Agent worker error")
                await output_queue.put({"type": "error", "content": "Internal server error"})
            finally:
                await output_queue.put(None)

        agent_task = asyncio.create_task(_agent_worker())

        # 主循环：从队列取事件并转换为 OpenAI 格式
        # Bug 8: overall wall-clock deadline in addition to per-event timeout
        stream_deadline = time.monotonic() + 300  # 300 s overall
        try:
            while True:
                remaining = stream_deadline - time.monotonic()
                if remaining <= 0:
                    yield to_openai_error_sse("Request timed out (overall deadline)", model=model, chunk_id=chunk_id)
                    return
                try:
                    event = await asyncio.wait_for(output_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    yield to_openai_error_sse("Request timed out", model=model, chunk_id=chunk_id)
                    return

                if event is None:
                    break

                event_type = event.get("type")
                if event_type == "rag_step":
                    # 实时推送 RAG 步骤作为 reasoning_content
                    step = event.get("step", {})
                    icon = step.get("icon", "")
                    label = step.get("label", "")
                    detail = step.get("detail", "")
                    reasoning_text = f"{icon} {label}"
                    if detail:
                        reasoning_text += f"\n{detail}"
                    yield to_openai_reasoning_chunk(reasoning_text + "\n", model, chunk_id, created=created_ts)
                elif event_type == "content":
                    content = event.get("content", "")
                    if content:
                        yield to_openai_chunk(content, model, chunk_id, created=created_ts, first=first_chunk)
                        first_chunk = False
                elif event_type == "error":
                    yield to_openai_error_sse(event.get("content", "Unknown error"), model=model, chunk_id=chunk_id)
                    return
        except GeneratorExit:
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            raise
        finally:
            set_rag_step_queue(None)
            if not agent_task.done():
                agent_task.cancel()

        # 获取 RAG 引用（优先从 proxy 捕获，ContextVar 在异步环境下不可靠）
        # Bug 3 修复: 让 call_soon_threadsafe 回调有机会执行
        await asyncio.sleep(0)
        rag_context = rag_proxy.rag_context or get_last_rag_context(clear=True)
        rag_trace = rag_context.get("rag_trace") if rag_context else None
        citations = None
        retrieved_docs = None
        if rag_trace:
            citations = rag_trace.get("citations")
            retrieved_docs = rag_trace.get("retrieved_chunks")

        # Fallback: LLM 没调用 search_knowledge_base 时，自动运行 RAG 管线
        print(f"[RAG-FALLBACK] check: rag_trace={bool(rag_trace)}, user_text_len={len(user_text or '')}")
        if not rag_trace and user_text and len(user_text.strip()) > 2:
            print("[RAG-FALLBACK] LLM did not call search_knowledge_base, running RAG fallback")
            try:
                from tools import get_current_kb_ids
                from agentic_rag.runner import run_agentic_rag_sync, format_rag_result
                kb_ids = get_current_kb_ids()
                rag_result = await asyncio.to_thread(
                    run_agentic_rag_sync, user_text, user_id=user_id, kb_ids=kb_ids
                )
                formatted = format_rag_result(rag_result)
                docs = formatted.get("docs", [])
                rag_trace = formatted.get("rag_trace", {})
                if rag_trace:
                    citations = rag_trace.get("citations")
                    retrieved_docs = rag_trace.get("retrieved_chunks")
                    # 用 RAG 结果重新生成回答
                    rag_context_text = ""
                    for i, doc in enumerate(docs[:5], 1):
                        src = doc.get("filename", "Unknown")
                        page = doc.get("page_number", "N/A")
                        text = doc.get("text", "")[:500]
                        rag_context_text += f"[{i}] {src} (Page {page}):\n{text}\n\n"

                    if rag_context_text:
                        # 清空之前的内容，重新生成
                        full_response = ""
                        first_chunk = True
                        regen_messages = lc_messages + [
                            SystemMessage(content=(
                                "以下是知识库检索到的相关内容，请基于这些内容回答用户的问题：\n\n"
                                f"{rag_context_text}"
                            )),
                            HumanMessage(content=user_text),
                        ]
                        async for msg, metadata in agent.astream(
                            {"messages": regen_messages},
                            stream_mode="messages",
                            config={"recursion_limit": 20},
                        ):
                            if not isinstance(msg, AIMessageChunk):
                                continue
                            content = ""
                            if isinstance(msg.content, str):
                                content = msg.content
                            elif isinstance(msg.content, list):
                                for block in msg.content:
                                    if isinstance(block, str):
                                        content += block
                                    elif isinstance(block, dict) and block.get("type") == "text":
                                        content += block.get("text", "")
                            if content:
                                full_response += content
                                await output_queue.put({"type": "content", "content": content})
            except Exception as e:
                log.warning("RAG fallback failed: %s", e, exc_info=True)

        # 发送 sources（Open WebUI 交互式引用卡片）
        if citations:
            sources = format_citations_as_sources(citations, retrieved_docs)
            if sources:
                yield to_openai_sources_event(sources, model, chunk_id, created=created_ts)

        # 持久化对话（流式完成后）
        try:
            session_id = f"openai_compat:{user_id}"
            all_messages = lc_messages + [AIMessage(content=full_response + citations_text)]
            storage.save(user_id, session_id, all_messages)
        except Exception:
            log.warning("Failed to persist OpenAI-compatible conversation", exc_info=True)

        # 估算 token 用量
        prompt_tokens = sum(_estimate_msg_tokens(m) for m in lc_messages)
        completion_tokens = _estimate_text_tokens(full_response + citations_text)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        # 结束
        yield to_openai_finish(model, chunk_id, created=created_ts, usage=usage)
        yield to_openai_done()

    except Exception as e:
        log.exception("Stream response error")
        yield to_openai_error_sse("Internal server error", model=model, chunk_id=chunk_id)
    finally:
        ACTIVE_STREAMS.dec()


# ── 非流式响应 ──


async def _non_stream_response(
    messages_dict: list[dict],
    user_text: str,
    user_id: str,
    model: str,
    starlink_kb_ids: Optional[list[str]] = None,
):
    """非流式生成 OpenAI 格式的响应。"""
    # P0-1 修复: 在函数内部设置 ContextVar
    set_current_user(user_id)
    # 如果前端指定了 starlink_kb_ids，使用指定的；否则由 agent.py 加载用户可访问的
    if starlink_kb_ids:
        set_current_kb_ids(starlink_kb_ids)
    LLM_CALLS.labels(model="openai_compat", purpose="chat").inc()

    try:
        # 构建 LangChain 消息列表
        lc_messages = openai_messages_to_langchain(messages_dict)

        # 清理残留 RAG 上下文
        get_last_rag_context(clear=True)
        reset_tool_call_guards()

        # 设置 RAG 步骤队列（非流式下丢弃）
        set_rag_step_queue(_RagStepProxy())

        # 上下文压缩
        if sum(_estimate_msg_tokens(m) for m in lc_messages) > agent_cfg.COMPRESSION_THRESHOLD_AGENT:
            result = compressor.compress(lc_messages)
            lc_messages = result.messages

        # 记忆注入
        memory_ctx = await asyncio.to_thread(_inject_memory_context, user_id, user_text)
        if memory_ctx:
            lc_messages = [SystemMessage(content=memory_ctx)] + lc_messages

        # 循环检测
        loop_state = loop_detector.detect(lc_messages)
        if loop_state.is_stuck:
            lc_messages = loop_detector.recover(lc_messages, loop_state)

        # Agent 调用（P2-4: 使用 asyncio.to_thread 避免阻塞事件循环）
        # 在线程内捕获 rag_context，因为 ContextVar 跨线程不可见
        _captured = {}

        def _invoke():
            r = agent.invoke(
                {"messages": lc_messages},
                {"recursion_limit": int(os.environ.get("RECURSION_LIMIT", "100"))},
            )
            _captured["rag"] = get_last_rag_context(clear=False)
            return r

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_invoke),
                timeout=300,
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=408,
                content=to_openai_error("Request timed out", 408),
            )

        # 提取响应内容
        response_content = ""
        if isinstance(result, dict):
            if "output" in result:
                response_content = result["output"]
            elif "messages" in result and result["messages"]:
                msg = result["messages"][-1]
                response_content = getattr(msg, "content", str(msg))
            else:
                response_content = str(result)
        elif hasattr(result, "content"):
            response_content = result.content
        else:
            response_content = str(result)

        # 获取引用（从线程内捕获，ContextVar 跨线程不可见）
        citations = None
        retrieved_docs = None
        rag_context = _captured.get("rag")
        if rag_context:
            rag_trace = rag_context.get("rag_trace")
            if rag_trace:
                citations = rag_trace.get("citations")
                retrieved_docs = rag_trace.get("retrieved_chunks")

        # Fallback: LLM 没调用 search_knowledge_base 时，自动运行 RAG 管线
        if not rag_context and user_text and len(user_text.strip()) > 2:
            print("[RAG-FALLBACK] LLM did not call search_knowledge_base (non-stream), running RAG fallback")
            try:
                from tools import get_current_kb_ids
                from agentic_rag.runner import run_agentic_rag_sync, format_rag_result
                kb_ids = get_current_kb_ids()
                rag_result = await asyncio.to_thread(
                    run_agentic_rag_sync, user_text, user_id=user_id, kb_ids=kb_ids
                )
                formatted = format_rag_result(rag_result)
                docs = formatted.get("docs", [])
                rag_trace = formatted.get("rag_trace", {})
                if rag_trace:
                    citations = rag_trace.get("citations")
                    retrieved_docs = rag_trace.get("retrieved_chunks")
                    rag_context_text = ""
                    for i, doc in enumerate(docs[:5], 1):
                        src = doc.get("filename", "Unknown")
                        page = doc.get("page_number", "N/A")
                        text = doc.get("text", "")[:500]
                        rag_context_text += f"[{i}] {src} (Page {page}):\n{text}\n\n"

                    if rag_context_text:
                        regen_messages = lc_messages + [
                            SystemMessage(content=(
                                "以下是知识库检索到的相关内容，请基于这些内容回答用户的问题：\n\n"
                                f"{rag_context_text}"
                            )),
                            HumanMessage(content=user_text),
                        ]

                        def _regen():
                            r = agent.invoke(
                                {"messages": regen_messages},
                                {"recursion_limit": 20},
                            )
                            if isinstance(r, dict):
                                if "output" in r:
                                    return r["output"]
                                elif "messages" in r and r["messages"]:
                                    return getattr(r["messages"][-1], "content", str(r["messages"][-1]))
                            return getattr(r, "content", str(r))

                        response_content = await asyncio.wait_for(
                            asyncio.to_thread(_regen),
                            timeout=120,
                        )
            except Exception as e:
                log.warning("RAG fallback failed (non-stream): %s", e, exc_info=True)

        # 持久化对话
        try:
            session_id = f"openai_compat:{user_id}"
            persist_content = response_content
            if citations:
                persist_content += format_citations_as_text(citations)
            all_messages = lc_messages + [AIMessage(content=persist_content)]
            storage.save(user_id, session_id, all_messages)
        except Exception:
            log.warning("Failed to persist OpenAI-compatible conversation (non-stream)", exc_info=True)

        # 估算 token 用量
        prompt_tokens = sum(_estimate_msg_tokens(m) for m in lc_messages)
        completion_tokens = _estimate_text_tokens(response_content)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        return JSONResponse(
            content=to_openai_response(
                response_content, model,
                citations=citations, usage=usage,
                retrieved_docs=retrieved_docs,
            ),
        )

    except asyncio.TimeoutError:
        log.warning("Non-stream response timed out")
        return JSONResponse(
            status_code=408,
            content=to_openai_error("Request timed out", 408),
        )
    except Exception as e:
        log.exception("Non-stream response error")
        return JSONResponse(
            status_code=500,
            content=to_openai_error("Internal server error", 500),
        )
    finally:
        set_rag_step_queue(None)


# ── 文档上传端点（Open WebUI Tool 调用）──


@router.post("/v1/documents/upload")
async def upload_document_via_api_key(
    request: Request,
    file: UploadFile = File(...),
    folder: str = Form(""),
    kb_id: str = Form(""),
):
    """通过 API Key 认证上传文档到 SuperMew 知识库。

    供 Open WebUI Tool 调用，认证方式与 /v1/chat/completions 一致。
    """
    user_id = await verify_api_key(request)

    # 复用 documents 路由的上传逻辑
    from pathlib import Path
    import os
    from document_ops import UPLOAD_DIR
    from upload_jobs import upload_job_manager

    raw_name = file.filename or ""
    if not raw_name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    filename = Path(raw_name).name

    # 检查文件格式
    _SUPPORTED_EXTS = {
        ".pdf", ".docx", ".doc", ".xlsx", ".xls",
        ".md", ".txt", ".json", ".csv", ".html", ".pptx",
    }
    ext = Path(filename).suffix.lower()
    if ext not in _SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    # 确定目标目录
    folder = (folder or "").strip()
    if ".." in folder or "/" in folder or "\\" in folder:
        raise HTTPException(status_code=400, detail="非法文件夹路径")
    target_dir = (UPLOAD_DIR / folder) if folder else UPLOAD_DIR
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_target_dir = os.path.realpath(str(target_dir))
    if not real_target_dir.startswith(real_upload_dir + os.sep) and real_target_dir != real_upload_dir:
        raise HTTPException(status_code=403, detail="访问被拒绝")
    os.makedirs(target_dir, exist_ok=True)

    # 保存文件
    file_path = target_dir / filename
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 创建后台任务：解析 + 向量化
    job = upload_job_manager.create_job(filename)
    job_id = job["job_id"]

    async def _run_upload():
        import asyncio
        from document_ops import _process_upload_job
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, _process_upload_job, job_id, str(file_path), filename, kb_id
            )
        except Exception as e:
            log.exception("后台上传任务失败")
            upload_job_manager.fail_job(job_id, "index", str(e))

    import asyncio
    asyncio.create_task(_run_upload())

    return JSONResponse(content={
        "job_id": job_id,
        "filename": filename,
        "status": "processing",
        "message": "文件已上传，后台正在处理索引",
    })


# ── 星链知识库 CRUD（Open WebUI 前端调用）──


@router.post("/v1/knowledge/create")
async def create_starlink_knowledge(request: Request):
    """创建星链知识库元数据。"""
    import uuid
    from database import SessionLocal
    from models import StarlinkKnowledge

    user_id, role = await get_user_info(request)
    body = await request.json()
    name = (body.get("name") or "").strip()
    description = (body.get("description") or "").strip()
    is_public = bool(body.get("is_public", False))

    if not name:
        raise HTTPException(status_code=400, detail="名称不能为空")

    kb_id = uuid.uuid4().hex[:12]
    with SessionLocal() as db:
        kb = StarlinkKnowledge(id=kb_id, name=name, description=description,
                               user_id=user_id, is_public=is_public)
        db.add(kb)
        db.commit()
        db.refresh(kb)

    return JSONResponse(content={"id": kb_id, "name": name, "description": description,
                                 "is_public": is_public})


@router.get("/v1/knowledge/list")
async def list_starlink_knowledge(request: Request):
    """列出当前用户可见的星链知识库（自己的 + 公开的，admin 看全部）。"""
    from database import SessionLocal
    from models import StarlinkKnowledge
    from sqlalchemy import or_

    user_id, role = await get_user_info(request)
    with SessionLocal() as db:
        if role == "admin":
            kbs = db.query(StarlinkKnowledge).order_by(StarlinkKnowledge.created_at.desc()).all()
        else:
            kbs = db.query(StarlinkKnowledge).filter(
                or_(StarlinkKnowledge.user_id == user_id, StarlinkKnowledge.is_public == True)
            ).order_by(StarlinkKnowledge.created_at.desc()).all()
        items = [{"id": k.id, "name": k.name, "description": k.description,
                  "user_id": k.user_id, "is_public": k.is_public,
                  "created_at": k.created_at.isoformat()} for k in kbs]

    return JSONResponse(content={"items": items})


@router.get("/v1/knowledge/{kb_id}")
async def get_starlink_knowledge(kb_id: str, request: Request):
    """获取单个星链知识库（需要是创建者、公开的、或 admin）。"""
    from database import SessionLocal
    from models import StarlinkKnowledge

    user_id, role = await get_user_info(request)
    with SessionLocal() as db:
        kb = db.query(StarlinkKnowledge).filter(StarlinkKnowledge.id == kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        if role != "admin" and kb.user_id != user_id and not kb.is_public:
            raise HTTPException(status_code=403, detail="无权访问此知识库")

    return JSONResponse(content={
        "id": kb.id, "name": kb.name, "description": kb.description,
        "user_id": kb.user_id, "is_public": kb.is_public,
        "created_at": kb.created_at.isoformat(),
    })


@router.delete("/v1/knowledge/{kb_id}")
async def delete_starlink_knowledge(kb_id: str, request: Request):
    """删除星链知识库元数据（只有创建者或 admin 可删）。"""
    from database import SessionLocal
    from models import StarlinkKnowledge

    user_id, role = await get_user_info(request)
    with SessionLocal() as db:
        kb = db.query(StarlinkKnowledge).filter(StarlinkKnowledge.id == kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
        if role != "admin" and kb.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权删除此知识库")
        db.delete(kb)
        db.commit()

    return JSONResponse(content={"deleted": True})


@router.get("/v1/documents")
async def list_documents_via_api_key(request: Request):
    """通过 API Key 认证列出知识库文档。"""
    import os
    from document_ops import UPLOAD_DIR

    await verify_api_key(request)

    items = []
    if UPLOAD_DIR.exists():
        for f in sorted(UPLOAD_DIR.rglob("*")):
            if f.is_file():
                rel = str(f.relative_to(UPLOAD_DIR))
                items.append({
                    "filename": rel,
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                })

    return JSONResponse(content={"items": items})


@router.delete("/v1/documents/{filename:path}")
async def delete_document_via_api_key(filename: str, request: Request):
    """通过 API Key 认证删除知识库文档。"""
    import os
    from document_ops import UPLOAD_DIR

    await verify_api_key(request)

    file_path = UPLOAD_DIR / filename
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_file_path = os.path.realpath(str(file_path))

    if not real_file_path.startswith(real_upload_dir + os.sep):
        raise HTTPException(status_code=403, detail="访问被拒绝")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")

    return JSONResponse(content={"deleted": True, "filename": filename})


@router.get("/v1/files/{filename:path}/content")
async def get_document_content_via_api_key(filename: str, request: Request):
    """通过 API Key 认证读取文档内容（供引用链接使用）。"""
    import mimetypes
    from document_ops import UPLOAD_DIR

    await verify_api_key(request)

    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")

    file_path = UPLOAD_DIR / filename
    real_upload_dir = os.path.realpath(str(UPLOAD_DIR))
    real_file_path = os.path.realpath(str(file_path))
    if not real_file_path.startswith(real_upload_dir + os.sep) and real_file_path != real_upload_dir:
        raise HTTPException(status_code=403, detail="访问被拒绝")

    # 精确路径找不到则递归搜索
    if not file_path.is_file():
        for f in UPLOAD_DIR.rglob("*"):
            if f.is_file() and f.name == filename:
                file_path = f
                break

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="文档不存在")

    MAX_PREVIEW_BYTES = 1 * 1024 * 1024
    file_size = file_path.stat().st_size
    try:
        if file_size <= MAX_PREVIEW_BYTES:
            content = file_path.read_text(encoding="utf-8")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(MAX_PREVIEW_BYTES)
            content += f"\n\n... [预览截断]"
    except UnicodeDecodeError:
        if file_size <= MAX_PREVIEW_BYTES:
            content = file_path.read_bytes().decode("utf-8", errors="replace")
        else:
            with open(file_path, "rb") as f:
                content = f.read(MAX_PREVIEW_BYTES).decode("utf-8", errors="replace")
            content += f"\n\n... [预览截断]"

    mime, _ = mimetypes.guess_type(str(file_path))
    return JSONResponse(content={
        "filename": filename,
        "size": file_size,
        "content": content,
    })
