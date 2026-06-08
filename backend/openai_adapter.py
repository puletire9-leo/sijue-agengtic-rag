"""OpenAI 兼容适配层 — 消息格式转换 + SSE 格式转换。

将 Open WebUI 的 OpenAI 格式请求转换为 SuperMew 内部格式，
并将 SuperMew 的 SSE 响应转换回 OpenAI 格式。
"""

import json
import logging
import time
import uuid
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

log = logging.getLogger(__name__)


# ── OpenAI → LangChain 消息转换 ──


def openai_messages_to_langchain(messages: list[dict]) -> list[BaseMessage]:
    """将 OpenAI messages[] 转换为 LangChain 消息列表。

    支持:
    - system / user / assistant / tool 角色
    - 多模态 content（只提取文本部分）
    - tool_calls（assistant 消息中的函数调用）
    """
    result: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "")
        raw_content = msg.get("content", "")

        # 多模态 content: 只提取文本部分
        if raw_content is None:
            content = None
        elif isinstance(raw_content, list):
            text_parts = []
            for block in raw_content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "input_text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        img_url = ""
                        url_data = block.get("image_url", {})
                        if isinstance(url_data, dict):
                            img_url = url_data.get("url", "")
                        elif isinstance(url_data, str):
                            img_url = url_data
                        if img_url:
                            text_parts.append(f"[用户发送了一张图片: {img_url}]")
                        else:
                            text_parts.append("[用户发送了一张图片]")
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        else:
            content = str(raw_content)

        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            # 处理 tool_calls
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                lc_tool_calls = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        log.warning("Failed to parse tool call arguments: %s", args_str[:100])
                        args = {}
                    lc_tool_calls.append({
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "name": func.get("name", ""),
                        "args": args,
                    })
                result.append(AIMessage(content=content, tool_calls=lc_tool_calls))
            else:
                result.append(AIMessage(content=content))
        elif role == "tool":
            result.append(ToolMessage(
                content=content,
                tool_call_id=msg.get("tool_call_id", ""),
            ))
    return result


def extract_last_user_message(messages: list[dict]) -> str:
    """从 OpenAI messages[] 中提取最后一条 user 消息。"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("text", "input_text"):
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return "\n".join(text_parts)
            return str(content or "")
    return ""


# ── SuperMew SSE → OpenAI SSE 格式转换 ──


def make_chunk_id() -> str:
    """生成 OpenAI 格式的 chunk ID。"""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def to_openai_chunk(
    content: str,
    model: str,
    chunk_id: str,
    created: Optional[int] = None,
    first: bool = False,
) -> str:
    """将文本内容转换为 OpenAI SSE chunk。"""
    delta = {}
    if first:
        delta["role"] = "assistant"
    delta["content"] = content

    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def to_openai_reasoning_chunk(
    reasoning_content: str,
    model: str,
    chunk_id: str,
    created: Optional[int] = None,
) -> str:
    """将 RAG 步骤转换为 OpenAI reasoning_content SSE chunk。

    Open WebUI 支持 reasoning_content 字段，用于展示模型的思考过程。
    我们借用这个字段来展示 RAG 的检索/评估/重写步骤。
    """
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "reasoning_content": reasoning_content,
            },
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def to_openai_sources_event(
    sources: list,
    model: str,
    chunk_id: str,
    created: Optional[int] = None,
) -> str:
    """将 sources 数据作为 SSE 事件发送。

    这不是标准 OpenAI 协议，但 Open WebUI 可以解析自定义字段。
    通过在 finish_reason=stop 之前发送，Open WebUI 可以捕获并展示引用。
    """
    event = {
        "id": chunk_id,
        "object": "chat.completion.sources",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "sources": sources,
    }
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def to_openai_finish(
    model: str,
    chunk_id: str,
    created: Optional[int] = None,
    finish_reason: str = "stop",
    usage: Optional[dict] = None,
) -> str:
    """生成结束 chunk（finish_reason=stop/tool_calls）。"""
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    }
    if usage:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def to_openai_done() -> str:
    """生成 [DONE] 事件。"""
    return "data: [DONE]\n\n"


_HTTP_TO_OPENAI_ERROR_TYPE: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    408: "timeout",
    409: "conflict",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "server_error",
    502: "server_error",
    503: "server_error",
}


def to_openai_error(message: str, code: int = 500) -> dict:
    """生成 OpenAI 格式的错误响应（非流式）。"""
    error_type = _HTTP_TO_OPENAI_ERROR_TYPE.get(code, "server_error")
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": str(code),
        }
    }


def to_openai_error_sse(message: str, model: str = "supermew-rag", chunk_id: Optional[str] = None) -> str:
    """生成 OpenAI 格式的错误 SSE 事件。"""
    chunk = {
        "id": chunk_id or make_chunk_id(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": f"\n\n[错误: {message}]"},
            "finish_reason": "stop",
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"


def format_citations_as_text(citations: list) -> str:
    """将引用列表格式化为文本，嵌入到 content 尾部。"""
    if not citations:
        return ""
    sources = []
    for c in citations:
        source = c.get("source", "")
        page = c.get("page", "")
        if source:
            if page and page != "N/A":
                sources.append(f"{source} (Page {page})")
            else:
                sources.append(source)
    if not sources:
        return ""
    return "\n\n---\n来源: " + ", ".join(sources)


def format_citations_as_sources(citations: list, retrieved_docs: list = None) -> list:
    """将 SuperMew citations 转换为 Open WebUI 的 sources 格式。

    Open WebUI 期望的格式:
    [{
        "source": {"name": "filename.pdf", "id": "file_id"},
        "document": ["chunk text..."],
        "metadata": [{"source": "filename.pdf", "page": 3}]
    }]

    Args:
        citations: SuperMew 引用列表 [{"source": filename, "page": page, "score": score}]
        retrieved_docs: 检索到的文档列表（可选，用于提取 chunk text）

    Returns:
        Open WebUI sources 格式列表
    """
    if not citations:
        return []

    # 按 filename 分组检索文档
    docs_by_file = {}
    if retrieved_docs:
        for doc in retrieved_docs:
            fname = doc.get("filename", "")
            if fname:
                docs_by_file.setdefault(fname, []).append(doc)

    sources = []
    for c in citations:
        filename = c.get("source", "")
        page = c.get("page", "")
        if not filename:
            continue

        # 从检索文档中提取该文件的 chunk text
        documents = []
        metadata = []
        file_docs = docs_by_file.get(filename, [])
        if file_docs:
            for doc in file_docs[:3]:  # 最多 3 个 chunk
                text = doc.get("text", "")
                if text:
                    documents.append(text[:500])
                    metadata.append({
                        "source": filename,
                        "page": doc.get("page_number", page),
                        "chunk_id": doc.get("chunk_id", ""),
                        "score": doc.get("score", 0),
                        "file_id": f"starlink/{filename}",
                    })

        if not documents:
            # 没有检索文档时，只放文件名
            documents = [f"[{filename}]"]
            metadata = [{"source": filename, "page": page}]

        sources.append({
            "source": {
                "name": filename,
                "id": filename.replace(" ", "_").lower(),
            },
            "document": documents,
            "metadata": metadata,
        })

    return sources


# ── 非流式响应格式 ──


def to_openai_response(
    content: str,
    model: str,
    citations: Optional[list] = None,
    usage: Optional[dict] = None,
    sources: Optional[list] = None,
    retrieved_docs: Optional[list] = None,
) -> dict:
    """生成非流式 OpenAI chat.completion 响应。"""
    full_content = content
    if citations:
        full_content = content + format_citations_as_text(citations)

    response = {
        "id": make_chunk_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": full_content,
            },
            "finish_reason": "stop",
        }],
    }
    if usage:
        response["usage"] = usage
    else:
        response["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # Open WebUI sources 格式（交互式引用卡片）
    if sources:
        response["sources"] = sources
    elif citations:
        response["sources"] = format_citations_as_sources(citations, retrieved_docs)

    return response
