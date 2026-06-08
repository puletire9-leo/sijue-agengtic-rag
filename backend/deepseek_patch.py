"""Monkey-patch: 修复 langchain-openai 丢失 DeepSeek reasoning_content。

DeepSeek thinking mode 返回 reasoning_content，但 langchain-openai 的
_convert_dict_to_message/_convert_chunk_to_message 没有提取该字段到
AIMessage.additional_kwargs，导致多轮工具调用时 API 报 400 错误。

加载此模块即可修复。
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from langchain_openai.chat_models import base as _base


def _patched_convert_dict_to_message(_dict: Mapping[str, Any]):
    """与原函数一致，额外提取 reasoning_content 到 additional_kwargs。"""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
        convert_to_messages,
    )
    from langchain_core.output_parsers.openai_tools import (
        make_invalid_tool_call,
        parse_tool_call,
    )

    role = _dict.get("role")
    name = _dict.get("name")
    id_ = _dict.get("id")

    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)

    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict[str, Any] = {}

        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)

        # ── 修复 1: 非流式响应中提取 reasoning_content ──
        if rc := _dict.get("reasoning_content"):
            additional_kwargs["reasoning_content"] = rc

        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )

        if audio := _dict.get("audio"):
            additional_kwargs["audio"] = audio

        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )

    if role in ("system", "developer"):
        additional_kwargs = {"__openai_role__": role} if role == "developer" else {}
        return SystemMessage(
            content=_dict.get("content", ""),
            name=name,
            id=id_,
            additional_kwargs=additional_kwargs,
        )

    if role == "tool":
        additional_kwargs = {}
        if name := _dict.get("name"):
            additional_kwargs["name"] = name
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id", ""),
            name=name,
            id=id_,
            additional_kwargs=additional_kwargs,
        )

    # Fallback: 未知角色试图用标准转换
    _converted = convert_to_messages([_dict])
    return _converted[0] if _converted else AIMessage(content="")


def _patched_convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type
):
    """与原 _convert_delta_to_message_chunk 一致，额外提取 reasoning_content。

    这个函数是流式响应的核心转换路径，代理的 astream 走这条路。
    """
    from langchain_core.messages import (
        AIMessageChunk,
        HumanMessageChunk,
        SystemMessageChunk,
        ToolMessageChunk,
    )
    from langchain_core.messages.tool import tool_call_chunk

    id_ = _dict.get("id")
    role = str(_dict.get("role", ""))
    content = str(_dict.get("content") or "")
    additional_kwargs: dict[str, Any] = {}

    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call

    # ── 修复 2: 流式 delta 中提取 reasoning_content ──
    if rc := _dict.get("reasoning_content"):
        additional_kwargs["reasoning_content"] = rc

    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        for i, rtc in enumerate(raw_tool_calls):
            try:
                func = rtc.get("function") or {}
                tc = tool_call_chunk(
                    name=func.get("name") if isinstance(func, dict) else None,
                    args=func.get("arguments") if isinstance(func, dict) else None,
                    id=rtc.get("id"),
                    index=rtc.get("index", i),
                )
                tool_call_chunks.append(tc)
            except Exception as e:
                logger.debug("tool_call chunk %d parse error (skipped): %s", i, e)

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,
        )
    if role in ("system", "developer") or default_class == SystemMessageChunk:
        if role == "developer":
            additional_kwargs = {"__openai_role__": "developer"}
        else:
            additional_kwargs = {}
        return SystemMessageChunk(
            content=content,
            id=id_,
            additional_kwargs=additional_kwargs,
        )
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict.get("tool_call_id", ""), id=id_
        )
    from langchain_core.messages import convert_to_messages
    _converted = convert_to_messages([_dict])
    return _converted[0] if _converted else AIMessageChunk(content="")





# ── 修复 3: wrapper 方式追加 reasoning_content，不重写原函数 ──
_original_convert_message_to_dict = _base._convert_message_to_dict


def _wrapper_convert_message_to_dict(message, api: str = "chat/completions") -> dict:
    """调用原函数，仅追加 reasoning_content。"""
    result = _original_convert_message_to_dict(message, api)
    if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
        rc = message.additional_kwargs.get('reasoning_content')
        if rc:
            result['reasoning_content'] = rc
            # Debug: log when RC is being passed back
            logging.getLogger('deepseek_patch').debug(
                'reasoning_content passed back (%d chars)', len(str(rc))
            )
    # Check if message has tool_calls but no RC - this is the error case
    if hasattr(message, 'tool_calls') and message.tool_calls:
        rc = message.additional_kwargs.get('reasoning_content') if hasattr(message, 'additional_kwargs') else None
        if not rc:
            logging.getLogger('deepseek_patch').debug(
                'AIMessage has tool_calls but no reasoning_content (expected for non-DeepSeek models).'
            )
    return result


def apply():
    """应用 monkey-patch。"""
    _base._convert_dict_to_message = _patched_convert_dict_to_message
    _base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk
    _base._convert_message_to_dict = _wrapper_convert_message_to_dict
