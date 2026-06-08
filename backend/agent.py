import json
import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

import deepseek_patch; deepseek_patch.apply()  # 必须在 init_chat_model 之前
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from tools import (
    get_last_rag_context, reset_tool_call_guards, set_current_user, set_current_kb_ids,
)
from events import set_rag_step_queue
from core.loop_detector import LoopDetector
from core.incremental_save import IncrementalSaveTracker
from core.pagination import MessagePagination
from core.context_compressor import ContextCompressor, _estimate_msg_tokens
from agentic_rag.config import agent as agent_cfg, budget as budget_cfg
from conversation_storage import ConversationStorage
from agent_factory import create_agent_instance, rebuild_agent

agent, model = create_agent_instance()


def hot_reload_agent(model_override=None):
    """热重载 Agent 并更新本模块的全局变量。"""
    global agent, model
    new_agent, new_model, stats = rebuild_agent(model_override=model_override)
    agent = new_agent
    model = new_model
    return stats


save_tracker = IncrementalSaveTracker()
pagination = MessagePagination(default_window=50)
storage = ConversationStorage(save_tracker, pagination)
loop_detector = LoopDetector()
compressor = ContextCompressor(
    budget_tokens=agent_cfg.COMPRESSION_THRESHOLD_AGENT // 4,
    keep_head_rounds=3,
)


def _inject_memory_context(user_id: str, question: str, max_items: int = 5) -> Optional[str]:
    """三层记忆注入：Redis session → PostgreSQL short_term → 对话历史偏好。

    在每次 Agent 调用前同步执行。使用 MemoryInjector 统一服务。
    """
    try:
        from memory.memory_injector import memory_injector
        result = memory_injector.inject(
            user_id=user_id,
            question=question,
            max_items=max_items,
            parallel=False,  # Agent hook 中顺序执行，避免线程池开销
            xml_wrap=False,
        )
        return result.context
    except Exception:
        logger.warning("Memory injection failed for user %s, skipping", user_id, exc_info=True)
        return None


def _load_accessible_kb_ids(user_id: str) -> list[str]:
    """查询用户可访问的知识库 ID 列表（自己的 + 公开的）。"""
    try:
        from database import SessionLocal
        from models import StarlinkKnowledge
        from sqlalchemy import or_
        with SessionLocal() as db:
            kbs = db.query(StarlinkKnowledge.id).filter(
                or_(StarlinkKnowledge.user_id == user_id, StarlinkKnowledge.is_public == True)
            ).all()
            return [k.id for k in kbs]
    except Exception:
        logger.warning("Failed to load kb_ids for user %s", user_id, exc_info=True)
        return []


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并返回响应"""
    set_current_user(user_id)
    set_current_kb_ids(_load_accessible_kb_ids(user_id))
    loop_detector.reset()
    messages = list(storage.load(user_id, session_id))

    # 清理可能残留的 RAG 上下文，避免跨请求污染
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 上下文压缩：Token 超阈值时触发五阶段压缩管线
    if sum(_estimate_msg_tokens(m) for m in messages) > agent_cfg.COMPRESSION_THRESHOLD_AGENT:
        result = compressor.compress(messages)
        messages = result.messages

    messages.append(HumanMessage(content=user_text))

    # 记忆注入：从 Redis 读取用户偏好和事实，注入为 SystemMessage
    memory_ctx = _inject_memory_context(user_id, user_text)
    if memory_ctx:
        memory_msg = SystemMessage(content=memory_ctx)
        memory_msg.__dict__["_supermew_injected"] = True
        messages = [memory_msg] + messages

    # 循环检测：在调用前检查历史消息
    loop_state = loop_detector.detect(messages)
    if loop_state.is_stuck:
        messages = loop_detector.recover(messages, loop_state)

    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": int(os.environ.get("RECURSION_LIMIT", "100"))},
    )
    # 注入的记忆只用于本次调用，不持久化到存储
    if messages and getattr(messages[0], "_supermew_injected", False):
        messages = messages[1:]

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

    messages.append(AIMessage(content=response_content))

    # Bug 4 fix: Remove recovery SystemMessages before saving
    messages = [m for m in messages if not (isinstance(m, SystemMessage) and str(m.content).startswith("[检测到死循环]"))]

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并流式返回响应。

    架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送，
    而非等待工具完成后才显示。
    """
    set_current_user(user_id)
    set_current_kb_ids(_load_accessible_kb_ids(user_id))
    loop_detector.reset()
    messages = list(storage.load(user_id, session_id))

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的事件放入统一输出队列。"""
        def __init__(self):
            self._loop = asyncio.get_running_loop()

        def put_nowait(self, event):
            if isinstance(event, dict):
                if event.get("type") == "rag_context":
                    output_queue.put_nowait(event)
                elif event.get("type") == "rag_step":
                    output_queue.put_nowait({"type": "rag_step", "step": event.get("step", {})})

    set_rag_step_queue(_RagStepProxy())

    # 上下文压缩：Token 超阈值时触发五阶段压缩管线
    if sum(_estimate_msg_tokens(m) for m in messages) > agent_cfg.COMPRESSION_THRESHOLD_AGENT:
        result = compressor.compress(messages)
        messages = result.messages

    messages.append(HumanMessage(content=user_text))

    # 记忆注入：从 Redis/PG 读取用户偏好，用 to_thread 避免阻塞事件循环
    memory_ctx = await asyncio.to_thread(_inject_memory_context, user_id, user_text)
    if memory_ctx:
        memory_msg = SystemMessage(content=memory_ctx)
        memory_msg.__dict__["_supermew_injected"] = True
        messages = [memory_msg] + messages

    # 循环检测：在调用前检查历史消息
    loop_state = loop_detector.detect(messages)
    if loop_state.is_stuck:
        messages = loop_detector.recover(messages, loop_state)

    full_response = ""

    async def _agent_worker():
        """后台任务：运行 agent 并将内容 chunk 推入输出队列。"""
        nonlocal full_response
        try:
            async for msg, metadata in agent.astream(
                {"messages": messages},
                stream_mode="messages",
                config={"recursion_limit": int(os.environ.get("RECURSION_LIMIT", "100"))},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                # 不跳过 tool_call_chunks：DeepSeek 等模型可能同时发送 content + tool_calls
                # content 提取逻辑只读 str/text blocks，不会泄漏 tool_call 参数片段

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
            logger.error("Agent worker error: %s", e, exc_info=True)
            await output_queue.put({"type": "error", "content": "系统执行错误，请稍后重试"})
        finally:
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    rag_trace = None
    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        # RAG 步骤在工具执行期间通过 call_soon_threadsafe 实时入队，不需要等 agent 产出 chunk
        queued_rag_context = None
        while True:
            event = await output_queue.get()
            if event is None:
                break
            # 捕获从线程池传播回来的 RAG 上下文（Bug 2 修复）
            if isinstance(event, dict) and event.get("type") == "rag_context":
                queued_rag_context = event.get("context")
                continue
            yield f"data: {json.dumps(event)}\n\n"

        # 获取 RAG trace：优先从队列传播的上下文获取，回退到 ContextVar
        rag_context = queued_rag_context or get_last_rag_context(clear=True)
        rag_trace = rag_context.get("rag_trace") if rag_context else None

        # 发送增强的 trace 信息（含 confidence, budget, citations）
        if rag_trace:
            yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

            # 发送置信度
            confidence = rag_trace.get("confidence")
            if confidence and isinstance(confidence, dict):
                yield f"data: {json.dumps({'type': 'confidence', 'score': confidence.get('score'), 'level': confidence.get('level'), 'reason': confidence.get('reason', '')})}\n\n"

            # 发送预算状态
            budget = rag_trace.get("budget")
            if budget and isinstance(budget, dict):
                yield f"data: {json.dumps({'type': 'budget', 'used': budget.get('used', 0), 'max': budget.get('max', budget_cfg.MAX_ITERATIONS), 'exhausted': budget.get('exhausted', False)})}\n\n"

            # 发送引用
            citations = rag_trace.get("citations")
            if citations and isinstance(citations, list) and len(citations) > 0:
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

            # 发送降级标记
            if rag_trace.get("is_degraded"):
                yield f"data: {json.dumps({'type': 'degraded', 'message': '系统预算耗尽，回答可能不完整'})}\n\n"

            # 发送拦截信息
            if rag_trace.get("is_blocked"):
                yield f"data: {json.dumps({'type': 'blocked', 'reason': rag_trace.get('block_reason', '查询被安全护栏拦截')})}\n\n"

    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务
        agent_task.cancel()
        try:
            await asyncio.wait_for(agent_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass  # 任务已成功取消或超时
        raise  # 重新抛出 GeneratorExit 以便 FastAPI 正确处理关闭
    finally:
        # 正常结束或异常退出时清理
        set_rag_step_queue(None)
        if not agent_task.done():
            agent_task.cancel()
        # 保存对话（包括 GeneratorExit 场景，先移除注入的记忆，不持久化到存储）
        try:
            if messages and getattr(messages[0], "_supermew_injected", False):
                messages = messages[1:]
            # Bug 4 fix: Remove recovery SystemMessages before saving
            messages = [m for m in messages if not (isinstance(m, SystemMessage) and str(m.content).startswith("[检测到死循环]"))]
            all_messages = list(messages)
            if full_response:
                # 如果 agent 任务未正常完成（如客户端断开），标记为部分响应
                suffix = "[回复被中断]" if not agent_task.done() else ""
                all_messages.append(AIMessage(content=full_response + suffix))
            elif not any(isinstance(m, AIMessage) for m in messages[-2:]):
                all_messages.append(AIMessage(content="[无文本回复]"))
            extra_message_data = [None] * (len(all_messages) - 1) + [{"rag_trace": rag_trace}]
            storage.save(user_id, session_id, all_messages, extra_message_data=extra_message_data)
        except Exception:
            logger.exception("Failed to save conversation for session %s", session_id)
