"""InjectMemory v4 — 注入三层记忆上下文。

使用 memory.memory_injector.MemoryInjector 统一服务，
消除与 agent.py 的重复实现。
"""

import logging

from agentic_rag.state import AgenticRAGState
from events import emit_rag_step

log = logging.getLogger(__name__)


def inject_memory(state: AgenticRAGState) -> dict:
    """三层记忆注入。

    查询顺序: Redis session → PostgreSQL short_term → 近期对话偏好
    合并后按重要性 + 相关性排序，取 top 5。
    """
    from memory.memory_injector import memory_injector

    user_id = state.get("user_id", "")
    question = state.get("question", "")

    if not user_id or not question:
        return {"memory_context": None, "session_memory": [], "memory_sources": []}

    result = memory_injector.inject(
        user_id=user_id,
        question=question,
        max_items=5,
        parallel=True,
        xml_wrap=True,
    )

    if not result.context:
        return {"memory_context": None, "session_memory": [], "memory_sources": []}

    emit_rag_step("🧠", f"记忆注入: {len(result.memories)} 条 (L0+L1+L2)", ",".join(result.sources[:3]))

    return {
        "memory_context": result.context,
        "session_memory": result.memories,
        "memory_sources": result.sources,
    }
