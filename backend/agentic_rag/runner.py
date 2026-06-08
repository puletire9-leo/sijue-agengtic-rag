"""Runner — Agentic RAG 图运行的同步/异步接口。

提供对外统一调用入口，供 LangChain Agent 工具和 FastAPI 路由使用。
"""

import logging
from typing import Optional

from agentic_rag.config import budget as budget_cfg
from agentic_rag.state import create_initial_state
from agentic_rag.graph import get_agentic_rag_graph

log = logging.getLogger(__name__)


def run_agentic_rag_sync(
    question: str,
    session_id: str = "default",
    user_id: str = "default",
    max_iterations: Optional[int] = None,
    kb_ids: Optional[list[str]] = None,
) -> dict:
    """同步运行 Agentic RAG 管线。

    LangGraph StateGraph.invoke() 本身是同步的，即使节点包含 async 函数也可在
    事件循环中运行。这个包装函数确保工具调用时能正确执行。

    Args:
        question: 用户问题
        session_id: 会话 ID
        user_id: 用户 ID
        max_iterations: 预算上限
        kb_ids: 可访问的知识库 ID 列表

    Returns:
        完整的状态字典（包含 answer, retrieved_docs, citations, trace_steps 等）
    """
    initial_state = create_initial_state(
        question=question,
        session_id=session_id,
        user_id=user_id,
        max_iterations=max_iterations,
        kb_ids=kb_ids,
    )

    try:
        final_state = get_agentic_rag_graph().invoke(initial_state)
        return final_state
    except Exception as e:
        # 图执行异常时返回错误状态
        log.exception("Agentic RAG execution failed")
        return {
            **initial_state,
            "answer": "抱歉，系统处理出现问题，请稍后重试。",
            "is_degraded_answer": True,
            "trace_steps": initial_state.get("trace_steps", []) + [
                {"node": "error", "error": str(e), "timestamp": __import__("time").time()}
            ],
        }


def format_rag_result(state: dict) -> dict:
    """将完整状态格式化为 tools.py 兼容的检索结果 + 增强上下文。

    Returns:
        {"docs": [...], "rag_trace": {...}, "confidence": {...}, "budget": {...}, "citations": [...]}
    """
    docs = state.get("retrieved_docs") or state.get("filtered_docs", [])
    meta = state.get("retrieval_metadata", {})

    # 置信评估
    confidence = state.get("confidence_assessment")
    if not confidence or not isinstance(confidence, dict):
        confidence = {"level": "unknown", "score": 0.5}

    # 预算状态
    budget_dict = state.get("iteration_budget", {})
    budget = {
        "used": budget_dict.get("used", 0),
        "max": budget_dict.get("max", budget_cfg.MAX_ITERATIONS),
        "exhausted": state.get("budget_exhausted", False),
        "grace": budget_dict.get("grace", 1),
    }

    # 门控信息
    is_degraded = state.get("is_degraded_answer", False)
    is_blocked = state.get("_query_blocked", False)
    block_reason = state.get("_block_reason")
    relevance = state.get("relevance_grade")

    # 引用
    citations = state.get("citations", [])
    rewrite_strategy = state.get("rewrite_strategy")
    retry_count = state.get("retry_count", 0)

    # 构建兼容的 rag_trace
    rag_trace = {
        "tool_used": len(docs) > 0,
        "tool_name": "search_knowledge_base",
        "query": state.get("query", ""),
        "retrieved_chunks": docs,
        "retrieval_mode": meta.get("retrieval_mode", "hybrid"),
        "rerank_enabled": meta.get("rerank_enabled"),
        "rerank_applied": meta.get("rerank_applied"),
        "rerank_model": meta.get("rerank_model"),
        "auto_merge_enabled": meta.get("auto_merge_enabled"),
        "auto_merge_applied": meta.get("auto_merge_applied"),
        "auto_merge_replaced_chunks": meta.get("auto_merge_replaced_chunks", 0),
        # 新增增强字段
        "confidence": confidence,
        "budget": budget,
        "citations": citations,
        "is_degraded": is_degraded,
        "is_blocked": is_blocked,
        "block_reason": block_reason,
        "relevance": relevance,
        "rewrite_strategy": rewrite_strategy,
        "retry_count": retry_count,
    }
    return {
        "docs": docs,
        "rag_trace": rag_trace,
        "confidence": confidence,
        "budget": budget,
        "citations": citations,
        "is_degraded": is_degraded,
        "is_blocked": is_blocked,
    }
