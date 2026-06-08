"""PostProcess — 后处理与追踪。

收集执行过程中的追踪信息，计算延迟和成本估算。
"""

import logging
import time

from agentic_rag.state import AgenticRAGState
from events import emit_rag_step

logger = logging.getLogger(__name__)


def post_process(state: AgenticRAGState) -> dict:
    """最终后处理：收集追踪信息、估算成本、记录情节记忆。"""
    trace_steps = state.get("trace_steps", [])

    step = {
        "node": "post_process",
        "timestamp": time.time(),
        "has_answer": bool(state.get("answer")),
        "is_degraded": state.get("is_degraded_answer", False),
        "doc_count": len(state.get("retrieved_docs", [])),
        "query_blocked": state.get("_query_blocked", False),
    }
    trace_steps = trace_steps + [step]

    doc_count = len(state.get("retrieved_docs", []))
    estimated_cost = doc_count * 0.001

    # ── v2: 情节记忆记录 ──
    ca = state.get("confidence_assessment")
    is_good_answer = (
        ca and isinstance(ca, dict) and
        ca.get("level") in ("answer", "partial") and
        ca.get("score", 0) >= 0.5
    )
    if is_good_answer:
        filenames = state.get("_retrieved_filenames")
        if filenames:
            try:
                from core.episodic_memory import episodic_memory
                episodic_memory.record_success(
                    state.get("question", ""),
                    filenames,
                )
            except Exception as e:
                logger.warning("episodic memory record failed: %s", e)

    emit_rag_step("✅", "回答完成")

    return {
        "trace_steps": trace_steps,
        "latency_ms": 0,  # 由外部计算
        "cost_usd": estimated_cost,
    }
