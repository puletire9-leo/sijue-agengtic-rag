"""GuardrailQuery — 支持运行时开关。"""

from agentic_rag.state import AgenticRAGState
from routers.guardrail_toggle import is_guardrail_enabled
from guardrails.query_guard import QueryGuard
from agentic_rag.config import guardrail as guardrail_cfg
from agentic_rag.llm import get_lightweight_llm

_guard = QueryGuard(
    enabled=True,
    max_length=guardrail_cfg.MAX_QUERY_LENGTH,
    llm_enabled=guardrail_cfg.LLM_ENABLED,
)


def guardrail_query(state: AgenticRAGState) -> dict:
    """运行时开关控制 — 关闭时直接放行。"""
    if not is_guardrail_enabled():
        return {
            "_query_blocked": False,
            "_block_reason": None,
            "query": state.get("question", ""),
        }

    question = state.get("question", "")
    result = _guard.check_with_llm(question, get_lightweight_llm())

    return {
        "_query_blocked": result.blocked,
        "_block_reason": result.reason if result.blocked else None,
        "query": result.transformed_text or question,
    }
