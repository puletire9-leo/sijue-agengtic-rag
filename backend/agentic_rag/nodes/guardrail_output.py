"""GuardrailOutput — 支持运行时开关。"""

from agentic_rag.state import AgenticRAGState
from routers.guardrail_toggle import is_guardrail_enabled
from guardrails.output_guard import OutputGuard
from agentic_rag.config import guardrail as guardrail_cfg
from agentic_rag.llm import get_lightweight_llm

_guard = OutputGuard(
    enabled=True,
    llm_enabled=guardrail_cfg.LLM_ENABLED,
)


def guardrail_output(state: AgenticRAGState) -> dict:
    """运行时开关控制 — 关闭时直接放行。"""
    if not is_guardrail_enabled():
        return {"_redactions": []}

    answer = state.get("answer", "")
    if not answer:
        return {"_redactions": []}

    result = _guard.check_with_llm(answer, get_lightweight_llm())

    if result.blocked:
        redactions = result.redactions if hasattr(result, "redactions") and result.redactions else []
        if not redactions and hasattr(result, "matched_rules") and result.matched_rules:
            redactions = [r.name for r in result.matched_rules]
        if not redactions and result.reason:
            redactions = [result.reason]
        return {
            "answer": "[回答已被安全护栏拦截]",
            "is_degraded_answer": True,
            "_redactions": redactions,
        }

    return {
        "answer": result.redacted_text or answer,
        "_redactions": result.redactions if hasattr(result, "redactions") else [],
    }
