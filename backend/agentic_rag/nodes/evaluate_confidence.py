"""EvaluateConfidence — LLM 置信评估与三路路由。

评估回答质量，决定：
- answer → 输出
- retry → 重试检索
- partial → 降级输出
"""

import logging
logger = logging.getLogger(__name__)

from agentic_rag.state import AgenticRAGState
from agentic_rag.schemas import ConfidenceAssessment
from agentic_rag.config import ConfidenceConfig
from agentic_rag.llm import get_lightweight_llm
from events import emit_rag_step


CONFIDENCE_PROMPT = (
    "你是一个回答质量评估员。请评估 AI 回答的质量。\n"
    "用户问题：{question}\n"
    "AI 回答：{answer}\n"
    "检索文档：{context}\n\n"
    "仅返回 JSON：{{\"score\": 0~1, \"level\": \"answer/retry/partial\", \"reason\": \"理由\"}}\n"
    "- score >= 0.6: level=answer（高质量，直接输出）\n"
    "- score 0.3~0.6: level=partial（部分可靠，降级输出）\n"
    "- score < 0.3: level=retry（质量低，重试检索）\n"
)


def evaluate_confidence(state: AgenticRAGState) -> dict:
    """评估回答置信度并路由。"""
    question = state.get("question", "")
    answer = state.get("answer", "")
    docs = state.get("reranked_docs") or state.get("filtered_docs") or state.get("retrieved_docs", [])

    if not answer:
        return {
            "confidence_assessment": ConfidenceAssessment(
                score=0, level="retry", reason="回答为空，需要重试"
            ).model_dump(),
        }

    # 如果有检索文档但回答极短，降低置信度（Bug #3 fix: threshold 20→5）
    if docs and len(answer) < 5:
        return {
            "confidence_assessment": ConfidenceAssessment(
                score=0.3, level="partial",
                reason="回答过短，可能未充分利用检索文档"
            ).model_dump(),
        }

    llm = get_lightweight_llm()
    if not llm:
        # 无轻量模型时默认通过
        return {
            "confidence_assessment": ConfidenceAssessment(
                score=0.7, level="answer", reason="默认通过（无评估模型）"
            ).model_dump(),
        }

    # 准备上下文摘要
    context_summary = ""
    if docs:
        texts = [d.get("text", "")[:200] for d in docs[:3]]
        context_summary = "\n".join(texts)

    try:
        prompt = CONFIDENCE_PROMPT.format(
            question=question[:500],
            answer=answer[:2000],
            context=context_summary[:1000] or "无检索文档",
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        import json
        parsed = json.loads(content)
        score = float(parsed.get("score", 0.5))
        level = str(parsed.get("level", "partial"))
        reason = str(parsed.get("reason", ""))

        # 校验 level
        if level not in ("answer", "retry", "partial"):
            level = "partial"

        # 强制一致性：score 与 level 不匹配时修正
        if score >= 0.6 and level == "retry":
            level = "answer"
        if score < 0.3 and level == "answer":
            level = "retry"
        if 0.3 <= score < 0.6 and level == "retry":
            level = "partial"

        emit_rag_step(
            "📊",
            f"回答质量评估: {level}",
            f"score={score:.2f}, {reason}",
        )

        return {
            "confidence_assessment": ConfidenceAssessment(
                score=score, level=level, reason=reason
            ).model_dump(),
        }
    except Exception as e:
        logger.warning("confidence evaluation failed: %s", e)
        return {
            "confidence_assessment": ConfidenceAssessment(
                score=0.5, level="partial", reason="评估异常，保守处理"
            ).model_dump(),
        }
