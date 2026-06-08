"""GradeDocuments — LLM 评估文档相关性。

决定是生成回答（相关）还是重写查询重试（不相关）。
"""

import json
import logging

from agentic_rag.state import AgenticRAGState
from agentic_rag.llm import get_grader
from events import emit_rag_step
from metrics import LLM_CALLS

logger = logging.getLogger(__name__)


TIERED_GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved documents to a user question.\n"
    "用户问题是中文，检索文档也可能是中文。请基于语义相关性判断。\n\n"
    "Here are the retrieved documents:\n\n{context}\n\n"
    "Here is the user question: {question}\n\n"
    "请评估整批文档对回答问题的帮助程度，返回以下三个级别之一：\n"
    '- "full": 多数文档（3篇以上）直接相关，可以充分回答问题\n'
    '- "partial": 少数文档（1-2篇）部分相关，可以提供有限但有价值的回答\n'
    '- "none": 没有文档与问题相关，需要重新检索\n'
    'Return ONLY a JSON object: {{"level": "full/partial/none", "relevant_count": 数字, "reason": "简短理由(中文)"}}'
)


def _parse_tiered_response(content: str) -> dict:
    """Parse tiered grade response. Default: none (retry)."""
    text = content.strip()
    try:
        clean = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        level = parsed.get("level", "none")
        if level not in ("full", "partial", "none"):
            level = "none"
        return {"level": level, "relevant_count": parsed.get("relevant_count", 0)}
    except (json.JSONDecodeError, TypeError):
        lower = text.lower()
        if "full" in lower or "充分" in lower:
            return {"level": "full", "relevant_count": 5}
        if "partial" in lower or "部分" in lower:
            return {"level": "partial", "relevant_count": 2}
        return {"level": "none", "relevant_count": 0}


def grade_documents(state: AgenticRAGState) -> dict:
    """评估检索文档的相关性，决定下一步路由。"""
    question = state.get("question", "")
    docs = state.get("reranked_docs") or state.get("filtered_docs") or state.get("retrieved_docs", [])

    if not docs:
        emit_rag_step("⚠️", "未检索到文档，将重写查询", "")
        return {
            "relevance_grade": "no",
            "filtered_docs": [],
        }

    grader = get_grader()
    if not grader:
        emit_rag_step("✅", "文档相关性评估通过（无打分模型，默认通过）", "")
        return {
            "relevance_grade": "yes",
            "filtered_docs": docs,
        }

    # 将文档合并为上下文（取前 5 篇，每篇前 1200 字符）
    context_parts = []
    for i, doc in enumerate(docs[:5], 1):
        text = doc.get("text", "")
        context_parts.append(f"[{i}] {text[:1200]}")
    context = "\n\n".join(context_parts)

    try:
        prompt = TIERED_GRADE_PROMPT.format(question=question, context=context)
        LLM_CALLS.labels(model="tier2", purpose="grade").inc()
        response = grader.invoke([{"role": "user", "content": prompt}])
        result = _parse_tiered_response(response.content or "")
        level = result["level"]
    except Exception:
        emit_rag_step("⚠️", "文档评估异常，保守视为相关")
        return {
            "relevance_grade": "yes",
            "filtered_docs": docs,
        }

    if level == "full":
        emit_rag_step("✅", f"文档相关性评估：完整相关（{result['relevant_count']}篇相关）")
        return {
            "relevance_grade": "yes",
            "filtered_docs": docs,
        }
    elif level == "partial":
        emit_rag_step("⚠️", f"文档相关性评估：部分相关（{result['relevant_count']}篇相关），仍尝试生成")
        return {
            "relevance_grade": "partial",
            "filtered_docs": docs,
        }
    else:
        emit_rag_step("⚠️", "文档相关性不足，将重写查询重试")
        return {
            "relevance_grade": "no",
            "filtered_docs": docs,
        }
