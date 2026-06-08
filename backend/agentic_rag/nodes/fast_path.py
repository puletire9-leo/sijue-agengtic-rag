"""FastPath — 快速路径分类：是否需要检索。"""

from agentic_rag.state import AgenticRAGState
from agentic_rag.llm import get_lightweight_llm


FAST_PATH_PROMPT = (
    "你是一个分类器，判断用户问题是否需要检索知识库。\n"
    "仅返回 JSON：{{\"needs_retrieval\": true/false, \"confidence\": 0~1, \"reason\": \"简短理由\"}}\n\n"
    "需要检索的场景：询问公司政策、技术文档、产品信息、数据报告、具体事实、"
    "需要引用来源的问题。\n"
    "不需要检索的场景：打招呼、闲聊、简单问候、表达感谢、日常对话。\n\n"
    "用户问题：{question}"
)


def fast_path(state: AgenticRAGState) -> dict:
    """判断是否走检索路径还是直接回答。"""
    question = state.get("question", "")

    # 默认走检索
    if not question or len(question.strip()) < 2:
        return {
            "needs_retrieval": False,
            "retrieval_confidence": 1.0,
            "decision_source": "empty_question",
            "fast_path_result": {"reason": "问题为空，直接回答"},
        }

    llm = get_lightweight_llm()
    if not llm:
        # 没有轻量模型时保守走检索
        return {
            "needs_retrieval": True,
            "retrieval_confidence": 0.5,
            "decision_source": "no_llm_default_retrieval",
        }

    try:
        response = llm.invoke(FAST_PATH_PROMPT.format(question=question))
        content = response.content.strip()
        # 去掉可能的 markdown 代码块
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        import json
        parsed = json.loads(content)
        needs = bool(parsed.get("needs_retrieval", True))
        conf = float(parsed.get("confidence", 0.5))
        reason = str(parsed.get("reason", ""))

        # 低置信度时强制走检索
        if conf < 0.4:
            needs = True

        return {
            "needs_retrieval": needs,
            "retrieval_confidence": conf,
            "decision_source": reason or "llm_classifier",
            "fast_path_result": None if needs else {"reason": reason},
        }
    except Exception:
        return {
            "needs_retrieval": True,
            "retrieval_confidence": 0.5,
            "decision_source": "classifier_error_default_retrieval",
        }
