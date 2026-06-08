"""TransformQuery — 基础查询转换。

执行简单的清理和标准化，为查询扩展做准备。
"""

from agentic_rag.state import AgenticRAGState


def transform_query(state: AgenticRAGState) -> dict:
    """基础查询转换。"""
    question = state.get("question", "")
    query = state.get("query", question)

    # 基础清理
    cleaned = query.strip()

    # 首次调用时记录原始问题
    original = state.get("original_question")
    if not original:
        original = question

    return {
        "query": cleaned,
        "original_question": original,
    }
