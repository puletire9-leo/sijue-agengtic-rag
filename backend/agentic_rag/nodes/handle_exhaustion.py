"""HandleExhaustion — 预算耗尽处理。

当迭代预算耗尽时，生成降级回答并标记。
"""

from langchain_core.messages import AIMessage

from agentic_rag.state import AgenticRAGState
from events import emit_rag_step


EXHAUSTION_MESSAGE = (
    "⚠️ **提示**: 经过多次检索，知识库中未找到与您问题直接相关的文档。"
    "以下回答基于模型的通用知识生成，仅供参考，建议查阅原始资料确认。\n\n"
    "如果您希望获得基于知识库的回答，可以尝试：\n"
    "1) 换一种方式描述问题\n"
    "2) 使用更简洁的关键词\n"
    "3) 确认相关文档已上传到知识库\n"
)


def handle_exhaustion(state: AgenticRAGState) -> dict:
    """处理预算耗尽状态，生成降级回答。"""
    emit_rag_step("⚠️", "预算耗尽，输出降级回答")

    answer = state.get("answer")
    docs = state.get("retrieved_docs") or state.get("filtered_docs", [])

    if answer:
        degraded = f"{EXHAUSTION_MESSAGE}\n\n{answer}"
    elif docs:
        doc_summary = "\n".join([
            f"- {d.get('filename', 'Unknown')}" for d in docs[:5]
        ])
        degraded = (
            f"{EXHAUSTION_MESSAGE}\n\n"
            f"已检索到 {len(docs)} 条相关文档：\n{doc_summary}\n\n"
            "请重试或精简问题。"
        )
    else:
        degraded = "抱歉，系统无法完成您的请求。请稍后重试。"

    # 保留已有 citations（来自之前的 generate_answer）
    existing_citations = state.get("citations", [])

    return {
        "answer": degraded,
        "is_degraded_answer": True,
        "budget_exhausted": True,
        "citations": existing_citations,
        "messages": [AIMessage(content=degraded)],
    }
