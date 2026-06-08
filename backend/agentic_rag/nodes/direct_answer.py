"""DirectAnswer — 直接回答（不经过检索路径）。"""

from langchain_core.messages import HumanMessage, AIMessage

from agentic_rag.state import AgenticRAGState
from agentic_rag.llm import get_llm
from events import emit_rag_step


DIRECT_ANSWER_PROMPT = (
    "你是一个友好的 AI 助手。请直接回答用户的问题，不要编造事实。\n"
    "如果问题被系统拦截，请礼貌地说明无法回答。\n\n"
    "用户：{question}"
)

BLOCKED_MESSAGE = (
    "您的输入包含系统不允许的内容，已被拦截。\n"
    "请确保您的输入不包含注入代码、危险命令或敏感信息。"
)


def direct_answer(state: AgenticRAGState) -> dict:
    """不经过检索，直接生成回答。"""
    # 如果被查询护栏拦截
    if state.get("_query_blocked"):
        emit_rag_step("🚫", "查询已被安全护栏拦截", state.get("_block_reason", ""))
        return {
            "answer": BLOCKED_MESSAGE,
            "citations": [],
            "messages": [AIMessage(content=BLOCKED_MESSAGE)],
        }

    question = state.get("question", "")

    # 尝试用 LLM 生成回答
    llm = get_llm()
    if llm:
        try:
            msg = HumanMessage(content=DIRECT_ANSWER_PROMPT.format(question=question))
            response = llm.invoke([msg])
            answer = response.content or ""
        except Exception:
            answer = "系统暂时无法处理您的问题，请稍后重试或换个方式提问。"
    else:
        answer = "系统暂时无法处理您的问题，请稍后重试或换个方式提问。"

    return {
        "answer": answer,
        "citations": [],
        "messages": [AIMessage(content=answer)],
        "needs_retrieval": False,
    }
