"""澄清工具 — Agent 在问题模糊时主动反问。

对应 hermes-agent 的 clarify_tool.py。
"""

from langchain_core.tools import tool


@tool("ask_clarification")
def ask_clarification(question_for_user: str, options: str = "") -> str:
    """Ask the user for clarification when the question is ambiguous.

    Use this BEFORE calling search_knowledge_base when:
    1. The question could be interpreted in multiple ways
    2. Key details are missing (e.g. time range, person name, version number)
    3. You need to narrow down the search scope

    DO NOT use this when:
    - The question is clear enough to search directly
    - You can make a reasonable assumption and verify with search results

    Args:
        question_for_user: The clarification question to ask
        options: Optional comma-separated options (e.g. "2023, 2024, 2025")

    Returns:
        Formatted clarification request
    """
    if options:
        return (
            f"[需要澄清] {question_for_user}\n"
            f"可选: {options}\n"
            "请选择或提供更多信息。"
        )
    return f"[需要澄清] {question_for_user}\n请提供更多细节以便精准检索。"
