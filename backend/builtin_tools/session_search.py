"""会话搜索工具 — 跨历史对话检索。

对应 hermes-agent 的 session_search_tool.py。
"""

from langchain_core.tools import tool


@tool("search_conversations")
def search_conversations(query: str, message_limit: int = 100) -> str:
    """Search through the user's past conversation history.

    Use this when the user asks about something discussed in a previous conversation
    (e.g. "what did we talk about last time?", "when did we discuss the budget report?").

    Args:
        query: Keywords to search for in past conversations
        message_limit: Maximum number of messages to scan per session (default 100)

    Returns:
        Matching conversation snippets with session IDs and timestamps
    """
    try:
        from tools import get_current_user
        user_id = get_current_user()

        from agent import storage

        # 列出所有会话
        sessions = storage.list_session_infos(user_id)
        if not sessions:
            return "没有找到历史对话记录。"

        results = []
        for sess in sessions[:20]:  # 最多检查最近20个会话
            sid = sess.get("session_id", "")
            messages = storage.get_session_messages(user_id, sid)
            for msg in messages[:message_limit]:
                content = str(msg.get("content", "") or "")
                if query.lower() in content.lower():
                    snippet = content[:200]
                    results.append({
                        "session_id": sid,
                        "timestamp": msg.get("timestamp", ""),
                        "snippet": snippet + ("..." if len(content) > 200 else ""),
                    })
                    break  # 每个会话只取第一条匹配

        if not results:
            return f"在所有 {len(sessions)} 个历史会话中未找到与 '{query}' 相关的内容。"

        lines = [f"在 {len(results)} 个历史会话中找到 '{query}' 相关内容:"]
        for r in results[:10]:
            lines.append(
                f"- [{r['session_id'][:12]}...] {r['timestamp']}\n"
                f"  \"{r['snippet']}\""
            )
        return "\n".join(lines)
    except Exception as e:
        return f"会话搜索失败: {e}"
