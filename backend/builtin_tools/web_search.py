"""Web 搜索工具 — 知识库没有答案时的 fallback。

使用 DuckDuckGo Instant Answer API（免费，无需 API Key）。
"""

import re
from typing import Optional

import requests
from langchain_core.tools import tool


_DDG_API = "https://api.duckduckgo.com/"

# 移除 HTML 标签
_HTML_RE = re.compile(r"<[^>]+>")


def _clean_html(text: str) -> str:
    return _HTML_RE.sub("", text).strip()


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for information when the knowledge base doesn't have the answer.

    Use this as a fallback when search_knowledge_base returns no results or
    the user explicitly asks for real-time/latest information.

    Args:
        query: Search query string (be specific, use keywords)

    Returns:
        Search result snippets
    """
    try:
        resp = requests.get(
            _DDG_API,
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=10,
            headers={"User-Agent": "SuperMew/0.1 RAG Agent"},
        )
        resp.raise_for_status()
        data = resp.json()

        parts = []

        # Instant Answer
        answer = data.get("Answer")
        if answer:
            parts.append(f"[Instant Answer] {_clean_html(answer)}")

        # Abstract
        abstract = data.get("AbstractText")
        if abstract:
            parts.append(f"[Abstract] {_clean_html(abstract)}")

        # Related Topics
        related = data.get("RelatedTopics", [])
        for topic in related[:5]:
            if isinstance(topic, dict):
                text = topic.get("Text", "")
                if text:
                    parts.append(f"- {_clean_html(text)}")

        if not parts:
            return f"No web results found for: {query}"

        return "\n\n".join(parts)

    except requests.RequestException as e:
        return f"Web search failed: {e}"
    except Exception as e:
        return f"Web search error: {e}"
