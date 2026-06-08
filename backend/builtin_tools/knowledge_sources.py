"""知识库源文档工具 — 让用户了解知识库中有哪些文档。"""

import threading
from langchain_core.tools import tool

# Module-level lazy MilvusManager singleton — avoids creating a new connection per tool call.
_milvus_manager = None
_milvus_manager_lock = threading.Lock()


def _get_milvus_manager():
    global _milvus_manager
    if _milvus_manager is None:
        with _milvus_manager_lock:
            if _milvus_manager is None:
                from milvus_client import MilvusManager
                _milvus_manager = MilvusManager()
                _milvus_manager.init_collection()
    return _milvus_manager


def _sanitize_milvus_string(value: str) -> str:
    """Escape special characters for Milvus filter expressions."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


@tool("list_knowledge_sources")
def list_knowledge_sources(query: str = "") -> str:
    """List all documents currently available in the knowledge base.

    Use this when the user asks:
    - "What documents are available?"
    - "What can I ask about?"
    - "Do you have the product manual?"

    Args:
        query: Optional filter — only show documents matching this keyword

    Returns:
        List of available documents with type and chunk count
    """
    try:
        milvus = _get_milvus_manager()

        results = milvus.query(
            output_fields=["filename", "file_type"],
            limit=10000,
        )

        file_stats = {}
        for item in results:
            fname = item.get("filename", "")
            ftype = item.get("file_type", "")
            if fname not in file_stats:
                file_stats[fname] = {"filename": fname, "file_type": ftype, "chunks": 0}
            file_stats[fname]["chunks"] += 1

        docs = list(file_stats.values())

        if query:
            docs = [d for d in docs if query.lower() in d["filename"].lower()]

        if not docs:
            return "知识库中没有匹配的文档。" if query else "知识库中暂无文档。请管理员上传。"

        lines = [f"知识库共有 {len(docs)} 个文档:" if not query else f"匹配 '{query}' 的文档:"]
        for d in docs:
            icon = {"pdf": "📄", "docx": "📝", "doc": "📝", "xlsx": "📊", "xls": "📊"}.get(d["file_type"].lower(), "📁")
            lines.append(f"  {icon} {d['filename']} ({d['file_type']}, {d['chunks']} 片段)")
        return "\n".join(lines)
    except Exception as e:
        return f"获取文档列表失败: {e}"


@tool("get_document_summary")
def get_document_summary(filename: str) -> str:
    """Get a content summary of a specific document in the knowledge base.

    Use this when the user asks about a specific document:
    - "What's in report.pdf?"
    - "Give me an overview of the employee handbook"

    Args:
        filename: The exact filename to summarize

    Returns:
        Document summary with page count and content preview
    """
    try:
        milvus = _get_milvus_manager()

        # 获取 L1 级别块（最大粒度，适合摘要）
        results = milvus.query(
            filter_expr=f'filename == "{_sanitize_milvus_string(filename)}" && chunk_level == 1',
            output_fields=["text", "page_number", "chunk_idx"],
            limit=20,
        )

        if not results:
            # 降级：取任意级别的块
            results = milvus.query(
                filter_expr=f'filename == "{_sanitize_milvus_string(filename)}"',
                output_fields=["text", "page_number", "chunk_level"],
                limit=10,
            )

        if not results:
            return f"知识库中未找到文档: {filename}"

        pages = sorted(set(r.get("page_number", 0) for r in results if r.get("page_number")))

        lines = [
            f"📄 {filename}",
            f"页面范围: {min(pages)}-{max(pages)} 页" if pages else "",
            "",
            "内容摘要:",
        ]

        for r in results[:5]:
            text = (r.get("text") or "")[:300]
            page = r.get("page_number", "?")
            lines.append(f"  [p.{page}] {text}...")

        if len(results) > 5:
            lines.append(f"  ... 还有 {len(results) - 5} 个片段")

        return "\n".join(lines)
    except Exception as e:
        return f"获取文档摘要失败: {e}"
