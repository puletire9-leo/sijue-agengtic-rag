"""GenerateAnswer — 基于检索结果生成回答。"""

import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

from agentic_rag.state import AgenticRAGState
from agentic_rag.llm import get_llm
from events import emit_rag_step
from metrics import LLM_CALLS, LLM_LATENCY


SYSTEM_PROMPT = (
    "你是一个基于知识库的智能问答助手。\n"
    "请根据提供的检索文档回答问题。\n"
    "要求：\n"
    "1. 只使用检索文档中的信息，不要编造事实\n"
    "2. 如果文档信息不足以回答，请诚实说明\n"
    "3. 提及引用的文档来源（文件名），用括号注明即可，不要生成 markdown 链接\n"
    "4. 使用中文回答\n"
    "5. 绝对不要生成 [文件名](/path/...) 这样的链接，文件名直接用文字引用\n"
    "6. 数据精度匹配知识来源：如果知识库中没有具体数值，绝对不要编造精确数字（如毫秒、百分比）。"
    "  用定性描述替代（如「低延迟」「毫秒级」「可忽略」），并注明「知识库中无性能基准，以下为定性分析」\n"
    "7. 回答长度匹配信息量：如果知识库只提供架构设计没有性能数据，回答应简洁，"
    "  用一个汇总表即可，不要展开多个详细子表格和场景分析\n"
    "8. 推测必须明确标注：如果某部分回答是基于推理而非知识库原文，必须用【推测】标签明确标记\n"
)

ANSWER_PROMPT = (
    "用户问题：{question}\n\n"
    "检索到的相关文档：\n{context}\n\n"
    "请基于以上文档回答用户问题。\n"
    "如果文档中没有具体数值，用定性描述（如「低/中/高」「毫秒级」「秒级」）代替，不要编造数字。"
)


def _format_context(docs: list) -> str:
    """格式化检索文档为 LLM 上下文。"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        truncated = text[:2000] + ("..." if len(text) > 2000 else "")
        parts.append(f"[{i}] 来源: {source} (Page {page})\n{truncated}")
    return "\n\n---\n\n".join(parts)


def generate_answer(state: AgenticRAGState) -> dict:
    """基于检索文档生成最终回答。"""
    question = state.get("question", "")
    docs = state.get("reranked_docs") or state.get("filtered_docs") or state.get("retrieved_docs", [])

    # ── 语义缓存命中：直接使用缓存答案，跳过 LLM 调用 ──
    retrieval_meta = state.get("retrieval_metadata") or {}
    if retrieval_meta.get("cache_hit") and retrieval_meta.get("cached_answer"):
        cached_answer = retrieval_meta["cached_answer"]
        cached_citations = retrieval_meta.get("cached_citations", [])
        emit_rag_step("💾", "使用语义缓存答案", "")
        return {
            "answer": cached_answer,
            "citations": cached_citations,
            "messages": [AIMessage(content=cached_answer)],
        }

    if not docs:
        emit_rag_step("🤔", "无检索文档，基于自身知识回答")
        llm = get_llm()
        if llm:
            try:
                msg = HumanMessage(content=f"请回答：{question}")
                LLM_CALLS.labels(model="tier1", purpose="answer").inc()
                response = llm.invoke([msg])
                answer = response.content or ""
            except Exception as e:
                logger.warning("LLM invoke failed (no docs): %s", e)
                answer = f"无法回答：{question}（服务暂时不可用）"
        else:
            answer = f"无法回答：{question}（无知识库结果）"

        return {
            "answer": answer,
            "citations": [],
            "messages": [AIMessage(content=answer)],
        }

    context = _format_context(docs)

    # 提取引用来源
    citations = []
    seen_sources = set()
    for doc in docs:
        filename = doc.get("filename", "Unknown")
        if filename not in seen_sources:
            seen_sources.add(filename)
            citations.append({
                "source": filename,
                "page": doc.get("page_number", "N/A"),
                "score": doc.get("score", 0),
            })

    emit_rag_step("🤔", "正在评估文档并生成回答...")

    llm = get_llm()
    if not llm:
        source_lines = [f"- {c['source']} (Page {c['page']})" for c in citations]
        answer = (
            "系统已检索到以下相关文档，但当前无法生成回答：\n"
            + "\n".join(source_lines)
            + "\n\n请稍后重试，或联系管理员检查服务状态。"
        )
        return {
            "answer": answer,
            "citations": citations,
            "messages": [AIMessage(content=answer)],
        }

    try:
        # Use compressed messages if available, otherwise original messages
        conversation = state.get("compressed_messages") or state.get("messages", [])
        memory_context = state.get("memory_context")
        system_text = SYSTEM_PROMPT
        if memory_context:
            system_text += "\n\n" + memory_context
        messages = [
            SystemMessage(content=system_text),
        ]
        messages.extend(conversation)
        messages.append(
            HumanMessage(content=ANSWER_PROMPT.format(question=question, context=context)),
        )
        LLM_CALLS.labels(model="tier1", purpose="answer").inc()
        response = llm.invoke(messages)
        answer = response.content or ""

        # partial 级别：在回答前加免责声明
        if state.get("relevance_grade") == "partial":
            answer = (
                "ℹ️ **提示**: 知识库中仅有部分文档与本问题相关，以下回答可能不完整。\n\n"
                + answer
            )
    except Exception as e:
        source_lines = [f"- {c['source']} (Page {c['page']})" for c in citations]
        answer = (
            "系统已检索到以下相关文档，但回答生成过程中出现错误：\n"
            + "\n".join(source_lines)
            + "\n\n请稍后重试。"
        )

    # ── 语义缓存：将生成的答案存入缓存 ──
    try:
        from core.semantic_cache import semantic_cache
        from embedding import embedding_service

        if semantic_cache.enabled and answer and docs:
            query_vec = embedding_service.get_embeddings([question])[0]
            cache_meta = {
                "docs": [
                    {"filename": d.get("filename"), "text": d.get("text", "")[:200], "score": d.get("score")}
                    for d in docs[:5]
                ],
                "citations": citations,
            }
            semantic_cache.set(query_vec, answer, cache_meta)
    except Exception as e:
        logger.debug("Semantic cache set failed (non-fatal): %s", e)

    return {
        "answer": answer,
        "citations": citations,
        "messages": [AIMessage(content=answer)],
    }
