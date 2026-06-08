"""HybridRetrieve v2 — 混合检索节点，v2 新增子问题合并检索 + 实体锚点 + 情节记忆提权 + 渐进式回填。

直接复用现有 rag_utils.retrieve_documents() 逻辑。
注意：rag_utils 导入是懒加载的，避免模块级触发 HF Hub 下载。
"""

import logging
import threading

logger = logging.getLogger(__name__)
from agentic_rag.state import AgenticRAGState
from agentic_rag.config import retrieval as retrieval_cfg
from events import emit_rag_step

# ── Sub-query concurrency control ──
_subquery_semaphore = threading.Semaphore(retrieval_cfg.MAX_SUBQUERY_CONCURRENCY)


def _limited_retrieve(query: str, **kwargs):
    """Retrieve with concurrency control via semaphore."""
    with _subquery_semaphore:
        from rag_utils import retrieve_documents
        return retrieve_documents(query, **kwargs)


def _sanitize_milvus_string(value: str) -> str:
    """Escape special characters for Milvus filter expressions."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_docs_brief(docs: list) -> str:
    """格式化文档列表为简短上下文。"""
    parts = []
    for i, doc in enumerate(docs[:10], 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        parts.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(parts)


def _apply_episodic_boosts(docs: list, query: str) -> list:
    """应用情节记忆加权。"""
    try:
        from core.episodic_memory import episodic_memory
        boosts = episodic_memory.get_boosts(query)
        if not boosts:
            return docs

        emit_rag_step("🧠", f"情节记忆匹配: {len(boosts)} 个文档", "")
        for doc in docs:
            fname = doc.get("filename", "")
            if fname in boosts:
                old_score = float(doc.get("score", 0) or 0)
                doc["score"] = old_score * boosts[fname]
                doc["episodic_boost"] = round(boosts[fname], 2)
        docs.sort(key=lambda d: float(d.get("score", 0) or 0), reverse=True)
    except Exception as e:
        logger.warning("episodic boost failed: %s", e)
    return docs


def _apply_entity_boosts(docs: list, query: str, boost_factor: float = 1.15) -> list:
    """应用实体锚点加权：查询中出现的实体如果在文档 chunk 中精确匹配，加分。"""
    try:
        from agentic_rag.nodes.decide_retrieval import _extract_entities_heuristic
        query_entities = set(_extract_entities_heuristic(query))

        if not query_entities:
            return docs

        boosted = 0
        for doc in docs:
            text = doc.get("text", "")
            matches = sum(1 for e in query_entities if e in text)
            if matches > 0:
                old_score = float(doc.get("score", 0) or 0)
                doc["score"] = old_score * (boost_factor ** matches)
                doc["entity_matches"] = matches
                boosted += 1

        if boosted:
            emit_rag_step("🏷️", f"实体锚点匹配: {boosted}/{len(docs)} 个片段",
                          f"boost_factor={boost_factor}")
            docs.sort(key=lambda d: float(d.get("score", 0) or 0), reverse=True)
    except Exception as e:
        logger.warning("entity boost failed: %s", e)
    return docs


def _apply_progressive_backfill(docs: list) -> tuple:
    """v2 渐进式回填。"""
    try:
        from agentic_rag.retrieval_v2 import progressive_backfill
        from parent_chunk_store import ParentChunkStore
        store = ParentChunkStore()
        new_docs, backfill_meta = progressive_backfill(docs, parent_store=store, top_k=len(docs))
        if backfill_meta.get("backfill_applied"):
            emit_rag_step("🧩", f"渐进式回填: {backfill_meta.get('backfill_count', 0)} 个父块",
                          f"最终 {backfill_meta.get('total_chunks', 0)} 个片段")
        return new_docs, backfill_meta
    except Exception as e:
        logger.warning("progressive backfill failed: %s", e)
        return docs, {"backfill_enabled": False}


def hybrid_retrieve(state: AgenticRAGState) -> dict:
    """执行混合检索，v2: 子问题合并 + 实体锚点 + 情节记忆 + 渐进回填。

    消费 retrieval_plan（由 plan_retrieval 节点产出）来动态调整检索参数。
    """
    from rag_utils import retrieve_documents

    query = state.get("query", "") or state.get("question", "")

    # ── 语义缓存检查 ──
    try:
        from core.semantic_cache import semantic_cache
        from embedding import embedding_service

        if semantic_cache.enabled:
            query_vec = embedding_service.get_embeddings([query])[0]
            cached = semantic_cache.get(query_vec)
            if cached:
                emit_rag_step("💾", "语义缓存命中", f"相似度: {cached['score']:.4f}")
                cached_meta = cached.get("metadata", {})
                return {
                    "retrieved_docs": cached_meta.get("docs", []),
                    "retrieval_metadata": {
                        "retrieval_mode": "semantic_cache",
                        "cache_hit": True,
                        "cache_score": cached["score"],
                        "cached_answer": cached.get("answer", ""),
                        "cached_citations": cached_meta.get("citations", []),
                    },
                }
    except Exception as e:
        logger.debug("Semantic cache check failed (non-fatal): %s", e)
    routed_chapters = state.get("routed_chapters") or []
    fallback_docs = state.get("route_fallback_docs") or []

    # ── 从 retrieval_plan 读取参数（fallback 到全局配置）──
    plan = state.get("retrieval_plan") or {}
    top_k = plan.get("top_k", retrieval_cfg.RETRIEVE_TOP_K)
    entity_boost_enabled = plan.get("entity_boost_enabled", False)
    entity_boost_factor = plan.get("entity_boost_factor", 1.15)
    retrieval_strategy = state.get("retrieval_strategy", "hybrid")

    # ── v2: 子问题分解检索 ──
    sub_queries = state.get("_sub_queries")
    if sub_queries and isinstance(sub_queries, list) and len(sub_queries) > 1:
        emit_rag_step("🔀", f"并行检索 {len(sub_queries)} 个子问题", "")
        from agentic_rag.retrieval_v2 import merge_sub_query_results

        # ── v4.8: 树导航章节过滤（子查询也适用）──
        sub_chapter_filter = ""
        if routed_chapters:
            ids_literal = ", ".join(f'"{_sanitize_milvus_string(rid)}"' for rid in routed_chapters)
            sub_chapter_filter = f'root_chunk_id in [{ids_literal}]'

        sub_results = []
        for sq in sub_queries:
            emit_rag_step("🔍", f"检索子问题: {sq[:50]}...", "")
            try:
                sub_results.append(_limited_retrieve(
                    sq, top_k=max(3, top_k // 2),
                    extra_filter=sub_chapter_filter,
                ))
            except Exception as e:
                import warnings
                warnings.warn(f"子问题检索失败: {sq[:50]}... — {e}", RuntimeWarning, stacklevel=2)
                emit_rag_step("⚠️", f"子问题检索失败: {sq[:50]}...", str(e)[:80])

        if not sub_results:
            docs = []
            meta = {"retrieval_mode": "hybrid_v2_multi_query_failed", "sub_queries": len(sub_queries)}
            emit_rag_step("⚠️", "所有子查询检索失败", "")
        else:
            docs = merge_sub_query_results(sub_results, top_k=top_k * 2)
            meta = {"retrieval_mode": "hybrid_v2_multi_query", "sub_queries": len(sub_queries),
                    "candidate_k": len(docs)}
    else:
        # ── 章节过滤 or 路由兜底文件过滤 ──
        chapter_filter = ""
        if routed_chapters:
            ids_literal = ", ".join(f'"{_sanitize_milvus_string(rid)}"' for rid in routed_chapters)
            chapter_filter = f'root_chunk_id in [{ids_literal}]'
        elif fallback_docs:
            names_literal = ", ".join(f'"{_sanitize_milvus_string(d)}"' for d in fallback_docs[:15])
            chapter_filter = f'filename in [{names_literal}]'

        filter_label = (
            f' (章节过滤: {len(routed_chapters)} 章)' if routed_chapters else
            f' (兜底过滤: {len(fallback_docs)} 文档)' if fallback_docs else ''
        )
        emit_rag_step("🔍", f"正在检索知识库... [{retrieval_strategy}]",
                      f"查询: {query[:60]}{filter_label}")
        try:
            result = retrieve_documents(
                query, top_k=top_k,
                strategy=retrieval_strategy,
                skip_rerank=True,  # 由 progressive_rerank 统一处理
                extra_filter=chapter_filter,
                kb_ids=state.get("kb_ids"),
            )
            docs = result.get("docs") or []
            meta = result.get("meta", {})
        except Exception as e:
            logger.warning("retrieve_documents failed: %s", e, exc_info=True)
            emit_rag_step("⚠️", f"检索异常: {e}", "回退空结果")
            docs = []
            meta = {}

        # ── 回退：章节过滤结果不足时，回退全局检索 ──
        if chapter_filter and len(docs) < 3:
            emit_rag_step("🌳", f"章节过滤仅得 {len(docs)} 条，回退全局检索", "")
            try:
                result = retrieve_documents(
                    query, top_k=top_k,
                    strategy=retrieval_strategy,
                    skip_rerank=True,
                    extra_filter="",  # 无过滤 = 全局
                    kb_ids=state.get("kb_ids"),
                )
                docs = result.get("docs", [])
                meta = result.get("meta", {})
            except Exception as e:
                logger.warning("fallback retrieve_documents failed: %s", e, exc_info=True)
                emit_rag_step("⚠️", f"回退检索异常: {e}", "")

    emit_rag_step("🧱", f"多粒度召回", f"候选 {len(docs)} 个片段")

    # ── v2: 实体锚点加权（由 retrieval_plan 控制）──
    if entity_boost_enabled:
        docs = _apply_entity_boosts(docs, query, boost_factor=entity_boost_factor)

    # ── v2: 情节记忆提权 ──
    docs = _apply_episodic_boosts(docs, query)

    # ── v2: 渐进式回填（替代旧的固定阈值 auto-merge）──
    backfill_meta = {}
    try:
        docs, backfill_meta = _apply_progressive_backfill(docs)
    except Exception as e:
        logger.warning("backfill call failed: %s", e)

    if meta.get("auto_merge_applied") and not backfill_meta.get("backfill_applied"):
        emit_rag_step("🧩", "Auto-merging 合并", f"替换片段: {meta.get('auto_merge_replaced_chunks', 0)}")

    # ── v4.9 P0: Topic 交集跨文档关联 ──
    if routed_chapters:
        try:
            from document_summary import doc_summary_manager
            current_filenames = set(d.get("filename", "") for d in docs)
            related = doc_summary_manager.get_related_docs_for_filenames(
                list(current_filenames), min_overlap=1
            )
            # 排除已检索到的文档
            new_docs = [d for d in related if d not in current_filenames]
            if new_docs:
                extra_filter = " || ".join(f'filename == "{_sanitize_milvus_string(d)}"' for d in new_docs[:3])
                extra_result = retrieve_documents(
                    query, top_k=max(2, top_k // 2),
                    strategy=retrieval_strategy,
                    skip_rerank=True,
                    extra_filter=extra_filter,
                    kb_ids=state.get("kb_ids"),
                )
                extra_chunks = extra_result.get("docs", [])
                if extra_chunks:
                    emit_rag_step("🔗", f"Topic 关联: {len(new_docs[:3])} 个文档，+{len(extra_chunks)} 片段",
                                  f"关联文档: {new_docs[:3]}")
                    # 标记来源，追加到结果末尾
                    for d in extra_chunks:
                        d["topic_linked"] = True
                    docs.extend(extra_chunks)
        except Exception as e:
            logger.warning("topic cross-doc association failed: %s", e)

    emit_rag_step("✅", f"检索完成，找到 {len(docs)} 个相关片段", f"模式: {meta.get('retrieval_mode', 'hybrid')}")

    # Note: doc count should not trigger message compression (semantic mismatch removed)

    # ── v2: 记录到情节记忆（通过 return dict 传递，不直接突变 state）──
    retrieved_filenames = list(set(d.get("filename", "") for d in docs if d.get("filename")))

    return {
        "retrieved_docs": docs,
        "filtered_docs": docs,
        "retrieval_metadata": {**meta, "backfill_v2": backfill_meta,
                               "strategy": retrieval_strategy, "plan_top_k": top_k},
        "_retrieved_filenames": retrieved_filenames,
    }
