"""ProgressiveRerank — 多阶段渐进式重排节点。

阶段:
  Stage 1: Score Normalize  — 将不同检索通道的分数归一化到 [0,1]
  Stage 2: Cross-Encoder    — 调用 Rerank API 精排（复用 rag_utils._rerank_documents）
  Stage 3: Context-Aware    — 父块上下文感知调整（关联父块的子块相互提权）

该节点位于 hybrid_retrieve 和 grade_documents 之间。
"""

import copy
from typing import Dict, List, Tuple

from agentic_rag.state import AgenticRAGState
from agentic_rag.config import retrieval as retrieval_cfg
from events import emit_rag_step


# ═══════════════════════════════════════════════════════════════
# Stage 1: Score Normalization
# ═══════════════════════════════════════════════════════════════

def _minmax_normalize(scores: List[float]) -> List[float]:
    """Min-Max 归一化到 [0, 1]."""
    if not scores:
        return scores
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [0.5] * len(scores)  # 全部相同 → 中位
    return [(s - mn) / (mx - mn) for s in scores]


def _normalize_stage(docs: List[dict]) -> List[dict]:
    """Stage 1: 归一化 RRF 分数到 [0, 1]。

    Milvus 返回单一 RRF score，不区分 bm25_score / dense_score。
    """
    if not docs:
        return docs

    main_scores = [float(d.get("score", 0) or 0) for d in docs]
    normed_main = _minmax_normalize(main_scores)

    for i, doc in enumerate(docs):
        doc["_normed_score"] = round(normed_main[i], 4)

    return docs


# ═══════════════════════════════════════════════════════════════
# Stage 2: Cross-Encoder Rerank
# ═══════════════════════════════════════════════════════════════

def _cross_encoder_stage(
    query: str,
    docs: List[dict],
    top_k: int,
) -> Tuple[List[dict], dict]:
    """Stage 3: Cross-Encoder 精排。复用 rag_utils._rerank_documents。"""
    meta: dict = {
        "rerank_applied": False,
        "rerank_model": None,
        "rerank_error": None,
        "input_count": len(docs),
    }

    if not docs:
        return docs, meta

    try:
        from rag_utils import _rerank_documents
        reranked, rerank_meta = _rerank_documents(query=query, docs=docs, top_k=top_k)
        meta.update(rerank_meta)

        # 保留内部分数
        for doc in reranked:
            doc["_post_rerank_score"] = doc.get("rerank_score", doc.get("_normed_score", 0))

        return reranked, meta
    except ImportError:
        meta["rerank_error"] = "rag_utils._rerank_documents not available"
        return docs, meta
    except Exception as e:
        meta["rerank_error"] = str(e)
        return docs, meta


# ═══════════════════════════════════════════════════════════════
# Stage 3: Context-Aware Adjustment
# ═══════════════════════════════════════════════════════════════

def _context_aware_stage(docs: List[dict], top_k: int) -> Tuple[List[dict], dict]:
    """Stage 4: 父块上下文感知调整。

    同属一个父块的多个子块被检索到时，轻微加权（表明该区域语义集中）。
    """
    meta: dict = {"applied": False, "boosted_chunks": 0, "parent_groups": 0}

    if not docs or len(docs) < 2:
        return docs, meta

    # 按 parent_chunk_id 分组
    groups: Dict[str, List[int]] = {}
    for i, doc in enumerate(docs):
        pid = doc.get("parent_chunk_id")
        if pid:
            groups.setdefault(pid, []).append(i)

    # 同一父块下 ≥2 个 chunk 命中 → 互相轻微提权
    for pid, indices in groups.items():
        if len(indices) >= 2:
            meta["parent_groups"] += 1
            for idx in indices:
                current = float(docs[idx].get("_post_rerank_score",
                                              docs[idx].get("_normed_score", 0)))
                docs[idx]["_context_boost"] = 0.05 * (len(indices) - 1)
                docs[idx]["_post_rerank_score"] = round(
                    current + docs[idx]["_context_boost"], 4
                )
                meta["boosted_chunks"] += 1

    if meta["boosted_chunks"] > 0:
        docs.sort(key=lambda d: d.get("_post_rerank_score", 0), reverse=True)
        meta["applied"] = True

    return docs[:top_k], meta


# ═══════════════════════════════════════════════════════════════
# 主节点
# ═══════════════════════════════════════════════════════════════

def progressive_rerank(state: AgenticRAGState) -> dict:
    """多阶段渐进式重排。

    流程: Normalize → Cross-Encoder → Context-Aware
    """
    query = state.get("query", "") or state.get("question", "")
    docs = copy.deepcopy(state.get("retrieved_docs", []))
    plan = state.get("retrieval_plan") or {}
    top_k = plan.get("top_k", retrieval_cfg.RETRIEVE_TOP_K)

    if not docs:
        emit_rag_step("⚠️", "无文档可重排", "")
        return {
            "reranked_docs": [],
            "rerank_trace": {"stages": []},
        }

    rerank_trace: dict = {"stages": [], "input_count": len(docs)}

    # Stage 1: 归一化
    docs = _normalize_stage(docs)
    rerank_trace["stages"].append("normalize")
    emit_rag_step("📏", "Stage 1/3 分数归一化", f"{len(docs)} 个文档")

    # Stage 2: Cross-Encoder 精排（在原始 top_k × candidate_multiplier 范围内）
    candidate_k = min(len(docs), top_k * plan.get("candidate_multiplier", 3))
    docs_to_rerank = docs[:candidate_k]
    docs_rest = docs[candidate_k:]

    docs_reranked, rerank_meta = _cross_encoder_stage(query, docs_to_rerank, top_k)
    rerank_trace.update(rerank_meta)
    rerank_trace["stages"].append("cross_encoder")

    if rerank_meta.get("rerank_applied"):
        emit_rag_step("🎯", f"Stage 2/3 Cross-Encoder 精排",
                      f"{rerank_meta['input_count']} → {len(docs_reranked)} "
                      f"(model: {rerank_meta.get('rerank_model', 'unknown')})")
    else:
        emit_rag_step("⚠️", "Stage 2/3 Cross-Encoder 跳过",
                      rerank_meta.get("rerank_error", "reranker unavailable"))

    # 合并回未参与精排的文档（如果 reranker 返回的结果少于原始）
    reranked_keys = {d.get("chunk_id") or hash(d.get("text", "")) for d in docs_reranked}
    remaining = [d for d in docs_rest if (d.get("chunk_id") or hash(d.get("text", ""))) not in reranked_keys]
    combined = docs_reranked + remaining

    # Stage 3: 上下文感知调整
    combined, context_meta = _context_aware_stage(combined, top_k)
    rerank_trace["stages"].append("context_aware")
    rerank_trace.update({"context_" + k: v for k, v in context_meta.items()})

    if context_meta.get("applied"):
        emit_rag_step("🔗", "Stage 3/3 上下文感知调整",
                      f"{context_meta['parent_groups']} 个父块组, "
                      f"{context_meta['boosted_chunks']} 个 chunk 提权")

    emit_rag_step("✅", f"重排完成 → Top {len(combined)}", "")

    return {
        "retrieved_docs": combined,
        "filtered_docs": combined,
        "reranked_docs": combined,
        "rerank_trace": rerank_trace,
    }
