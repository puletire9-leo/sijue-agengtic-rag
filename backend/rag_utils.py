import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import os
import json
import time
import threading
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from metrics import RAG_RETRIEVAL_COUNT, RAG_RETRIEVAL_LATENCY
from milvus_client import MilvusManager
from embedding import embedding_service as _embedding_service
from parent_chunk_store import ParentChunkStore
from langchain.chat_models import init_chat_model

load_dotenv()

ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "3"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

# ── Rerank circuit breaker ──
_RERANK_FAILURE_COUNT = 0
_RERANK_CIRCUIT_OPEN_UNTIL = 0
_RERANK_LOCK = threading.Lock()
RERANK_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", "5"))
RERANK_CIRCUIT_THRESHOLD = 3
RERANK_CIRCUIT_COOLDOWN = 60


def _is_rerank_circuit_open():
    global _RERANK_CIRCUIT_OPEN_UNTIL
    with _RERANK_LOCK:
        if time.time() < _RERANK_CIRCUIT_OPEN_UNTIL:
            return True
    return False


def _record_rerank_failure():
    global _RERANK_FAILURE_COUNT, _RERANK_CIRCUIT_OPEN_UNTIL
    with _RERANK_LOCK:
        _RERANK_FAILURE_COUNT += 1
        if _RERANK_FAILURE_COUNT >= RERANK_CIRCUIT_THRESHOLD:
            _RERANK_CIRCUIT_OPEN_UNTIL = time.time() + RERANK_CIRCUIT_COOLDOWN
            _RERANK_FAILURE_COUNT = 0


def _record_rerank_success():
    global _RERANK_FAILURE_COUNT
    with _RERANK_LOCK:
        _RERANK_FAILURE_COUNT = 0

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()

_stepback_model = None


def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    # Pre-compute max child score per parent_id so the best score is preserved
    max_child_score: Dict[str, float] = {}
    for parent_id, children in groups.items():
        if parent_id not in parent_map:
            continue
        for child in children:
            s = child.get("score")
            if s is not None:
                s = float(s)
                if parent_id not in max_child_score or s > max_child_score[parent_id]:
                    max_child_score[parent_id] = s

    merged_docs: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        if parent_id in max_child_score:
            parent_doc["score"] = max(
                float(parent_doc.get("score", max_child_score[parent_id])),
                max_child_score[parent_id],
            )
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    deduped: List[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    if not AUTO_MERGE_ENABLED or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    # 两段自动合并：L3->L2，再 L2->L1。
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    merged_docs = merged_docs[:top_k]

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
    }


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }
    if not docs_with_rank or not meta["rerank_enabled"]:
        return docs_with_rank[:top_k], meta

    # ── Circuit breaker: skip rerank if too many recent failures ──
    if _is_rerank_circuit_open():
        meta["rerank_error"] = "circuit_breaker_open"
        logger.warning("Rerank circuit breaker is open, skipping rerank")
        return docs_with_rank[:top_k], meta

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [doc.get("text", "") for doc in docs_with_rank],
        "top_n": min(top_k, len(docs_with_rank)),
        "return_documents": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }
    try:
        meta["rerank_applied"] = True
        response = requests.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=RERANK_TIMEOUT,
        )
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
            _record_rerank_failure()
            return docs_with_rank[:top_k], meta

        items = response.json().get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_with_rank):
                doc = dict(docs_with_rank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            _record_rerank_success()
            return reranked[:top_k], meta

        meta["rerank_error"] = "empty_rerank_results"
        _record_rerank_failure()
        return docs_with_rank[:top_k], meta
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = str(e)
        _record_rerank_failure()
        return docs_with_rank[:top_k], meta


def _get_stepback_model():
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请将用户的具体问题改写为一个更高层次、不同角度的搜索查询，"
        "用于在知识库中检索相关的通用原理或背景知识。\n"
        "要求：\n"
        "- 不要包含原查询中的专有名词和具体术语\n"
        "- 使用不同的措辞和表达方式\n"
        "- 输出一句简短的搜索查询，不要解释\n\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    if step_back_question:
        expanded_query = (
            f"{step_back_question}\n"
            f"（原始问题背景：{query}）"
        )
        if step_back_answer:
            expanded_query += f"\n参考背景：{step_back_answer}"
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def retrieve_documents(
    query: str,
    top_k: int = 5,
    strategy: str = "hybrid",
    skip_rerank: bool = False,
    extra_filter: str = "",
    kb_ids: list[str] | None = None,
) -> Dict[str, Any]:
    """混合/单路检索，v3: 支持策略选择和跳过 rerank。

    Args:
        query: 查询文本
        top_k: 返回数量
        strategy: 检索策略 (hybrid / dense_only / sparse_only / entity_boosted)
        skip_rerank: True 时跳过 rerank（由 graph 的 progressive_rerank 节点处理）
        extra_filter: 额外 Milvus 过滤表达式（如按章节 root_chunk_id 过滤），
                      与 chunk_level 条件 AND 拼接
        kb_ids: 可访问的知识库 ID 列表，为空则搜索所有

    Returns:
        {"docs": [...], "meta": {...}}
    """
    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    if kb_ids:
        quoted = ", ".join(f'"{kid}"' for kid in kb_ids)
        filter_expr += f" && kb_id in [{quoted}]"
    if extra_filter:
        filter_expr += f" && {extra_filter}"
    rerank_meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_k": candidate_k,
        "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
    }

    # ── 按策略生成所需 embedding ──
    needs_dense = strategy in ("hybrid", "dense_only", "entity_boosted")
    needs_sparse = strategy in ("hybrid", "sparse_only", "entity_boosted")

    dense_embedding = None
    sparse_embedding = None

    try:
        if needs_dense:
            dense_embedding = _embedding_service.get_embeddings([query])[0]
        if needs_sparse:
            sparse_embedding = _embedding_service.get_sparse_embedding(query)

        # ── 按策略路由到不同检索方法 ──
        if strategy == "sparse_only":
            rerank_meta["retrieval_mode"] = "sparse_only"
            if sparse_embedding is None:
                raise ValueError("sparse_only strategy requires sparse embedding")
            retrieved = _milvus_manager.sparse_retrieve(
                sparse_embedding=sparse_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
        elif strategy == "dense_only":
            rerank_meta["retrieval_mode"] = "dense_only"
            if dense_embedding is None:
                raise ValueError("dense_only strategy requires dense embedding")
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
        else:  # hybrid / entity_boosted
            rerank_meta["retrieval_mode"] = strategy
            retrieved = _milvus_manager.hybrid_retrieve(
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )

        # ── Rerank（可跳过，由 graph 的 progressive_rerank 处理）──
        if skip_rerank:
            # 保留原始检索分数，后续由 progressive_rerank 精排
            docs_for_merge = [{**d, "rrf_rank": i} for i, d in enumerate(retrieved, 1)]
        else:
            docs_for_merge, rr_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            rerank_meta.update(rr_meta)

        merged_docs, merge_meta = _auto_merge_documents(docs=docs_for_merge, top_k=top_k)
        rerank_meta.update(merge_meta)
        RAG_RETRIEVAL_COUNT.labels(strategy=strategy, result="success").inc()
        return {"docs": merged_docs, "meta": rerank_meta}

    except Exception as e:
        logger.warning(f"Retrieval strategy {strategy} failed: {e}")
        # 降级：hybrid → dense_only → sparse_only 依次回退
        if strategy == "dense_only":
            # dense_only already failed; skip straight to sparse_only
            fallback_order = [("sparse_only", needs_sparse, True, sparse_embedding)]
        elif strategy in ("hybrid", "entity_boosted"):
            fallback_order = [
                ("dense_only", needs_dense, False, dense_embedding),
                ("sparse_only", needs_sparse, True, sparse_embedding),
            ]
        else:  # sparse_only
            fallback_order = [("dense_only", False, True, None)]

        for fb_strategy, already_generated, needs_gen, emb in fallback_order:
            try:
                if already_generated and emb is None:
                    continue
                if needs_gen and emb is None:
                    if fb_strategy == "dense_only":
                        emb = _embedding_service.get_embeddings([query])[0]
                        retrieved = _milvus_manager.dense_retrieve(
                            dense_embedding=emb, top_k=candidate_k, filter_expr=filter_expr,
                        )
                    else:
                        continue
                elif fb_strategy == "dense_only":
                    retrieved = _milvus_manager.dense_retrieve(
                        dense_embedding=emb, top_k=candidate_k, filter_expr=filter_expr,
                    )
                else:  # sparse_only fallback
                    if emb is None:
                        emb = _embedding_service.get_sparse_embedding(query)
                    retrieved = _milvus_manager.sparse_retrieve(
                        sparse_embedding=emb, top_k=candidate_k, filter_expr=filter_expr,
                    )

                if not skip_rerank:
                    docs_for_merge, rr_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
                    rerank_meta.update(rr_meta)
                else:
                    docs_for_merge = [{**d, "rrf_rank": i} for i, d in enumerate(retrieved, 1)]

                merged_docs, merge_meta = _auto_merge_documents(docs=docs_for_merge, top_k=top_k)
                rerank_meta["retrieval_mode"] = f"{fb_strategy}_fallback"
                rerank_meta.update(merge_meta)
                RAG_RETRIEVAL_COUNT.labels(strategy=fb_strategy, result="fallback").inc()
                return {"docs": merged_docs, "meta": rerank_meta}
            except Exception as e:
                logger.warning(f"Retrieval strategy {fb_strategy} failed: {e}")
                continue

        RAG_RETRIEVAL_COUNT.labels(strategy=strategy, result="empty").inc()
        return {
            "docs": [],
            "meta": {
                **rerank_meta,
                "retrieval_mode": "failed",
                "rerank_error": "retrieve_failed",
                "auto_merge_applied": False,
                "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                "auto_merge_replaced_chunks": 0,
                "auto_merge_steps": 0,
                "candidate_count": 0,
            },
        }
