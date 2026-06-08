"""检索增强 v2 — 子问题分解、实体提取、渐进式回填、上下文增强。

对应 v2 设计文档: docs/03-检索系统/更新查询v2.md
"""

import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from agentic_rag.llm import get_lightweight_llm
from agentic_rag.nodes.decide_retrieval import _extract_entities_heuristic


# ═══════════════════════════════════════════════════════════════
# 1. 子问题分解
# ═══════════════════════════════════════════════════════════════

_DECOMPOSE_PROMPT = (
    "你是一个查询分解器。判断用户问题是否复杂，如果是则拆成 2-4 个子问题。\n"
    "复杂问题指：跨多个主题、需要对比、包含多层嵌套条件、时间跨度大的问题。\n\n"
    "仅返回 JSON:\n"
    '{{"is_complex": true/false, "sub_questions": ["子问题1", "子问题2"], "reason": "判断理由"}}\n\n'
    "用户问题: {question}\n\n"
    "JSON:"
)


def is_complex_query(question: str) -> bool:
    """判断是否为复杂查询。"""
    if len(question) < 15:
        return False
    # Separate string markers from boolean conditions
    keyword_markers = ["对比", "比较", "区别", "分别", "哪些", "各自", "同时", "以及",
                       "vs"]
    keyword_count = sum(1 for kw in keyword_markers if kw in question.lower())
    has_question_mark = "?" in question
    return keyword_count + (1 if has_question_mark else 0) >= 2


def decompose_query(question: str) -> tuple[bool, List[str]]:
    """将复杂问题分解为子问题。

    Returns:
        (is_complex, [sub_questions])
    """
    if not is_complex_query(question):
        return False, [question]

    llm = get_lightweight_llm()
    if not llm:
        return False, [question]

    try:
        response = llm.invoke(_DECOMPOSE_PROMPT.format(question=question))
        text = response.content.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(text)

        if result.get("is_complex") and result.get("sub_questions"):
            subs = result["sub_questions"]
            if len(subs) > 1:
                return True, subs[:4]  # 最多4个子问题
    except Exception as e:
        logger.warning("query decomposition failed: %s", e)

    return False, [question]


# ═══════════════════════════════════════════════════════════════
# 2. 实体/关键词提取
# ═══════════════════════════════════════════════════════════════

_ENTITY_PROMPT = (
    "从以下文本中提取关键实体和专有名词（人名、地名、公司名、产品名、日期、数字、专业术语）。\n"
    "仅返回 JSON 数组: [\"实体1\", \"实体2\", ...]\n\n"
    "文本: {text}\n\n"
    "JSON:"
)


def extract_entities(texts: List[str], use_llm: bool = False) -> Dict[str, List[str]]:
    """从文本列表中提取实体。

    Args:
        texts: 文本列表（文档块内容）
        use_llm: 是否使用 LLM 提取（更准但更慢）

    Returns:
        {实体: [出现的文本索引列表]}
    """
    entity_map: Dict[str, List[int]] = defaultdict(list)

    for idx, text in enumerate(texts):
        if use_llm:
            llm = get_lightweight_llm()
            if llm:
                try:
                    response = llm.invoke(_ENTITY_PROMPT.format(text=text[:500]))
                    content = response.content.strip().removeprefix("```json").removesuffix("```").strip()
                    entities = json.loads(content)
                    if isinstance(entities, list):
                        for e in entities:
                            entity_map[e].append(idx)
                    continue
                except Exception as e:
                    logger.warning("entity extraction (LLM) failed: %s", e)

        # Fallback: 启发式提取
        for e in _extract_entities_heuristic(text):
            entity_map[e].append(idx)

    return dict(entity_map)


# ═══════════════════════════════════════════════════════════════
# 3. 渐进式回填 (Progressive Backfill)
# ═══════════════════════════════════════════════════════════════

def progressive_backfill(
    docs: List[dict],
    parent_store=None,
    top_k: int = 5,
    min_score: float = 0.4,
) -> tuple[List[dict], dict]:
    """渐进式回填：按分数和连续度决定回填哪些父块。

    与固定阈值 auto-merge 不同，这里:
    1. 高分 L3 块 → 回填 L2
    2. 多个相关 L2 → 回填 L1
    3. 低分块不触发回填
    4. 返回回填后的文档列表 + 回填元数据

    Args:
        docs: 检索到的文档列表 (L3)
        parent_store: ParentChunkStore 实例
        top_k: 最终返回 top-k
        min_score: 低于此分数的块不触发回填

    Returns:
        (回填后的文档列表, 回填元数据)
    """
    if not parent_store:
        return docs[:top_k], {"backfill_enabled": False, "reason": "no_parent_store"}

    # 按 parent_chunk_id 分组
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        pid = (doc.get("parent_chunk_id") or "").strip()
        if pid:
            groups[pid].append(doc)

    result: List[dict] = []
    backfill_count = 0
    seen_ids = set()

    for doc in docs:
        pid = (doc.get("parent_chunk_id") or "").strip()
        score = float(doc.get("score", 0) or 0)
        chunk_id = doc.get("chunk_id", "")

        if chunk_id in seen_ids:
            continue

        if not pid or score < min_score:
            # 低分或没有父块：保留原样
            seen_ids.add(chunk_id)
            result.append(doc)
            continue

        children = groups.get(pid, [])
        if len(children) < 2:
            seen_ids.add(chunk_id)
            result.append(doc)
            continue

        # 计算是否需要回填父块
        scores = [float(c.get("score", 0) or 0) for c in children]
        avg_score = sum(scores) / len(scores) if scores else 0

        # 只有平均分足够高才回填
        if avg_score >= min_score + 0.2:
            parent_docs = parent_store.get_documents_by_ids([pid])
            if parent_docs:
                parent_doc = dict(parent_docs[0])
                parent_doc["score"] = max(parent_doc.get("score", avg_score), avg_score)
                parent_doc["backfilled_from_children"] = True
                parent_doc["backfill_child_count"] = len(children)
                parent_doc["backfill_avg_score"] = round(avg_score, 3)

                if parent_doc.get("chunk_id") not in seen_ids:
                    seen_ids.add(parent_doc["chunk_id"])
                    result.append(parent_doc)
                    backfill_count += 1
                    # 标记所有子块为已处理
                    for c in children:
                        seen_ids.add(c.get("chunk_id", ""))
                    continue

        # 不回填：保留子块
        seen_ids.add(chunk_id)
        result.append(doc)

    # 按分数排序，取 top_k
    result.sort(key=lambda d: float(d.get("score", 0) or 0), reverse=True)
    result = result[:top_k]

    return result, {
        "backfill_enabled": True,
        "backfill_applied": backfill_count > 0,
        "backfill_count": backfill_count,
        "total_chunks": len(result),
    }


# ═══════════════════════════════════════════════════════════════
# 4. 上下文增强嵌入 (Contextual Retrieval)
# ═══════════════════════════════════════════════════════════════

_CONTEXT_GEN_PROMPT = (
    "为以下文档片段生成一句上下文描述，说明它在整个文档中的位置和背景。"
    "只输出描述句，不要解释。\n\n"
    "文档名: {filename}\n"
    "片段内容: {chunk_text}\n\n"
    "上下文描述:"
)


def generate_chunk_context(filename: str, chunk_text: str) -> Optional[str]:
    """为文档块生成上下文描述（Anthropic Contextual Retrieval 方案）。

    例如:
      原始块: "它同比增长了 15%"
      上下文: "该内容来自 2024 年财报，描述的是华东区营收"
    """
    llm = get_lightweight_llm()
    if not llm:
        return None

    try:
        response = llm.invoke(
            _CONTEXT_GEN_PROMPT.format(filename=filename, chunk_text=chunk_text[:300])
        )
        return response.content.strip()[:200]
    except Exception as e:
        logger.warning("chunk context gen failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
# 5. 摘要粗筛 (Summary Coarse Filtering)
# ═══════════════════════════════════════════════════════════════

_SUMMARY_PROMPT = (
    "为以下文档内容生成一句简洁摘要，概括文档的核心内容和用途。只输出摘要句。\n\n"
    "文档名: {filename}\n"
    "文档内容 (开头片段):\n{text}\n\n"
    "摘要:"
)


def generate_document_summary(filename: str, text: str) -> Optional[str]:
    """为文档生成一句摘要，用于粗筛层。"""
    llm = get_lightweight_llm()
    if not llm:
        return None

    try:
        response = llm.invoke(
            _SUMMARY_PROMPT.format(filename=filename, text=text[:1000])
        )
        return response.content.strip()[:200]
    except Exception as e:
        logger.warning("document summary gen failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
# 6. 合并多个子查询的检索结果
# ═══════════════════════════════════════════════════════════════

def merge_sub_query_results(
    sub_results: List[dict],
    top_k: int = 5,
    dedup_by: str = "chunk_id",
) -> List[dict]:
    """合并多个子查询的检索结果，去重并按分数排序。

    Args:
        sub_results: 每个子查询的检索结果 [{docs: [...], meta: {...}}, ...]
        top_k: 最终返回数量
        dedup_by: 去重字段

    Returns:
        合并去重后的文档列表
    """
    seen = set()
    merged = []

    for result in sub_results:
        docs = result.get("docs", []) if isinstance(result, dict) else (result if isinstance(result, list) else [])
        if isinstance(docs, dict):
            docs = docs.get("docs", [])
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            key = doc.get(dedup_by, "") or doc.get("text", "")[:50]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

    merged.sort(key=lambda d: float(d.get("score", 0) or 0), reverse=True)
    return merged[:top_k]
