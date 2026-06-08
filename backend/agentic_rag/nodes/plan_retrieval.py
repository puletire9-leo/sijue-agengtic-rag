"""PlanRetrieval — 检索参数规划节点。

根据选定的检索策略，配置具体的检索参数：
- top_k: 各阶段返回数量
- chunk_level: 检索粒度 L1/L2/L3
- metadata_filters: 元数据过滤
- use_hyde: 是否使用 HyDE 假想文档
- use_step_back: 是否使用 step-back 抽象化
- entity_boost_fields: 实体增强配置
"""

import re
from typing import Dict, List, Optional

from agentic_rag.state import AgenticRAGState
from agentic_rag.config import RetrievalConfig, BudgetConfig
from events import emit_rag_step


# ── 检索策略 → 默认参数映射 ──

STRATEGY_DEFAULTS = {
    "dense_only": {
        "top_k": 8,
        "candidate_multiplier": 3,
        "chunk_level": "L3",
        "dense_weight": 1.0,
        "sparse_weight": 0.0,
    },
    "sparse_only": {
        "top_k": 5,
        "candidate_multiplier": 2,
        "chunk_level": "L2",
        "dense_weight": 0.0,
        "sparse_weight": 1.0,
    },
    "entity_boosted": {
        "top_k": 6,
        "candidate_multiplier": 4,
        "chunk_level": "L3",
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "entity_boost_enabled": True,
        "entity_boost_factor": 1.15,
    },
    "hybrid": {
        "top_k": 5,
        "candidate_multiplier": 3,
        "chunk_level": "L3",
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
    },
}


def _extract_year_filter(question: str) -> Optional[int]:
    """从查询中提取年份过滤。"""
    m = re.search(r'(\d{4})\s*年', question)
    return int(m.group(1)) if m else None


def _extract_source_filter(question: str) -> Optional[str]:
    """从查询中提取来源过滤。"""
    markers = ["财报", "年报", "季报", "公告", "新闻稿", "白皮书", "手册", "SOP"]
    for marker in markers:
        if marker in question:
            return marker
    return None


def _detect_language(question: str) -> str:
    """检测查询语言以调整检索权重。"""
    en_chars = sum(1 for c in question if c.isascii() and c.isalpha())
    cn_chars = sum(1 for c in question if '一' <= c <= '鿿')
    return "en" if en_chars > cn_chars else "zh"


def _detect_query_complexity(question: str) -> str:
    """检测查询复杂度类型。返回: "simple" | "comparison" | "survey" | "standard" """
    comparison_markers = [
        "对比", "比较", "区别", "差异", "vs", "VS", "和",
        "相比", "优于", "领先", "哪个好", "哪种更好",
    ]
    survey_markers = [
        "综述", "总结", "概述", "全面", "系统", "所有",
        "各种", "多种", "不同", "分类", "类型",
    ]
    has_comparison = sum(1 for m in comparison_markers if m in question) >= 2
    has_survey = any(m in question for m in survey_markers)
    if has_comparison:
        return "comparison"
    if has_survey:
        return "survey"
    if len(question) > 60:
        return "survey"
    return "standard"


def plan_retrieval(state: AgenticRAGState) -> dict:
    """根据检索策略规划参数。

    输出 retrieval_plan dict，供 hybrid_retrieve 节点消费。
    """
    question = state.get("query", "") or state.get("question", "")
    strategy = state.get("retrieval_strategy", "hybrid")

    # ── 基础参数 ──
    defaults = STRATEGY_DEFAULTS.get(strategy, STRATEGY_DEFAULTS["hybrid"])
    plan: dict = {
        "top_k": defaults["top_k"],
        "candidate_multiplier": defaults["candidate_multiplier"],
        "chunk_level": defaults.get("chunk_level", "L3"),
        "dense_weight": defaults.get("dense_weight", 0.5),
        "sparse_weight": defaults.get("sparse_weight", 0.5),
        "use_hyde": defaults.get("use_hyde", False),
        "use_step_back": defaults.get("use_step_back", False),
        "entity_boost_enabled": defaults.get("entity_boost_enabled", False),
        "entity_boost_factor": defaults.get("entity_boost_factor", 1.0),
        "metadata_filters": {},
        "language": _detect_language(question),
    }

    # ── 动态调整 ──

    # 年份过滤
    year = _extract_year_filter(question)
    if year:
        plan["metadata_filters"]["year"] = year
        emit_rag_step("📅", f"年份过滤: {year}", "")

    # 来源过滤
    source = _extract_source_filter(question)
    if source:
        plan["metadata_filters"]["source_type"] = source
        emit_rag_step("📂", f"来源过滤: {source}", "")

    # 多实体查询 → 提高 top_k
    from agentic_rag.nodes.decide_retrieval import _extract_entities_heuristic
    entity_count = len(_extract_entities_heuristic(question))
    if entity_count >= 5:
        plan["top_k"] = min(plan["top_k"] + 3, 15)
        plan["candidate_multiplier"] = min(plan["candidate_multiplier"] + 1, 5)

    # 查询复杂度 → 动态 top_k
    complexity = _detect_query_complexity(question)
    if complexity == "comparison":
        plan["top_k"] = min(plan["top_k"] + 5, 15)
        plan["candidate_multiplier"] = min(plan.get("candidate_multiplier", 3) + 1, 5)
        emit_rag_step("📊", f"比较类查询，扩大检索: top_k={plan['top_k']}", "")
    elif complexity == "survey":
        plan["top_k"] = min(plan["top_k"] + 3, 12)
        plan["candidate_multiplier"] = min(plan.get("candidate_multiplier", 3) + 1, 5)
        emit_rag_step("📚", f"综述类查询，扩大检索: top_k={plan['top_k']}", "")

    # 超长查询 → 降低 HyDE 概率（已经是详细描述，不需要再生成假想答案）
    if len(question) > 100 and plan.get("use_hyde"):
        plan["use_hyde"] = False

    # 英文查询 → 提高稀疏权重（英文 BM25 效果更好）
    if plan["language"] == "en" and strategy == "hybrid":
        plan["sparse_weight"] = 0.55
        plan["dense_weight"] = 0.45

    # ── 预算约束：如果剩余迭代少，减少 top_k ──
    budget_dict = state.get("iteration_budget")
    if budget_dict and isinstance(budget_dict, dict):
        remaining = budget_dict.get("max", 30) - budget_dict.get("used", 0)
        if remaining <= 3:
            plan["top_k"] = max(3, plan["top_k"] - 2)
            plan["candidate_multiplier"] = max(1, plan["candidate_multiplier"] - 1)
            emit_rag_step("💰", f"预算紧张，缩减检索规模: top_k={plan['top_k']}", "")

    emit_rag_step(
        "📋", f"检索计划: {strategy}",
        f"top_k={plan['top_k']}, level={plan['chunk_level']}, "
        f"dense={plan['dense_weight']}, sparse={plan['sparse_weight']}"
    )

    return {"retrieval_plan": plan}
