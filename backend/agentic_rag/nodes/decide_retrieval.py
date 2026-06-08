"""DecideRetrieval v3 — 检索策略决策节点。

根据查询特征选择最佳检索策略：
- hybrid: 默认，Dense + Sparse 融合
- dense_only: 纯语义搜索（自然语言、概念性问题）
- sparse_only: 纯关键词搜索（实体查询、精确匹配）
- entity_boosted: 实体锚点增强检索（命名实体密集的查询）

v3 变更：主路径使用轻量 LLM 做 NER + 意图分类（一次调用），正则规则作为降级。
"""

import json
import re
from typing import Optional

from agentic_rag.state import AgenticRAGState
from agentic_rag.llm import get_lightweight_llm
from events import emit_rag_step


# ── 策略枚举 ──
STRATEGY_HYBRID = "hybrid"
STRATEGY_DENSE_ONLY = "dense_only"
STRATEGY_SPARSE_ONLY = "sparse_only"
STRATEGY_ENTITY_BOOSTED = "entity_boosted"

ALL_STRATEGIES = [STRATEGY_HYBRID, STRATEGY_DENSE_ONLY, STRATEGY_SPARSE_ONLY, STRATEGY_ENTITY_BOOSTED]

# ── v3: 统一 LLM 分类 Prompt（NER + 意图分类 + 策略选择）──

QUERY_ANALYSIS_PROMPT = (
    "你是查询分析专家。分析用户问题，返回 JSON。\n\n"
    "## 检索策略说明\n"
    "- hybrid: 向量+关键词混合检索，适合大多数场景\n"
    "- dense_only: 纯语义向量检索，适合概念解释、原理描述型问题\n"
    "- sparse_only: 纯关键词检索，适合精确查找（具体的名称、日期、数据）\n"
    "- entity_boosted: 实体锚点增强，适合包含多个命名实体的问题\n\n"
    "## 查询类型\n"
    "- keyword_lookup: 精确查找具体信息（人名、日期、数字、地名）\n"
    "- semantic: 需要理解和解释的问题（如何、为什么、什么是）\n"
    "- entity_dense: 包含 3+ 个命名实体的查询\n"
    "- general: 一般性问题\n\n"
    "仅返回 JSON:\n"
    '{{"query_type": "keyword_lookup/semantic/entity_dense/general",'
    ' "entities": ["实体1", "实体2"],'
    ' "strategy": "hybrid/dense_only/sparse_only/entity_boosted",'
    ' "confidence": 0~1,'
    ' "reason": "简短理由"}}\n\n'
    "查询: {question}\n"
    "JSON:"
)


# ── v3: LLM 分析（主路径）──

def _llm_analyze_query(question: str) -> Optional[dict]:
    """使用轻量 LLM 分析查询特征，返回策略决策。"""
    llm = get_lightweight_llm()
    if not llm:
        return None

    try:
        response = llm.invoke(QUERY_ANALYSIS_PROMPT.format(question=question))
        content = response.content.strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(content)

        strategy = parsed.get("strategy", STRATEGY_HYBRID)
        if strategy not in ALL_STRATEGIES:
            strategy = STRATEGY_HYBRID

        return {
            "strategy": strategy,
            "confidence": float(parsed.get("confidence", 0.5)),
            "reason": str(parsed.get("reason", "llm_analysis")),
            "query_type": str(parsed.get("query_type", "general")),
            "entities": list(parsed.get("entities", [])),
            "method": "llm",
        }
    except Exception:
        return None


# ── 降级：正则启发式 ──

def _extract_entities_heuristic(text: str) -> list:
    """提取命名实体特征（降级用）。"""
    entities = []
    cn_entity_patterns = [
        r'(?:公司|集团|有限|股份|科技|银行|保险|证券|基金)',
        r'(?:[一-鿿]{1,4}(?:省|市|区|县|镇|村))',
        r'(?:第[一二三四五六七八九十\d]+(?:季度|期|批|阶段))',
        r'(?:\d{4}年\d{1,2}月\d{1,2}日?)',
        r'(?:\d{4}年)',
        r'(?:Q[1-4])',
        r'(?:[\d]+(?:\.\d+)?(?:%|亿|万|千|元|美元|欧元))',
    ]
    for pattern in cn_entity_patterns:
        entities.extend(re.findall(pattern, text))
    en_entity_patterns = [
        r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b',
        r'\b[A-Z]{2,8}\b',
    ]
    for pattern in en_entity_patterns:
        entities.extend(re.findall(pattern, text))
    return entities


def _is_keyword_query(question: str) -> bool:
    """判断是否为关键词型查询。"""
    if len(question) < 20:
        entities = _extract_entities_heuristic(question)
        return len(entities) >= 2
    num_count = len(re.findall(r'\d+(?:\.\d+)?', question))
    return num_count > 3 and len(question) < 50


def _is_semantic_query(question: str) -> bool:
    """判断是否为语义型查询。"""
    semantic_markers = [
        r'如何', r'怎么', r'为什么', r'什么(?:是|叫|意思)',
        r'解释', r'定义', r'概念', r'原理', r'过程', r'步骤',
        r'how\s+to', r'what\s+is', r'explain', r'describe',
    ]
    for pattern in semantic_markers:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    return len(question) > 50 and _extract_entities_heuristic(question) == []


def _has_many_entities(question: str) -> bool:
    """是否有 ≥3 个命名实体。"""
    return len(_extract_entities_heuristic(question)) >= 3


def _has_any_entities(question: str) -> bool:
    """是否有至少 1 个命名实体。"""
    return len(_extract_entities_heuristic(question)) >= 1


def _heuristic_decide(question: str) -> dict:
    """正则降级路径：启发式规则选择策略。"""
    if _is_keyword_query(question):
        return {
            "strategy": STRATEGY_SPARSE_ONLY,
            "confidence": 0.85,
            "reason": "keyword_query_heuristic",
            "method": "regex",
        }

    if _has_many_entities(question):
        return {
            "strategy": STRATEGY_ENTITY_BOOSTED,
            "confidence": 0.85,
            "reason": "multi_entity_heuristic",
            "method": "regex",
        }

    # 含单个实体 + 语义查询 → hybrid（兼顾关键词匹配和语义理解）
    if _has_any_entities(question) and _is_semantic_query(question):
        return {
            "strategy": STRATEGY_HYBRID,
            "confidence": 0.85,
            "reason": "entity_semantic_hybrid_heuristic",
            "method": "regex",
        }

    if _is_semantic_query(question):
        return {
            "strategy": STRATEGY_DENSE_ONLY,
            "confidence": 0.8,
            "reason": "semantic_query_heuristic",
            "method": "regex",
        }

    return {
        "strategy": STRATEGY_HYBRID,
        "confidence": 0.7,
        "reason": "default_heuristic",
        "method": "regex",
    }


# ═══════════════════════════════════════════════════════════════
# 主节点
# ═══════════════════════════════════════════════════════════════

def decide_retrieval(state: AgenticRAGState) -> dict:
    """v3: 主路径 LLM 分析，降级路径正则启发式。

    根据查询特征输出 retrieval_strategy，供 plan_retrieval / hybrid_retrieve 消费。
    """
    question = state.get("query", "") or state.get("question", "")

    if not question.strip():
        return {
            "retrieval_strategy": STRATEGY_HYBRID,
            "retrieval_strategy_reason": "empty_query_default",
            "retrieval_strategy_confidence": 1.0,
        }

    # ── v3 主路径: LLM NER + 意图分类 ──
    llm_result = _llm_analyze_query(question)
    if llm_result:
        emit_rag_step("🧭", f"检索策略: {llm_result['strategy']}",
                      f"type={llm_result.get('query_type')}, "
                      f"entities={llm_result.get('entities', [])[:3]}, "
                      f"method=llm, conf={llm_result['confidence']:.2f}")
        return {
            "retrieval_strategy": llm_result["strategy"],
            "retrieval_strategy_reason": f"llm:{llm_result['reason']}",
            "retrieval_strategy_confidence": llm_result["confidence"],
        }

    # ── 降级: 正则启发式 ──
    result = _heuristic_decide(question)
    emit_rag_step("🧭", f"检索策略: {result['strategy']} (降级)",
                  f"reason={result['reason']}, method=regex")
    return {
        "retrieval_strategy": result["strategy"],
        "retrieval_strategy_reason": f"regex:{result['reason']}",
        "retrieval_strategy_confidence": result["confidence"],
    }
