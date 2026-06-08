"""RewriteExpand v3 — 查询扩展与重写，v3 消费 retrieval_plan 控制策略。

支持三种策略 + 子问题分解:
- step_back: 生成退步问题 + 回答
- hyde: 生成假设性文档
- complex: 同时使用两种策略 + 子问题分解

策略受 retrieval_plan 中 use_hyde / use_step_back 控制。
"""

import logging

from agentic_rag.state import AgenticRAGState
from agentic_rag.llm import get_lightweight_llm
from events import emit_rag_step

logger = logging.getLogger(__name__)


STRATEGY_PROMPT = (
    "请根据用户问题选择最合适的查询扩展策略，仅返回策略名。\n"
    "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
    "- hyde：模糊、概念性、需要解释或定义的问题。\n"
    "- complex：多步骤、需要分解或综合多种信息的复杂问题。\n"
    "用户问题：{question}"
)

# 所有可用策略（complex 在 step_back 和 hyde 都启用时才可用）
ALL_STRATEGIES = ["step_back", "hyde", "complex"]


def _is_simple_query(query: str) -> bool:
    """Detect simple queries that don't need expansion."""
    # Short queries (likely entity lookups)
    if len(query) <= 10:
        return True
    # Single entity names (no question words)
    question_words = {"什么", "怎么", "如何", "为什么", "哪个", "哪些", "多少", "是否", "能否",
                      "请", "帮我", "告诉", "介绍", "说明", "解释", "what", "how", "why", "which"}
    query_lower = query.lower()
    has_question_word = any(w in query_lower for w in question_words)
    if not has_question_word and len(query) <= 20:
        return True
    return False


def _get_allowed_strategies(plan: dict) -> list:
    """根据 retrieval_plan 返回允许的改写策略列表。重试时使用。"""
    if not plan:
        # 无 plan 时允许全部策略，让 LLM 根据问题特征选择
        return ["step_back", "hyde", "complex"]

    use_step_back = plan.get("use_step_back", True)
    use_hyde = plan.get("use_hyde", True)

    allowed = ["step_back"] if use_step_back else []
    if use_hyde:
        allowed.append("hyde")
    if use_step_back and use_hyde:
        allowed.append("complex")
    return allowed or ["step_back"]


def _choose_strategy(question: str, allowed: list) -> str:
    """从允许的策略中选择最佳策略。如果只允许一种，直接返回。"""
    if len(allowed) == 1:
        return allowed[0]

    llm = get_lightweight_llm()
    if not llm:
        return allowed[0]

    try:
        response = llm.invoke(STRATEGY_PROMPT.format(question=question))
        strategy = response.content.strip().lower()
        if strategy in allowed:
            return strategy
    except Exception as e:
        logger.warning("strategy choice LLM failed: %s", e)
    return allowed[0]


def rewrite_expand(state: AgenticRAGState) -> dict:
    """对查询进行扩展以改善检索效果，v3: 策略选择受 retrieval_plan 约束。"""
    from rag_utils import step_back_expand, generate_hypothetical_document

    question = state.get("question", "")
    query = state.get("query", question)
    retry_count = state.get("retry_count", 0)

    # 首轮不扩展：直接用原始 query 检索，避免 HyDE/step_back 带偏方向
    # NOTE (Bug #6): retry_count here and budget_check's Budget.used both increment
    # independently. retry_count drives strategy exhaustion (MAX_RETRY_ITERATIONS),
    # Budget.used drives the hard iteration ceiling (MAX_ITERATIONS + GRACE_ITERATIONS).
    # They are two separate safety nets; this is intentional — retry_count gates
    # rewrite quality, Budget gates total loop iterations.
    if retry_count == 0:
        return {
            "query": query,
            "rewrite_strategy": "none",
            "retry_count": 1,
            "_sub_queries": None,
            "tried_strategies": [],
        }

    # Skip expansion for simple queries (but still increment retry_count)
    if _is_simple_query(query) and retry_count > 0:
        return {
            "query": query,
            "rewrite_strategy": "none_simple",
            "retry_count": retry_count + 1,
            "budget_exhausted": True,
        }

    # ── v3: 从 retrieval_plan 获取策略约束 ──
    plan = state.get("retrieval_plan") or {}
    allowed_strategies = _get_allowed_strategies(plan)

    # L-05: 追踪已尝试的策略，避免重复尝试同一策略
    tried = list(state.get("tried_strategies") or [])
    current_strategy = state.get("rewrite_strategy")
    if current_strategy and current_strategy != "none" and current_strategy not in tried:
        tried.append(current_strategy)

    # 从未尝试过的策略中选择
    untried = [s for s in allowed_strategies if s not in tried]

    if not untried:
        # 所有策略已穷尽，标记预算耗尽
        emit_rag_step("⚠️", "所有改写策略已穷尽", f"已尝试: {tried}")
        return {
            "query": query,
            "rewrite_strategy": current_strategy or allowed_strategies[0],
            "retry_count": retry_count + 1,
            "_sub_queries": None,
            "tried_strategies": tried,
            "budget_exhausted": True,
        }

    strategy = untried[0]

    emit_rag_step("✏️", f"查询扩展 (策略: {strategy})", f"重试 #{retry_count}")

    expanded_query = query
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""

    # ── v2: 子问题分解 (仅 complex 时触发) ──
    sub_queries = None  # 默认清除，避免陈旧状态
    if strategy in ("complex",):
        from agentic_rag.retrieval_v2 import decompose_query
        is_complex, subs = decompose_query(query)
        if is_complex and len(subs) > 1:
            sub_queries = subs
            emit_rag_step("🔀", f"复杂问题分解为 {len(subs)} 个子问题", subs[0][:40])

    if strategy in ("step_back", "complex"):
        emit_rag_step("🧠", "Step-back 查询抽象", "")
        result = step_back_expand(query)
        step_back_question = result.get("step_back_question", "")
        step_back_answer = result.get("step_back_answer", "")
        expanded_query = result.get("expanded_query", query)

    if strategy in ("hyde", "complex"):
        emit_rag_step("📝", "HyDE 假设文档生成", "")
        hypothetical_doc = generate_hypothetical_document(query)

    return {
        "query": expanded_query,
        "rewrite_strategy": strategy,
        "retry_count": retry_count + 1,
        # v2: 始终显式设置 _sub_queries（None 表示不使用子问题）
        "_sub_queries": sub_queries if (sub_queries and len(sub_queries) > 1) else None,
        "tried_strategies": tried,
    }
