"""BudgetCheck — 迭代预算检查。

在每次检索重试前检查预算，避免无限循环。
"""

from agentic_rag.state import AgenticRAGState
from agentic_rag.schemas import Budget
from agentic_rag.config import budget as budget_cfg
from metrics import BUDGET_EXHAUSTIONS


def budget_check(state: AgenticRAGState) -> dict:
    """检查迭代预算和重试次数，决定是否允许继续。

    两层限制（任一触及即停止）：
    1. Budget 剩余次数（默认 5 次总迭代）
    2. retry_count 上限（默认 3 次改写重试，覆盖全部 3 种策略）
    """
    # Respect budget_exhausted already set by upstream nodes (e.g. rewrite_expand)
    if state.get("budget_exhausted"):
        return {"budget_exhausted": True}

    budget_dict = state.get("iteration_budget")
    if not budget_dict:
        budget = Budget(max=budget_cfg.MAX_ITERATIONS, used=0, grace=budget_cfg.GRACE_ITERATIONS)
    else:
        budget = Budget.from_dict(budget_dict)

    retry_count = state.get("retry_count", 0)
    max_retry = budget_cfg.MAX_RETRY_ITERATIONS

    # 重试次数已耗尽 → 逃生
    if retry_count >= max_retry:
        BUDGET_EXHAUSTIONS.inc()
        return {
            "budget_exhausted": True,
            "iteration_budget": budget.to_dict(),
        }

    # 预算耗尽 → 逃生（考虑 grace 容忍值）
    is_exhausted = budget.used >= budget.max + budget.grace
    if is_exhausted:
        BUDGET_EXHAUSTIONS.inc()
        return {
            "budget_exhausted": True,
            "iteration_budget": budget.to_dict(),
        }

    # 消耗一次预算
    new_budget = budget.consume(1)

    return {
        "budget_exhausted": False,
        "iteration_budget": new_budget.to_dict(),
    }
