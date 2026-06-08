"""Agent 预算管理器 — 协调父子 Agent 间的预算分配。

参考: Hermes Agent delegation.max_iterations 配置

设计说明:
- 父 Agent 创建总预算池
- 子 Agent 可选择继承父预算 (共享) 或创建子预算 (隔离)
- 支持配置化的子 Agent 预算上限
"""

from typing import Dict, Any, Optional

from core.iteration_budget import IterationBudget


class BudgetManager:
    """Agent 预算管理器。"""

    DEFAULT_MAX_ITERATIONS = 90
    DEFAULT_SUBAGENT_MAX_ITERATIONS = 50

    def __init__(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        parent_budget: Optional[IterationBudget] = None,
    ):
        if parent_budget is not None:
            self.iteration_budget = parent_budget
            self._is_shared = True
        else:
            self.iteration_budget = IterationBudget(max_iterations)
            self._is_shared = False
        self.max_iterations = max_iterations

    def create_child_budget(
        self,
        max_iterations: Optional[int] = None,
        share_with_parent: bool = False,
    ) -> "BudgetManager":
        """为子 Agent 创建预算管理器。

        Args:
            max_iterations: 子 Agent 的最大迭代数 (默认 50)
            share_with_parent: 是否与父 Agent 共享预算

        Returns:
            新的 BudgetManager 实例
        """
        child_max = max_iterations or self.max_iterations or self.DEFAULT_SUBAGENT_MAX_ITERATIONS
        if share_with_parent:
            return BudgetManager(
                max_iterations=child_max,
                parent_budget=self.iteration_budget,
            )
        return BudgetManager(max_iterations=child_max)

    def consume(self) -> bool:
        return self.iteration_budget.consume()

    def refund(self) -> None:
        self.iteration_budget.refund()

    @property
    def used(self) -> int:
        return self.iteration_budget.used

    @property
    def remaining(self) -> int:
        return self.iteration_budget.remaining

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_shared": self._is_shared,
            "max_iterations": self.max_iterations,
            **self.iteration_budget.get_stats(),
        }
