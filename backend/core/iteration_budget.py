"""线程安全的迭代预算计数器 — 用于控制 LangChain Agent 的最大工具调用轮数。

参考: Hermes Agent v0.13.0 (run_agent.py:283-325)

区别于 agentic_rag.schemas.Budget:
  - IterationBudget: 线程安全(Lock)，可变状态，用于 Agent 层
  - Budget: frozen dataclass，不可变，用于 LangGraph State 层
"""

import threading
from typing import Optional


class IterationBudget:
    """线程安全的迭代预算计数器。

    核心特性:
    1. 每个 Agent 拥有独立的 IterationBudget 实例
    2. 父 Agent 的预算上限为 max_iterations (默认 90)
    3. 子 Agent 通过参数传入父的预算实例，实现共享
    4. 程序性工具调用可通过 refund() 返还预算

    使用示例:
        budget = IterationBudget(max_total=90)
        if not budget.consume():
            raise IterationBudgetExhausted("预算耗尽")
        budget.refund()  # 程序性调用返还
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """尝试消耗一次迭代。

        Returns:
            True: 消耗成功，仍有剩余预算
            False: 预算已耗尽
        """
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """返还一次迭代计数。

        使用场景:
        - 程序性工具调用（如代码执行）不应消耗预算
        - 工具执行失败需要重试时
        - 特定内部操作不计入用户可见迭代
        """
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)

    @property
    def is_exhausted(self) -> bool:
        with self._lock:
            return self._used >= self.max_total

    def get_stats(self) -> dict:
        """获取预算统计信息。"""
        with self._lock:
            used = self._used
            return {
                "max_total": self.max_total,
                "used": used,
                "remaining": max(0, self.max_total - used),
                "usage_percent": round(used / self.max_total * 100, 1) if self.max_total > 0 else 0,
            }

    def __repr__(self) -> str:
        return f"IterationBudget(used={self.used}/{self.max_total})"


class IterationBudgetExhausted(Exception):
    """迭代预算耗尽异常。"""

    def __init__(self, message: str = "迭代预算已耗尽", stats: Optional[dict] = None):
        super().__init__(message)
        self.stats = stats or {}
