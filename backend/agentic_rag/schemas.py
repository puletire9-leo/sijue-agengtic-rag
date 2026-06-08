"""Agentic RAG 核心数据模型。

v4.7 结构化对象：
- Budget: 迭代预算 value object（frozen dataclass，不可变语义）
- ConfidenceAssessment: 置信评估结构对象（Pydantic）
"""

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Budget:
    """
    迭代预算 value object（替代 v4.6 的裸字典 iteration_budget）。

    设计理由：
    - 裸字典跨节点传递，任何节点都可以随意修改 used 值
    - frozen=True 保证不可变，consume() 返回新对象
    - 所有预算逻辑收敛到此类，避免散落在各节点
    """
    max: int
    used: int
    grace: int = 1

    @property
    def remaining(self) -> int:
        return self.max - self.used

    @property
    def is_exhausted(self) -> bool:
        # Account for grace period — consistent with budget_check logic
        # which uses: budget.used >= budget.max + budget.grace
        return self.used >= self.max + self.grace

    def consume(self, cost: int = 1) -> "Budget":
        """消耗预算，返回新对象（不修改原状态）。"""
        return Budget(max=self.max, used=self.used + cost, grace=self.grace)

    def to_dict(self) -> dict:
        """序列化，用于 LangGraph State 存储（State 需要 JSON 可序列化）。"""
        return {"max": self.max, "used": self.used, "grace": self.grace}

    @classmethod
    def from_dict(cls, d: dict) -> "Budget":
        """反序列化。"""
        return cls(max=d["max"], used=d["used"], grace=d.get("grace", 1))

    def __repr__(self) -> str:
        return f"Budget(used={self.used}/{self.max})"


class ConfidenceAssessment(BaseModel):
    """
    置信评估结果（替代 v4.6 的 confidence + confidence_result 双字段）。

    设计理由：
    - v4.6 中 confidence(float) 和 confidence_result(str) 语义重叠
    - 两个独立字段需要手动保持同步，存在不一致风险
    - 合并为单一对象，保证原子性
    """
    score: float = Field(ge=0.0, le=1.0, description="置信分数 0~1")
    level: str = Field(description="决策级别: answer / retry / partial")
    reason: str = Field(description="决策理由，用于追踪和调试")
