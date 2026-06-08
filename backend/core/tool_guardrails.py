"""Tool-call loop guardrails — 从 hermes-agent 照搬。

三种检测模式:
  1. exact_failure: 同一工具+相同参数连续失败
  2. same_tool_failure: 同一工具连续失败（参数不同）
  3. idempotent_no_progress: 只读工具返回相同结果（无进展）

设计原则: side-effect free controller, 由调用方决定如何响应决策。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Mapping


# 只读工具（重复调用无副作用，但可能陷入无进展循环）
IDEMPOTENT_TOOL_NAMES: FrozenSet[str] = frozenset({
    "search_knowledge_base",
    "get_current_weather",
    "read_file",
    "search_files",
    "web_search",
    "web_extract",
})

# 变更工具（每次调用都可能产生不同结果）
MUTATING_TOOL_NAMES: FrozenSet[str] = frozenset({
    "terminal",
    "execute_code",
    "write_file",
    "patch",
    "send_message",
})


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """每次对话 turn 的工具调用检测阈值。

    warnings_enabled: 警告从不阻止工具执行
    hard_stop_enabled: 硬停止需显式开启
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)


@dataclass(frozen=True)
class ToolCallSignature:
    """工具名 + 参数哈希的稳定标识。"""
    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        canonical = canonical_tool_args(args or {})
        return cls(tool_name=tool_name, args_hash=_sha256(canonical))


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """工具调用决策。"""
    action: str = "allow"  # allow | warn | block | halt
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: ToolCallSignature | None = None

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        return self.action in {"block", "halt"}


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """排序后压缩 JSON。"""
    return json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


class ToolCallGuardrailController:
    """每次对话 turn 的重复失败/无进展工具调用检测器。"""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._halt_decision: ToolGuardrailDecision | None = None

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        """工具调用前检查（硬阻止时才生效）。"""
        signature = ToolCallSignature.from_call(tool_name, args or {})
        if not self.config.hard_stop_enabled:
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        # 精确失败检查
        exact_count = self._exact_failure_counts.get(signature, 0)
        if exact_count >= self.config.exact_failure_block_after:
            decision = ToolGuardrailDecision(
                action="block", code="repeated_exact_failure_block",
                message=(
                    f"Blocked {tool_name}: 相同参数已失败 {exact_count} 次。"
                    "请改变策略而非无变化重试。"
                ),
                tool_name=tool_name, count=exact_count, signature=signature,
            )
            self._halt_decision = decision
            return decision

        # 无进展检查（仅对只读工具）
        if self._is_idempotent(tool_name):
            record = self._no_progress.get(signature)
            if record is not None:
                _result_hash, repeat_count = record
                if repeat_count >= self.config.no_progress_block_after:
                    decision = ToolGuardrailDecision(
                        action="block", code="idempotent_no_progress_block",
                        message=(
                            f"Blocked {tool_name}: 只读调用返回相同结果 {repeat_count} 次。"
                            "使用已获得的结果或尝试不同查询。"
                        ),
                        tool_name=tool_name, count=repeat_count, signature=signature,
                    )
                    self._halt_decision = decision
                    return decision

        return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

    def after_call(
        self, tool_name: str, args: Mapping[str, Any] | None,
        result: str | None, *, failed: bool | None = None,
    ) -> ToolGuardrailDecision:
        """工具调用后分析结果。"""
        args = args or {}
        signature = ToolCallSignature.from_call(tool_name, args)

        if failed is None:
            failed = _detect_failure(result)

        if failed:
            return self._record_failure(signature, tool_name)
        else:
            return self._record_success(signature, tool_name, result)

    def _record_failure(self, signature: ToolCallSignature, tool_name: str) -> ToolGuardrailDecision:
        exact_count = self._exact_failure_counts.get(signature, 0) + 1
        self._exact_failure_counts[signature] = exact_count
        self._no_progress.pop(signature, None)

        same_count = self._same_tool_failure_counts.get(tool_name, 0) + 1
        self._same_tool_failure_counts[tool_name] = same_count

        if self.config.hard_stop_enabled and same_count >= self.config.same_tool_failure_halt_after:
            return ToolGuardrailDecision(
                action="halt", code="same_tool_failure_halt",
                message=f"Stopped {tool_name}: 本轮已失败 {same_count} 次。请换一种方式。",
                tool_name=tool_name, count=same_count, signature=signature,
            )

        if self.config.warnings_enabled and exact_count >= self.config.exact_failure_warn_after:
            return ToolGuardrailDecision(
                action="warn", code="repeated_exact_failure_warning",
                message=f"{tool_name} 相同参数已失败 {exact_count} 次。请检查错误并改变策略。",
                tool_name=tool_name, count=exact_count, signature=signature,
            )

        if self.config.warnings_enabled and same_count >= self.config.same_tool_failure_warn_after:
            return ToolGuardrailDecision(
                action="warn", code="same_tool_failure_warning",
                message=f"{tool_name} 本轮已失败 {same_count} 次。可能陷入循环，请改变方法。",
                tool_name=tool_name, count=same_count, signature=signature,
            )

        return ToolGuardrailDecision(tool_name=tool_name, count=exact_count, signature=signature)

    def _record_success(
        self, signature: ToolCallSignature, tool_name: str, result: str | None
    ) -> ToolGuardrailDecision:
        if signature in self._exact_failure_counts:
            self._exact_failure_counts[signature] -= 1
            if self._exact_failure_counts[signature] <= 0:
                del self._exact_failure_counts[signature]

        if tool_name in self._same_tool_failure_counts:
            self._same_tool_failure_counts[tool_name] -= 1
            if self._same_tool_failure_counts[tool_name] <= 0:
                del self._same_tool_failure_counts[tool_name]

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        result_hash = _result_hash(result)
        previous = self._no_progress.get(signature)
        repeat_count = 1
        if previous is not None and previous[0] == result_hash:
            repeat_count = previous[1] + 1
        self._no_progress[signature] = (result_hash, repeat_count)

        if self.config.warnings_enabled and repeat_count >= self.config.no_progress_warn_after:
            return ToolGuardrailDecision(
                action="warn", code="idempotent_no_progress_warning",
                message=f"{tool_name} 返回相同结果 {repeat_count} 次。使用已有结果或改变查询。",
                tool_name=tool_name, count=repeat_count, signature=signature,
            )
        return ToolGuardrailDecision(tool_name=tool_name, count=repeat_count, signature=signature)

    def _is_idempotent(self, tool_name: str) -> bool:
        if tool_name in self.config.mutating_tools:
            return False
        return tool_name in self.config.idempotent_tools


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """生成被阻止调用的合成工具结果。"""
    return json.dumps({"error": decision.message, "guardrail": {"action": decision.action, "code": decision.code, "count": decision.count}}, ensure_ascii=False)


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """在工具结果末尾追加引导信息。"""
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    return f"{result or ''}\n\n[{label}: {decision.code}; count={decision.count}; {decision.message}]"


# ── helpers ──

def _detect_failure(result: str | None) -> bool:
    """检测工具结果是否表示失败。只检查结果开头，避免误判包含这些词的正常内容。"""
    if result is None:
        return False
    result_lower = result.strip().lower()
    failure_prefixes = ["error:", "failed:", "失败：", "错误：", "exception:"]
    return any(result_lower.startswith(p) for p in failure_prefixes)


def _result_hash(result: str | None) -> str:
    """计算结果哈希（用于比较两次调用是否返回相同内容）。"""
    return _sha256(result or "")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
