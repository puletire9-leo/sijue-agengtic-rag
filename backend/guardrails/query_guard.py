"""查询护栏 (QueryGuard) — 在输入阶段检测用户意图中的危险模式。

两层防护:
1. 正则快速匹配 — 拦截已知攻击模式（零延迟）
2. LLM 语义判断 — 捕获变体和绕过攻击（仅当正则未匹配时触发）
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from guardrails.rules import QUERY_RULES, QUERY_GUARD_LLM_PROMPT, GuardRule, match_rules
from metrics import GUARDRAIL_BLOCKS

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """护栏检查结果。"""
    blocked: bool = False
    reason: Optional[str] = None
    matched_rules: List[GuardRule] = field(default_factory=list)
    severity: Optional[str] = None
    transformed_text: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "matched_rules": [r.name for r in self.matched_rules],
            "severity": self.severity,
        }


class QueryGuard:
    """查询护栏 — 用户输入安全检查，支持 LLM 语义二次判断。

    用法:
        guard = QueryGuard()
        result = guard.check("用户输入内容")
        # 或带 LLM:
        result = guard.check_with_llm("用户输入内容", llm)
    """

    def __init__(
        self,
        enabled: bool = True,
        max_length: int = 10000,
        rules: Optional[List[GuardRule]] = None,
        llm_enabled: bool = False,
    ):
        self.enabled = enabled
        self.max_length = max_length
        self.rules = rules or QUERY_RULES
        self.llm_enabled = llm_enabled

    def check(self, text: str) -> GuardResult:
        """正则快速检查。"""
        if not self.enabled:
            return GuardResult()

        if not text or not text.strip():
            GUARDRAIL_BLOCKS.labels(type="query", severity="low").inc()
            return GuardResult(
                blocked=True,
                reason="输入为空",
                severity="low",
            )

        if len(text) > self.max_length:
            # 先在完整文本上匹配规则，再截断（防止恶意内容放在截断点之后）
            matched = match_rules(text, self.rules)
            if matched:
                high_severity = any(r.severity == "high" for r in matched)
                if high_severity:
                    GUARDRAIL_BLOCKS.labels(type="query", severity="high").inc()
                    return GuardResult(
                        blocked=True,
                        reason=f"检测到危险内容: {matched[0].description}",
                        matched_rules=matched,
                        severity="high",
                    )
            GUARDRAIL_BLOCKS.labels(type="query", severity="low").inc()
            return GuardResult(
                blocked=True,
                reason=f"输入超过最大长度限制 ({self.max_length} 字符)",
                severity="low",
            )

        matched = match_rules(text, self.rules)
        if matched:
            high_severity = any(r.severity == "high" for r in matched)
            if high_severity:
                GUARDRAIL_BLOCKS.labels(type="query", severity="high").inc()
                return GuardResult(
                    blocked=True,
                    reason=f"检测到危险内容: {matched[0].description}",
                    matched_rules=matched,
                    severity="high",
                )
            return GuardResult(
                blocked=False,
                reason=matched[0].description,
                matched_rules=matched,
                severity=matched[0].severity,
            )

        return GuardResult()

    def check_with_llm(self, text: str, llm) -> GuardResult:
        """两层检查：先跑正则，未命中时用 LLM 做语义判断。"""
        result = self.check(text)
        if result.blocked or not self.llm_enabled or not llm:
            return result

        try:
            from guardrails.rules import run_llm_guard
            llm_result = run_llm_guard(llm, QUERY_GUARD_LLM_PROMPT, result.transformed_text or text)
            if not llm_result["safe"]:
                GUARDRAIL_BLOCKS.labels(type="query", severity="high").inc()
                return GuardResult(
                    blocked=True,
                    reason=f"LLM 语义检测: {llm_result['reason']}",
                    severity="high",
                )
        except Exception:
            logger.warning("LLM 查询护栏检查异常，放行原始结果", exc_info=True)
            return result

        return result
