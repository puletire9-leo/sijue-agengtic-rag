"""输出护栏 (OutputGuard) — 在输出阶段检测模型可能产生的危险/敏感内容。

两层防护:
1. 正则快速匹配 — 拦截已知泄露模式（密码、密钥、有害关键词）
2. LLM 语义判断 — 捕获变形泄露和隐含有害内容
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

from guardrails.rules import OUTPUT_RULES, OUTPUT_GUARD_LLM_PROMPT, GuardRule, match_rules
from metrics import GUARDRAIL_BLOCKS


@dataclass
class OutputGuardResult:
    """输出护栏检查结果。"""
    blocked: bool = False
    reason: Optional[str] = None
    matched_rules: List[GuardRule] = field(default_factory=list)
    redacted_text: Optional[str] = None
    redactions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "matched_rules": [r.name for r in self.matched_rules],
            "redaction_count": len(self.redactions),
        }


def _try_llm_rewrite(text: str, matches: list, llm=None) -> str:
    """Attempt to rewrite text to remove sensitive content using LLM."""
    if os.getenv("GUARDRAIL_REWRITE_ENABLED", "false").lower() != "true":
        return ""
    try:
        import re as _re
        if llm is None:
            from agentic_rag.llm import get_llm
            llm = get_llm()  # Tier 1
        if not llm:
            return ""

        # Redact sensitive patterns before sending to external LLM
        redacted_text = text
        redacted_text = _re.sub(
            r"(?i)(password|passwd|secret|token|api_key|private_key)\s*[:=]\s*\S+",
            r"\1=[REDACTED]",
            redacted_text,
        )
        redacted_text = _re.sub(
            r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----.*?-----END\s+(RSA\s+)?PRIVATE\s+KEY-----",
            "[REDACTED_PRIVATE_KEY]",
            redacted_text,
            flags=_re.DOTALL,
        )
        redacted_text = _re.sub(r"(?i)(sk-[a-zA-Z0-9]{20,})", "[REDACTED_KEY]", redacted_text)
        redacted_text = _re.sub(r"(?i)(ghp_[a-zA-Z0-9]{36})", "[REDACTED_TOKEN]", redacted_text)
        redacted_text = _re.sub(r"(?i)(xoxb-[a-zA-Z0-9-]+)", "[REDACTED_TOKEN]", redacted_text)

        match_descriptions = ", ".join(matches[:5])
        prompt = (
            f"以下文本包含敏感信息（{match_descriptions}），请改写为安全版本，"
            f"去除敏感部分但保留其余信息的完整性。只输出改写后的文本，不要解释。\n\n"
            f"原文：{redacted_text}"
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        rewritten = response.content.strip()
        if rewritten and len(rewritten) > len(text) * 0.3:  # Sanity check
            return rewritten
    except Exception as e:
        logger.warning(f"Guardrail LLM rewrite failed: {e}")
    return ""


class OutputGuard:
    """输出护栏 — 模型输出安全检查，支持 LLM 语义二次判断。

    用法:
        guard = OutputGuard()
        result = guard.check("模型输出内容")
        # 或带 LLM:
        result = guard.check_with_llm("模型输出内容", llm)
    """

    def __init__(
        self,
        enabled: bool = True,
        rules: Optional[List[GuardRule]] = None,
        llm_enabled: bool = False,
    ):
        self.enabled = enabled
        self.rules = rules or OUTPUT_RULES
        self.llm_enabled = llm_enabled

    def check(self, text: str, llm=None) -> OutputGuardResult:
        """正则快速检查。"""
        if not self.enabled or not text:
            return OutputGuardResult(redacted_text=text)

        matched = match_rules(text, self.rules)
        if not matched:
            return OutputGuardResult(redacted_text=text)

        redacted = text
        redactions: List[str] = []
        high_severity = False

        for rule in matched:
            if rule.severity == "high":
                high_severity = True
            redacted = rule.pattern.sub("[已过滤]", redacted)
            redactions.append(rule.name)

        if high_severity:
            GUARDRAIL_BLOCKS.labels(type="output", severity="high").inc()
            return OutputGuardResult(
                blocked=True,
                reason=f"输出包含高风险内容: {matched[0].description}",
                matched_rules=matched,
                redacted_text=None,
                redactions=redactions,
            )

        # For low-severity detections, attempt LLM rewrite before falling back to regex
        rewritten = _try_llm_rewrite(text, redactions, llm=llm)
        if rewritten:
            return OutputGuardResult(
                blocked=False,
                reason=f"已改写 {len(redactions)} 项敏感内容",
                matched_rules=matched,
                redacted_text=rewritten,
                redactions=redactions,
            )

        return OutputGuardResult(
            blocked=False,
            reason=f"已脱敏 {len(redactions)} 项内容",
            matched_rules=matched,
            redacted_text=redacted,
            redactions=redactions,
        )

    def check_with_llm(self, text: str, llm) -> OutputGuardResult:
        """两层检查：先跑正则，未命中时用 LLM 做语义判断。"""
        result = self.check(text, llm=llm)
        if result.blocked or not self.llm_enabled or not llm:
            return result

        try:
            from guardrails.rules import run_llm_guard
            llm_result = run_llm_guard(llm, OUTPUT_GUARD_LLM_PROMPT, result.redacted_text or text)
            if not llm_result["safe"]:
                GUARDRAIL_BLOCKS.labels(type="output", severity="high").inc()
                return OutputGuardResult(
                    blocked=True,
                    reason=f"LLM 语义检测: {llm_result['reason']}",
                    matched_rules=[],
                    redacted_text=None,
                )
        except Exception:
            logger.warning("LLM 输出护栏检查异常，放行原始结果", exc_info=True)
            return result

        return result
