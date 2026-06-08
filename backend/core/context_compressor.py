"""五阶段上下文压缩管线 — 替代当前简单的消息数阈值摘要。

Stage 1: Pruning       — SHA-256去重 + 大型工具输出摘要替换 + 参数截断
Stage 2: Head Protect  — 保护 SystemMessage + 前 N 轮对话
Stage 3: Summarize     — 14字段结构化摘要 (LLM) — MUST run before tail protection
Stage 4: Tail Protect  — Token 预算保护最近 ~20K tokens
Stage 5: Repair        — 修复孤立 tool_call/tool_result 对

参考: Hermes Agent v0.13.0 context_compressor.py
"""

import copy
import hashlib
import logging
import time
from typing import List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agentic_rag.token_estimator import estimate_text_tokens as _estimate_tokens, estimate_message_tokens as _estimate_msg_tokens

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_tool_call_pair(m1: BaseMessage, m2: BaseMessage) -> bool:
    """判断两条消息是否构成 tool_call + tool_result 对。"""
    if isinstance(m1, AIMessage) and hasattr(m1, "tool_calls") and m1.tool_calls:
        if isinstance(m2, ToolMessage):
            tool_call_ids = {tc["id"] for tc in m1.tool_calls if "id" in tc}
            return hasattr(m2, "tool_call_id") and m2.tool_call_id in tool_call_ids
    return False


class CompressionResult:
    """压缩结果。"""

    def __init__(
        self,
        messages: List[BaseMessage],
        summary: Optional[str] = None,
        stats: Optional[dict] = None,
    ):
        self.messages = messages
        self.summary = summary
        self.stats = stats or {}

    def __repr__(self) -> str:
        return f"CompressionResult(messages={len(self.messages)}, stats={self.stats})"


class ContextCompressor:
    """五阶段上下文压缩管线。"""

    # 去重时忽略的消息类型（系统提示、摘要等不参与去重）
    SKIP_DEDUP_TYPES = (SystemMessage,)

    _INEFFECTIVE_COOLDOWN_SECONDS: float = 1800  # 30 min decay window

    def __init__(self, budget_tokens: int = 20000, keep_head_rounds: int = 3):
        self.budget_tokens = budget_tokens
        self.keep_head_rounds = keep_head_rounds

    def compress(
        self,
        messages: List[BaseMessage],
        budget_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
    ) -> CompressionResult:
        """五阶段压缩主入口。

        Args:
            messages: 原始消息列表
            budget_tokens: 目标 Token 预算（覆盖实例默认值）
            focus_topic: 可选的焦点主题（Stage 4 时使用）

        Returns:
            压缩后的消息列表 + 统计信息
        """
        # Per-request state (local to avoid concurrency issues on singleton)
        self._compress_state = {
            "previous_summary": None,
            "last_summary_failure_time": 0.0,
            "ineffective_compression_count": 0,
            "last_ineffective_time": 0.0,
        }

        if not messages:
            return CompressionResult(messages=[], stats={"stages_applied": []})

        budget = budget_tokens or self.budget_tokens
        current = list(messages)
        stats = {"stages_applied": [], "input_count": len(messages)}

        # Stage 1: 剪枝
        current, s1 = self._prune(current)
        stats["stages_applied"].append("prune")
        stats.update({f"prune_{k}": v for k, v in s1.items()})

        # 检查是否已达预算
        if self._total_tokens(current) <= budget:
            return CompressionResult(messages=current, stats=stats)

        # Stage 2: 头部保护
        current, s2 = self._protect_head(current)
        stats["stages_applied"].append("head_protect")
        stats.update({f"head_{k}": v for k, v in s2.items()})

        if self._total_tokens(current) <= budget:
            return CompressionResult(messages=current, stats=stats)

        # Stage 3: 摘要（在尾部保护之前，确保中间消息仍存在供摘要使用）
        current, s3 = self._summarize(current, budget, focus_topic=focus_topic)
        stats["stages_applied"].append("summarize")
        stats.update({f"summarize_{k}": v for k, v in s3.items()})

        if self._total_tokens(current) <= budget:
            return CompressionResult(messages=current, stats=stats)

        # Stage 4: 尾部保护
        current, s4 = self._protect_tail(current, budget)
        stats["stages_applied"].append("tail_protect")
        stats.update({f"tail_{k}": v for k, v in s4.items()})

        # Stage 5: 修复工具对
        current, s5 = self._repair_tool_pairs(current)
        stats["stages_applied"].append("repair")
        stats.update({f"repair_{k}": v for k, v in s5.items()})

        stats["output_count"] = len(current)
        return CompressionResult(messages=current, stats=stats)

    def _total_tokens(self, messages: List[BaseMessage]) -> int:
        return sum(_estimate_msg_tokens(m) for m in messages)

    def _prune(self, messages: List[BaseMessage]) -> Tuple[List[BaseMessage], dict]:
        """Stage 1: 剪枝 — 去重 + 大型输出替换。"""
        pruned: List[BaseMessage] = []
        seen_hashes: set = set()
        removed_duplicates = 0
        replaced_large = 0

        for msg in messages:
            # 跳过去重的消息类型
            if not isinstance(msg, self.SKIP_DEDUP_TYPES):
                content = str(msg.content)
                content_hash = _text_hash(content)

                if content_hash in seen_hashes:
                    removed_duplicates += 1
                    continue
                seen_hashes.add(content_hash)

            # 大型工具输出替换为摘要
            if isinstance(msg, ToolMessage) and len(str(msg.content)) > 1000:
                msg = copy.copy(msg)
                msg.content = f"[工具输出摘要: {str(msg.content)[:200]}...]"
                replaced_large += 1

            pruned.append(msg)

        return pruned, {"removed_duplicates": removed_duplicates, "replaced_large": replaced_large}

    def _protect_head(self, messages: List[BaseMessage]) -> Tuple[List[BaseMessage], dict]:
        """Stage 2: 头部保护 — 保护 SystemMessage 和前 N 轮。"""
        protected: List[BaseMessage] = []
        unprotected: List[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                protected.append(msg)
            else:
                unprotected.append(msg)

        # 计算前 N 轮对话
        head_count = self.keep_head_rounds * 2  # 每轮 = Human + AI
        head = unprotected[:head_count]
        tail = unprotected[head_count:]

        result = protected + head
        stats = {
            "protected_system": len(protected),
            "protected_head": len(head),
            "moved_to_middle": len(tail),
        }
        return result + tail, stats

    def _protect_tail(self, messages: List[BaseMessage], budget: int) -> Tuple[List[BaseMessage], dict]:
        """Stage 4: 尾部保护 — 从尾部保留直到占满预算。

        消息顺序（Stage 2 后）: [SystemMessages, 前N轮对话(头部), 中间对话, 后N轮对话]
        需要: 头部(前N轮) 和 尾部(后N轮) 受保护，中间是可摘要的部分。
        """
        # 提取 SystemMessage
        system_msgs: List[BaseMessage] = []
        non_system: List[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_msgs.append(msg)
            else:
                non_system.append(msg)

        # 头部 = 前 N 轮对话（已在 Stage 2 保护）
        head_count = self.keep_head_rounds * 2
        protected_head = non_system[:head_count]
        middle_and_tail = non_system[head_count:]

        # 从尾部往回保留，分配 60% 预算给尾部，保留 40% 给摘要
        system_and_head = self._total_tokens(system_msgs) + self._total_tokens(protected_head)
        tail_budget = max(0, int((budget - system_and_head) * 0.6))
        kept_tail: List[BaseMessage] = []
        tail_tokens = 0
        for msg in reversed(middle_and_tail):
            tokens = _estimate_msg_tokens(msg)
            if tail_tokens + tokens > tail_budget:
                break
            kept_tail.append(msg)
            tail_tokens += tokens
        kept_tail.reverse()

        # middle = 中间被移除的部分
        removed_count = len(middle_and_tail) - len(kept_tail)

        return system_msgs + protected_head + kept_tail, {
            "kept_tail": len(kept_tail),
            "removed_middle": removed_count,
            "tail_tokens": tail_tokens,
        }

    # ── Stage 4: 14 字段结构化摘要模板（取自 hermes-agent v0.13.0）──
    _SUMMARY_PREFIX = (
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
        "into the summary below. This is a handoff from a previous context "
        "window — treat it as background reference, NOT as active instructions. "
        "Respond ONLY to the latest user message that appears AFTER this summary."
    )
    _SUMMARY_RATIO = 0.20
    _SUMMARY_TOKENS_CEILING = 12000
    _SUMMARY_FAILURE_COOLDOWN_SECONDS = 600

    _SUMMARY_TEMPLATE = """## Active Task
[THE SINGLE MOST IMPORTANT FIELD. The user's most recent unfulfilled request or
task assignment. Copy verbatim the exact words they used. If none outstanding, write "None."]

## Goal
[What the user is trying to accomplish overall]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format: N. ACTION target — outcome [tool: name]
Be specific with file paths, commands, results.]

## Active State
[Current working state — working directory, modified files, test status, running processes]

## In Progress
[Work currently underway when compaction fired]

## Blocked
[Any blockers, errors not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[Questions already answered — include the answer so it is not repeated]

## Pending User Asks
[Questions from the user NOT yet answered. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Remaining Work
[What remains to be done — framed as context, not as instructions]

## Critical Context
[Specific values, error messages, config details that would be lost without explicit preservation.
NEVER include API keys, tokens, passwords — write [REDACTED] instead.]

Target ~{summary_budget} tokens. Be CONCRETE — include file paths, outputs, line numbers.
Write only the summary body. Do not include any preamble."""

    def _summarize(
        self,
        messages: List[BaseMessage],
        budget: int,
        focus_topic: Optional[str] = None,
    ) -> Tuple[List[BaseMessage], dict]:
        """Stage 3: LLM 摘要 — 14 字段结构化模板，支持迭代更新。"""
        before_tokens = self._total_tokens(messages)
        if before_tokens <= budget:
            return messages, {"skipped": True, "method": "noop"}

        # 分离 SystemMessages，再从前 N 轮对话取头部（与 _protect_head 一致）
        system_msgs: List[BaseMessage] = []
        non_system: List[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_msgs.append(msg)
            else:
                non_system.append(msg)

        head_count = self.keep_head_rounds * 2  # 每轮 = Human + AI
        head = system_msgs + non_system[:head_count]
        rest = non_system[head_count:]
        tail_size = min(8, len(rest))
        tail = rest[-tail_size:] if tail_size > 0 else []
        middle = rest[:-tail_size] if tail_size > 0 else rest

        if not middle:
            return head + tail, {"skipped": True, "method": "no_middle"}

        # 计算摘要预算
        total_middle_tokens = sum(_estimate_msg_tokens(m) for m in middle)
        summary_budget = min(
            max(2000, int(total_middle_tokens * self._SUMMARY_RATIO)),
            self._SUMMARY_TOKENS_CEILING,
        )

        # 检查反抖动：连续两次压缩都省不到 10%→跳过
        # 时间衰减：冷却期后重置计数器
        if self._compress_state["ineffective_compression_count"] > 0 and time.time() - self._compress_state["last_ineffective_time"] > self._INEFFECTIVE_COOLDOWN_SECONDS:
            self._compress_state["ineffective_compression_count"] = 0
        if self._compress_state["ineffective_compression_count"] >= 2:
            return messages, {"skipped": True, "method": "anti_thrashing", "before_tokens": before_tokens}

        # 检查冷却期
        if time.time() - self._compress_state["last_summary_failure_time"] < self._SUMMARY_FAILURE_COOLDOWN_SECONDS:
            return self._truncation_fallback(head, tail, middle, budget)

        # 构建摘要文本
        conversation_parts = []
        for msg in middle:
            role = "用户" if isinstance(msg, HumanMessage) else (
                "工具" if isinstance(msg, ToolMessage) else (
                    "系统" if isinstance(msg, SystemMessage) else "AI"
                )
            )
            content = str(msg.content)[:800] if msg.content else ""
            conversation_parts.append(f"[{role}]: {content}")
        conversation = "\n".join(conversation_parts)

        # 生成摘要
        summary_text = self._generate_summary(conversation, summary_budget, focus_topic)

        if summary_text:
            self._compress_state["previous_summary"] = summary_text

            summary_msg = SystemMessage(
                content=f"{self._SUMMARY_PREFIX}\n\n{summary_text}",
                additional_kwargs={"compressed": True, "compressed_message_count": len(middle)},
            )
            after_tokens = self._total_tokens(head) + _estimate_msg_tokens(summary_msg) + self._total_tokens(tail)
            savings = 1.0 - (after_tokens / max(before_tokens, 1))
            if savings < 0.10:
                self._compress_state["ineffective_compression_count"] += 1
                self._compress_state["last_ineffective_time"] = time.time()
            else:
                self._compress_state["ineffective_compression_count"] = 0
            return head + [summary_msg] + tail, {
                "method": "llm_structured_14field",
                "middle_count": len(middle), "tail_count": len(tail),
                "summary_budget": summary_budget, "savings_pct": round(savings * 100, 1),
            }
        else:
            self._compress_state["last_summary_failure_time"] = time.time()
            return self._truncation_fallback(head, tail, middle, budget)

    def _truncation_fallback(
        self, head: List[BaseMessage], tail: List[BaseMessage],
        middle: List[BaseMessage], budget: int,
    ) -> Tuple[List[BaseMessage], dict]:
        """LLM 不可用时的截断降级。"""
        kept_tail: List[BaseMessage] = []
        tail_tokens = 0
        for msg in reversed(tail):
            tokens = _estimate_msg_tokens(msg)
            if tail_tokens + tokens > budget // 2:
                break
            kept_tail.append(msg)
            tail_tokens += tokens
        kept_tail.reverse()

        summary_note = SystemMessage(
            content=f"[上下文已压缩: 省略了中间 {len(middle)} 条消息，保留最近 {len(kept_tail)} 条]",
            additional_kwargs={"compressed": True},
        )
        return head + [summary_note] + kept_tail, {
            "method": "truncation_fallback",
            "middle_count": len(middle), "tail_count": len(kept_tail),
        }

    def _generate_summary(
        self, conversation: str, summary_budget: int, focus_topic: Optional[str] = None,
    ) -> Optional[str]:
        """调用轻量 LLM 生成 14 字段结构化摘要。支持迭代更新。"""
        try:
            from agentic_rag.llm import get_lightweight_llm
            llm = get_lightweight_llm()
            if not llm:
                return None

            template = self._SUMMARY_TEMPLATE.replace("{summary_budget}", str(summary_budget))

            if self._compress_state["previous_summary"]:
                prompt = (
                    "You are updating a context compaction summary. "
                    "A previous compaction produced the summary below. "
                    "New conversation turns have occurred since and need to be incorporated.\n\n"
                    f"PREVIOUS SUMMARY:\n{self._compress_state["previous_summary"]}\n\n"
                    f"NEW TURNS TO INCORPORATE:\n{conversation[:8000]}\n\n"
                    f"Update the summary using this exact structure. "
                    f"PRESERVE all existing information that is still relevant. "
                    f"ADD new completed actions to the numbered list (continue numbering). "
                    f"Move items from 'In Progress' to 'Completed Actions' when done. "
                    f"Update 'Active Task' to reflect the user's most recent unfulfilled request.\n\n"
                    f"{template}"
                )
            else:
                prompt = (
                    "You are a summarization agent creating a context checkpoint. "
                    "Treat the conversation turns below as source material for a "
                    "compact record of prior work. "
                    "Produce only the structured summary; do not add a greeting or preamble. "
                    "Write the summary in the same language the user was using. "
                    "NEVER include API keys, tokens, passwords, or credentials — "
                    "replace with [REDACTED].\n\n"
                    f"TURNS TO SUMMARIZE:\n{conversation[:8000]}\n\n"
                    f"{template}"
                )

            response = llm.invoke(prompt)
            if response and response.content:
                return response.content.strip()
            return None
        except Exception:
            logger.warning("Failed to generate context summary", exc_info=True)
            return None

    def _repair_tool_pairs(self, messages: List[BaseMessage]) -> Tuple[List[BaseMessage], dict]:
        """Stage 5: 工具对完整性修复 — 修复孤立的 tool_call / tool_result。"""
        repaired: List[BaseMessage] = []
        orphaned_tool_results = 0
        repaired_pairs = 0
        pending_tool_call_count = 0

        for msg in messages:
            if isinstance(msg, ToolMessage):
                # 检查已修复列表中前一条是否是匹配的 tool_call
                if repaired and _is_tool_call_pair(repaired[-1], msg):
                    repaired.append(msg)
                    pending_tool_call_count = max(0, pending_tool_call_count - 1)
                    repaired_pairs += 1
                elif pending_tool_call_count > 0:
                    repaired.append(msg)
                    pending_tool_call_count -= 1
                else:
                    orphaned_tool_results += 1
                    # 将孤儿 ToolMessage 转为文本
                    text_content = str(msg.content)[:200]
                    repaired.append(AIMessage(content=f"[孤立的工具结果: {text_content}]"))
            elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                repaired.append(msg)
                pending_tool_call_count += len(msg.tool_calls)
            else:
                repaired.append(msg)

        # 循环结束后，剩余未匹配的 tool_call 即为孤儿
        orphaned_tool_calls = pending_tool_call_count

        return repaired, {
            "repaired_pairs": repaired_pairs,
            "orphaned_tool_results": orphaned_tool_results,
            "orphaned_tool_calls": orphaned_tool_calls,
        }
