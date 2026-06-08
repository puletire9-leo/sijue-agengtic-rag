"""死循环检测与恢复 — 检测 Agent 的重复行为模式并自动恢复。

参考: Hermes Agent v0.13.0 _is_stuck / attempt_loop_recovery

检测模式:
1. 重复工具调用: 同一工具连续调用 N 次以上
2. 重复 LLM 输出: 输出内容高度相似
3. 语义循环: 工具调用序列出现循环模式
"""

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Set

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage


@dataclass
class LoopState:
    """死循环检测结果。"""
    is_stuck: bool = False
    pattern: Optional[str] = None          # 检测到的模式类型
    repeat_count: int = 0                  # 重复次数
    repeating_tool: Optional[str] = None   # 重复的工具名称
    confidence: float = 0.0                # 检测置信度 0~1
    details: dict = field(default_factory=dict)  # 附加详情


class LoopDetector:
    """死循环检测器。

    用法:
        detector = LoopDetector()
        state = detector.detect(messages)
        if state.is_stuck:
            messages = detector.recover(messages, state)
    """

    # 同一工具连续调用的最大允许次数
    MAX_REPEATED_TOOL_CALLS = 3
    # 语义相似度阈值（简化版：基于文本重合度）
    SIMILARITY_THRESHOLD = 0.85
    # 最大恢复尝试次数，超过后强制停止
    MAX_RECOVERY_ATTEMPTS = 3

    def __init__(self) -> None:
        self._recovery_count: int = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Reset recovery counter. Call at the start of each conversation."""
        with self._lock:
            self._recovery_count = 0

    def detect(self, messages: List[BaseMessage]) -> LoopState:
        """检测消息序列中是否存在死循环模式。"""
        # 模式1: 重复工具调用
        tool_state = self._detect_repeated_tool(messages)
        if tool_state.is_stuck:
            return tool_state

        # 模式2: 重复 LLM 输出
        output_state = self._detect_repeated_output(messages)
        if output_state.is_stuck:
            return output_state

        # 模式3: 工具调用序列循环
        sequence_state = self._detect_tool_sequence_loop(messages)
        if sequence_state.is_stuck:
            return sequence_state

        return LoopState()

    def recover(self, messages: List[BaseMessage], loop_state: LoopState) -> List[BaseMessage]:
        """从死循环中恢复。

        策略: 插入一条 SystemMessage 打破循环，指示 LLM 换一种方式。
        如果恢复次数超过 MAX_RECOVERY_ATTEMPTS，抛出异常强制停止。
        """
        with self._lock:
            self._recovery_count += 1
            current_count = self._recovery_count

        if current_count > self.MAX_RECOVERY_ATTEMPTS:
            raise RuntimeError(
                f"死循环恢复失败: 已尝试 {current_count} 次仍无法摆脱循环 "
                f"(模式: {loop_state.pattern}, 重复 {loop_state.repeat_count} 次)。"
                "强制停止以避免无限循环。"
            )

        recovery_msg = SystemMessage(
            content=(
                f"[检测到死循环] 系统检测到重复模式: {loop_state.pattern} "
                f"(重复 {loop_state.repeat_count} 次)。"
                f"[恢复尝试 {current_count}/{self.MAX_RECOVERY_ATTEMPTS}] "
                "请换一种方式处理当前任务，不要重复之前的操作。"
                "如果无法继续，请给出当前进展并结束。"
            )
        )
        return messages + [recovery_msg]

    def _detect_repeated_tool(self, messages: List[BaseMessage]) -> LoopState:
        """检测重复工具调用。

        以 AIMessage 为单位计数，用所有 tool_calls 的名称元组代表该 turn，
        避免单条多工具调用被误判为循环。
        """
        turn_tools: List[tuple] = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                continue
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                names = []
                for tc in msg.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if name:
                        names.append(name)
                if names:
                    turn_tools.append(tuple(names))
                if len(turn_tools) >= self.MAX_REPEATED_TOOL_CALLS:
                    break

        if len(turn_tools) >= self.MAX_REPEATED_TOOL_CALLS:
            first_pattern = turn_tools[0]
            if all(t == first_pattern for t in turn_tools):
                return LoopState(
                    is_stuck=True,
                    pattern="repeated_tool_call",
                    repeat_count=len(turn_tools),
                    repeating_tool=first_pattern[0] if len(first_pattern) == 1 else ",".join(first_pattern),
                    confidence=0.9,
                    details={"turns": [list(t) for t in turn_tools]},
                )

        return LoopState()

    def _detect_repeated_output(self, messages: List[BaseMessage]) -> LoopState:
        """检测重复 LLM 输出。"""
        outputs: List[str] = []
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                outputs.append(str(msg.content)[:200])
                if len(outputs) >= 3:
                    break

        if len(outputs) >= 2:
            # 简化相似度检测：字符串重合度
            sim = self._text_similarity(outputs[0], outputs[1])
            if sim > self.SIMILARITY_THRESHOLD:
                if len(outputs) >= 3:
                    sim2 = self._text_similarity(outputs[1], outputs[2])
                    if sim2 > self.SIMILARITY_THRESHOLD:
                        return LoopState(
                            is_stuck=True,
                            pattern="repeated_output",
                            repeat_count=len(outputs),
                            confidence=sim,
                            details={"similarity": sim},
                        )
                else:
                    # len == 2 with high similarity is enough to report stuck
                    return LoopState(
                        is_stuck=True,
                        pattern="repeated_output",
                        repeat_count=len(outputs),
                        confidence=sim,
                        details={"similarity": sim},
                    )

        return LoopState()

    def _detect_tool_sequence_loop(self, messages: List[BaseMessage]) -> LoopState:
        """检测工具调用序列循环（任意重复子序列，长度 2+）。"""
        # 提取工具调用序列
        sequence: List[str] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if name:
                        sequence.append(name)

        if len(sequence) < 4:
            return LoopState()

        # 检测任意长度 2..len//2 的重复子序列
        for pattern_len in range(2, len(sequence) // 2 + 1):
            pattern = sequence[:pattern_len]
            matches = True
            for i in range(pattern_len, len(sequence)):
                if sequence[i] != pattern[i % pattern_len]:
                    matches = False
                    break
            if matches and len(sequence) >= pattern_len * 2:
                return LoopState(
                    is_stuck=True,
                    pattern="tool_sequence_loop",
                    repeat_count=len(sequence),
                    confidence=0.85,
                    details={"sequence": sequence, "pattern_len": pattern_len},
                )

        return LoopState()

    def _text_similarity(self, a: str, b: str) -> float:
        """Calculate text similarity using character bigram Jaccard."""
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        def bigrams(text: str) -> set:
            return {text[i:i+2] for i in range(len(text) - 1)}

        bg_a = bigrams(a)
        bg_b = bigrams(b)

        if not bg_a or not bg_b:
            return 0.0

        intersection = bg_a & bg_b
        union = bg_a | bg_b
        return len(intersection) / len(union)
