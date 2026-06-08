"""Token 估算工具 — 用于上下文压缩触发判断、预算控制。

注意：这不是精确计数，是快速估算（~字符合/4）。
在非关键路径上做预判足够使用，精确计数依赖 LLM tokenizer。
"""

from typing import List

from langchain_core.messages import BaseMessage


def _is_cjk(c: str) -> bool:
    """判断字符是否为 CJK 汉字（含扩展区及标点）。"""
    cp = ord(c)
    return (
        0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or    # CJK Extension A
        0x20000 <= cp <= 0x2A6DF or  # CJK Extension B
        0x2A700 <= cp <= 0x2B73F or  # CJK Extension C
        0x2B740 <= cp <= 0x2B81F or  # CJK Extension D
        0x2B820 <= cp <= 0x2CEAF or  # CJK Extension E
        0x2CEB0 <= cp <= 0x2EBEF or  # CJK Extension F
        0x30000 <= cp <= 0x3134F or  # CJK Extension G
        0xF900 <= cp <= 0xFAFF or    # CJK Compatibility Ideographs
        0x2F800 <= cp <= 0x2FA1F or  # CJK Compatibility Ideographs Supplement
        0x3000 <= cp <= 0x303F or    # CJK Symbols and Punctuation
        0xFF00 <= cp <= 0xFFEF or    # Fullwidth Forms (fullwidth punctuation)
        0xFE30 <= cp <= 0xFE4F       # CJK Compatibility Forms
    )


def estimate_text_tokens(text: str) -> int:
    """估算单段文本的 token 数。

    中英文混合场景：中文约 1.5 字符/token，英文约 4 字符/token。
    """
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if _is_cjk(c))
    other_chars = len(text) - chinese_chars
    return max(1, int(chinese_chars / 1.5 + other_chars / 4))


def estimate_message_tokens(message: BaseMessage) -> int:
    """估算单条消息的 token 数（含 role 开销）。"""
    # 每条消息有 role/system 等元数据开销约 4 tokens
    overhead = 4
    content = message.content or ""
    if isinstance(content, str):
        return overhead + estimate_text_tokens(content)
    elif isinstance(content, list):
        total = overhead
        for block in content:
            if isinstance(block, dict):
                total += estimate_text_tokens(block.get("text", ""))
            elif isinstance(block, str):
                total += estimate_text_tokens(block)
        return total
    return overhead


def estimate_messages_tokens(messages: List[BaseMessage]) -> int:
    """估算消息列表的总 token 数。"""
    return sum(estimate_message_tokens(m) for m in messages)
