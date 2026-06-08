"""CompressCheck — 上下文压缩检查与执行。

在消息过多或文档过长时触发五阶段压缩。
"""

from agentic_rag.state import AgenticRAGState
from agentic_rag.config import compression as compression_cfg
from agentic_rag.token_estimator import estimate_messages_tokens
from core.context_compressor import ContextCompressor
from events import emit_rag_step


_compressor = ContextCompressor()


def compress_check(state: AgenticRAGState) -> dict:
    """检查上下文大小，必要时进行压缩。"""
    messages = state.get("messages", [])

    # 检查消息是否超阈值
    msg_tokens = estimate_messages_tokens(messages)
    should_compress = msg_tokens > compression_cfg.COMPRESSION_THRESHOLD

    if not should_compress:
        return {"_needs_compression": False, "compressed_messages": None}

    emit_rag_step("🗜️", "上下文压缩触发", f"当前消息 token: {msg_tokens}")

    try:
        result = _compressor.compress(messages)
        compressed = result.messages
        emit_rag_step("✅", f"压缩完成: {len(messages)} -> {len(compressed)} 条消息")
        return {
            "compressed_messages": compressed,
            "_needs_compression": True,
        }
    except Exception as e:
        emit_rag_step("⚠️", "压缩失败", str(e)[:50])
        return {"_needs_compression": False, "compressed_messages": None, "_compression_failed": True}
