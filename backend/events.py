"""SSE 事件发射器 — 无依赖共享模块。

被 tools.py 和 agentic_rag/nodes/*.py 共同导入，打破循环依赖。
"""

import threading

# 全局队列：所有请求共享，通过锁保护
_rag_step_queue = None
_rag_step_lock = threading.Lock()


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列。queue 应有 put_nowait 方法和 loop 属性。"""
    global _rag_step_queue
    with _rag_step_lock:
        _rag_step_queue = queue


def get_rag_step_queue():
    """获取当前的 RAG 步骤队列。"""
    with _rag_step_lock:
        return _rag_step_queue


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """线程安全地向 SSE 队列推送 RAG 步骤事件。"""
    step = {"type": "rag_step", "step": {"icon": icon, "label": label, "detail": detail}}
    with _rag_step_lock:
        queue = _rag_step_queue
    if queue is not None:
        loop = getattr(queue, '_loop', None)
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(queue.put_nowait, step)
            except RuntimeError:
                pass  # Loop was closed between check and call
