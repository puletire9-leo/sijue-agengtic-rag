from typing import Optional
import contextvars
import logging
import os
import threading
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
load_dotenv()

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

_RAG_CONTEXT_VAR: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar('_rag_context', default=None)
_TOOL_CALLS_GUARD: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar('_tool_calls_guard', default=None)
_CURRENT_USER_VAR: contextvars.ContextVar[str] = contextvars.ContextVar('_current_user', default='default')
_CURRENT_KB_IDS_VAR: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar('_current_kb_ids', default=None)

# 全局变量：跨线程传播 rag_context（ContextVar 在 LangChain 工具执行器中不可靠）
_last_rag_context: Optional[dict] = None
_rag_context_lock = threading.Lock()


def _set_last_rag_context(context: dict):
    global _last_rag_context
    _RAG_CONTEXT_VAR.set(context)
    with _rag_context_lock:
        _last_rag_context = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _last_rag_context
    # 优先从全局变量读（跨线程可靠）
    with _rag_context_lock:
        context = _last_rag_context
    if clear:
        with _rag_context_lock:
            _last_rag_context = None
        _RAG_CONTEXT_VAR.set(None)
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    _TOOL_CALLS_GUARD.set({"count": 0})


def set_current_user(user_id: str):
    """设置当前用户 ID（Agent 每次调用前设置）。"""
    _CURRENT_USER_VAR.set(user_id)


def get_current_user() -> str:
    """获取当前用户 ID。"""
    return _CURRENT_USER_VAR.get()


def set_current_kb_ids(kb_ids: list[str]):
    """设置当前可访问的知识库 ID 列表。"""
    _CURRENT_KB_IDS_VAR.set(kb_ids)


def get_current_kb_ids() -> Optional[list[str]]:
    """获取当前可访问的知识库 ID 列表。"""
    return _CURRENT_KB_IDS_VAR.get()


# Re-exported from events.py (shared module, no circular import)
from events import set_rag_step_queue, emit_rag_step


@tool("get_current_weather")
def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取天气信息"""
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    guard = _TOOL_CALLS_GUARD.get()
    if guard is None:
        guard = {"count": 0}
        _TOOL_CALLS_GUARD.set(guard)
    if guard["count"] >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base 工具本轮已调用一次，"
            "请使用现有检索结果直接生成最终回答。"
        )
    guard["count"] += 1

    from agentic_rag.runner import run_agentic_rag_sync, format_rag_result
    # search_knowledge_base 是 sync 函数，LangChain agent.astream()
    # 会在线程池中执行它，不会阻塞 asyncio 事件循环。
    # emit_rag_step → call_soon_threadsafe 从子线程安全投递事件。
    kb_ids = get_current_kb_ids()
    user_id = get_current_user()
    rag_result = run_agentic_rag_sync(query, user_id=user_id, kb_ids=kb_ids)
    result = format_rag_result(rag_result)

    docs = result.get("docs", [])
    rag_trace = result.get("rag_trace", {})
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})
        # 传播 RAG 上下文到主事件循环：线程池中 ContextVar 不会传播到父异步任务，
        # 需要通过 call_soon_threadsafe 将上下文推入 SSE 队列。
        try:
            from events import get_rag_step_queue
            _q = get_rag_step_queue()
            _loop = getattr(_q, '_loop', None)
            if _q is not None and _loop is not None and not _loop.is_closed():
                _loop.call_soon_threadsafe(
                    _q.put_nowait, {"type": "rag_context", "context": {"rag_trace": rag_trace}}
                )
        except Exception as e:
            logger.warning("Failed to propagate rag_context to SSE queue: %s", e)

    if not docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)
