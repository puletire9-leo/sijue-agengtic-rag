"""结构化日志 — JSON 格式 + request_id 全链路串联。

使用 python-json-logger（已在 pyproject.toml 依赖中）实现 JSON 格式日志，
每个请求生成唯一 request_id，贯穿 HTTP → RAG 管线 → LLM 调用。
"""

import logging
import uuid
from contextvars import ContextVar

# ContextVar 用于在异步/同步代码间传递 request_id
_request_id: ContextVar[str] = ContextVar("request_id", default="")
_user_id_ctx: ContextVar[str] = ContextVar("user_id", default="")


def get_request_id() -> str:
    """获取当前请求的 request_id。"""
    return _request_id.get("")


def set_request_id(request_id: str):
    """设置当前请求的 request_id。"""
    _request_id.set(request_id)


def get_user_id_ctx() -> str:
    """获取当前请求的 user_id。"""
    return _user_id_ctx.get("")


def set_user_id_ctx(user_id: str):
    """设置当前请求的 user_id。"""
    _user_id_ctx.set(user_id)


def generate_request_id() -> str:
    """生成新的 request_id。"""
    return uuid.uuid4().hex[:16]


class StructuredFormatter(logging.Formatter):
    """JSON 结构化日志格式化器。

    不依赖 python-json-logger，直接用 json.dumps 输出。
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
            "user_id": get_user_id_ctx(),
            "service": "supermew",
        }

        # 附加异常信息
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        # 附加额外字段
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_structured_logging(level: int = logging.INFO):
    """配置结构化日志。

    调用后，所有日志输出为 JSON 格式。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # 降低第三方库日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)


def request_id_middleware_factory():
    """创建 FastAPI request_id 中间件。

    用法:
        from core.structured_logging import request_id_middleware_factory
        middleware = request_id_middleware_factory()
        app.middleware("http")(middleware)
    """
    async def request_id_middleware(request, call_next):
        # 从请求头获取或生成 request_id
        rid = request.headers.get("x-request-id", "")
        if not rid:
            rid = generate_request_id()

        set_request_id(rid)

        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    return request_id_middleware
