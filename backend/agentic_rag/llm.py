"""LLM 工厂 — 3 级模型路由。

模型分层:
  Tier 1 (强力):  MODEL → generate_answer, direct_answer
  Tier 2 (中等):  GRADE_MODEL → grade_documents（相关性评估）
  Tier 3 (轻量):  FAST_MODEL → fast_path, decide_retrieval, rewrite_expand,
                               evaluate_confidence, context_compressor

未配置时自动降级: GRADE_MODEL → MODEL, FAST_MODEL → MODEL
"""

import logging
import os
import threading
from typing import Optional

from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
FAST_MODEL = os.getenv("FAST_MODEL")
GRADE_MODEL = os.getenv("GRADE_MODEL") or MODEL


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "LLM call failed (attempt %d), retrying: %s",
        retry_state.attempt_number,
        exc or "unknown",
    )


def _should_retry(retry_state: RetryCallState) -> bool:
    if not retry_state.outcome or not retry_state.outcome.failed:
        return False
    return TrackedLLM._is_retryable(retry_state.outcome.exception())


_llm: Optional[BaseChatModel] = None
_lightweight_llm: Optional[BaseChatModel] = None
_grader_llm: Optional[BaseChatModel] = None
_llm_lock = threading.Lock()
_lightweight_lock = threading.Lock()
_grader_lock = threading.Lock()


class TrackedLLM:
    """Wrapper that tracks LLM token usage and cost."""

    # Approximate cost per 1K tokens (input/output)
    COST_MAP = {
        "deepseek-chat": (0.0014, 0.0028),
        "deepseek-reasoner": (0.0056, 0.021),
        "gpt-4.1": (0.02, 0.08),
        "gpt-4.1-mini": (0.004, 0.016),
    }

    def __init__(self, llm, model_name: str, tier: str):
        self._llm = llm
        self._model_name = model_name
        self._tier = tier

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """判断异常是否可重试（网络/限流/服务端错误）。"""
        try:
            import openai
            if isinstance(exc, openai.APIStatusError):
                return exc.status_code == 429 or exc.status_code >= 500
            if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
                return True
        except ImportError:
            pass
        return False

    @retry(
        retry=_should_retry,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    def invoke(self, *args, **kwargs):
        import time
        start = time.time()
        result = self._llm.invoke(*args, **kwargs)
        duration = time.time() - start

        # Track usage
        usage = getattr(result, 'usage_metadata', None) or {}
        prompt_tokens = usage.get('input_tokens') or usage.get('prompt_tokens') or 0
        completion_tokens = usage.get('output_tokens') or usage.get('completion_tokens') or 0
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)

        # Estimate cost
        cost_input, cost_output = self.COST_MAP.get(self._model_name, (0.001, 0.002))
        estimated_cost = (prompt_tokens * cost_input + completion_tokens * cost_output) / 1000

        logger.info(
            f"LLM [{self._tier}] model={self._model_name} "
            f"tokens={total_tokens} (in={prompt_tokens}, out={completion_tokens}) "
            f"cost=${estimated_cost:.6f} duration={duration:.2f}s"
        )
        return result

    # Delegate other attributes
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._llm, name)


def log_model_cost_info():
    """Log model routing and cost info at startup."""
    from agentic_rag.config import validate_model_config
    warnings = validate_model_config()
    for w in warnings:
        logger.warning(f"[ModelConfig] {w}")


def get_llm() -> Optional[BaseChatModel]:
    """Tier 1 — 强力模型（生成回答、直接回答）。

    使用 MODEL 环境变量，通常为 doubao-seed-2-0-pro。
    """
    global _llm
    if _llm is None and API_KEY and MODEL:
        with _llm_lock:
            if _llm is None:
                raw = init_chat_model(
                    model=MODEL, model_provider="openai",
                    api_key=API_KEY, base_url=BASE_URL,
                    temperature=0.3, stream_usage=True, max_retries=2,
                )
                _llm = TrackedLLM(raw, MODEL, "T1")
    return _llm


def get_lightweight_llm() -> Optional[BaseChatModel]:
    """Tier 3 — 轻量模型（分类、评估、改写、压缩）。

    优先使用 FAST_MODEL（如 doubao-seed-1-6-flash），
    未配置时降级到 MODEL。
    """
    global _lightweight_llm
    if _lightweight_llm is None and API_KEY:
        with _lightweight_lock:
            if _lightweight_llm is None:
                model = FAST_MODEL or MODEL
                if model:
                    raw = init_chat_model(
                        model=model, model_provider="openai",
                        api_key=API_KEY, base_url=BASE_URL,
                        temperature=0, stream_usage=True, max_retries=2,
                    )
                    _lightweight_llm = TrackedLLM(raw, model, "T3")
    return _lightweight_llm


def get_grader() -> Optional[BaseChatModel]:
    """Tier 2 — 中等模型（文档相关性评分）。

    优先使用 GRADE_MODEL（如 doubao-seed-2-0-lite），
    未配置时降级到 MODEL。
    """
    global _grader_llm
    if _grader_llm is None and API_KEY and GRADE_MODEL:
        with _grader_lock:
            if _grader_llm is None:
                raw = init_chat_model(
                    model=GRADE_MODEL, model_provider="openai",
                    api_key=API_KEY, base_url=BASE_URL,
                    temperature=0, stream_usage=True, max_retries=2,
                )
                _grader_llm = TrackedLLM(raw, GRADE_MODEL, "T2")
    return _grader_llm


def get_model_routing_info() -> dict:
    """返回当前模型路由信息（用于启动日志）。"""
    return {
        "tier1_powerful": MODEL or "未配置",
        "tier2_medium": GRADE_MODEL or f"降级→{MODEL or '未配置'}",
        "tier3_lightweight": FAST_MODEL or f"降级→{MODEL or '未配置'}",
        "effective_models": len({MODEL, FAST_MODEL or MODEL, GRADE_MODEL or MODEL}),
        "tiers_configured": sum([
            bool(MODEL),
            bool(os.getenv("GRADE_MODEL")),
            bool(os.getenv("FAST_MODEL")),
        ]),
    }


def reset_llm_cache():
    """重置 LLM 缓存（用于测试或配置变更后刷新）。"""
    global _llm, _lightweight_llm, _grader_llm
    _llm = None
    _lightweight_llm = None
    _grader_llm = None
