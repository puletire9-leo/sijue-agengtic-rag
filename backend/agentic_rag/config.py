"""Agentic RAG 配置 — 所有阈值和常量收敛到此文件，允许环境变量覆盖。"""

import logging
import os

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field


class BudgetConfig(BaseModel):
    """预算配置"""
    MAX_ITERATIONS: int = Field(default=5, ge=1, le=20, description="最大迭代次数")
    MAX_RETRY_ITERATIONS: int = Field(default=3, ge=1, le=10, description="最大重试次数")
    GRACE_ITERATIONS: int = Field(default=1, ge=0, le=5, description="宽限迭代次数")
    WARNING_THRESHOLD: int = Field(default=4, ge=1, description="警告阈值")


class CompressionConfig(BaseModel):
    """上下文压缩配置"""
    MODEL_CONTEXT_WINDOW: int = Field(default=128000, ge=1024, description="模型上下文窗口大小")
    COMPRESSION_THRESHOLD: int = Field(default=64000, ge=1024, description="压缩触发阈值")
    KEEP_RECENT_MESSAGES: int = Field(default=10, ge=1, le=50, description="保留最近消息数")
    TARGET_MIDDLE_TOKENS: int = Field(default=15000, ge=1000, le=100000, description="中间摘要目标 token 数")


class RetrievalConfig(BaseModel):
    """检索配置"""
    RETRIEVE_TOP_K: int = Field(default=5, ge=1, le=50, description="检索返回文档数")
    CANDIDATE_MULTIPLIER: int = Field(default=3, ge=1, le=10, description="候选倍数")
    RERANK_ENABLED: bool = Field(default=True, description="是否启用重排")
    RERANK_TOP_K_MULTIPLIER: int = Field(default=3, ge=1, le=10, description="重排 top-k 倍数")
    ENTITY_BOOST_FACTOR: float = Field(default=1.15, ge=1.0, le=2.0, description="实体加权因子")
    RERANK_TIMEOUT: int = Field(default=5, ge=1, le=60, description="重排超时秒数")
    MAX_SUBQUERY_CONCURRENCY: int = Field(default=3, ge=1, le=10, description="子查询最大并发数")


class ConfidenceConfig(BaseModel):
    """置信评估配置"""
    CONFIDENCE_ANSWER_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0, description="回答置信阈值")
    SUFFICIENCY_ANSWER_TRUNCATE: int = Field(default=2000, ge=100, le=10000, description="充分性截断长度")


class MemoryConfig(BaseModel):
    """记忆系统配置"""
    SESSION_TTL: int = Field(default=86400, ge=60, le=604800, description="会话记忆 TTL 秒数")
    SHORT_TERM_DAYS: int = Field(default=30, ge=1, le=365, description="短期记忆保留天数")
    LONG_TERM_ENABLED: bool = Field(default=True, description="是否启用长期记忆")


class GuardrailConfig(BaseModel):
    """护栏配置"""
    QUERY_ENABLED: bool = Field(default=False, description="是否启用查询护栏")
    OUTPUT_ENABLED: bool = Field(default=False, description="是否启用输出护栏")
    MAX_QUERY_LENGTH: int = Field(default=100000, ge=100, le=100000, description="最大查询长度")
    LLM_ENABLED: bool = Field(default=False, description="是否启用 LLM 护栏")


class AgentConfig(BaseModel):
    """Agent 层配置"""
    MODEL_CONTEXT_WINDOW: int = Field(default=128000, ge=1024, description="模型上下文窗口大小")
    AGENT_MAX_ITERATIONS: int = Field(default=90, ge=10, le=500, description="Agent 最大迭代次数")
    COMPRESSION_THRESHOLD_AGENT: int = Field(default=64000, ge=1024, description="Agent 压缩触发阈值")


def _build_config(cls, overrides: dict = None):
    """从环境变量构建配置实例。"""
    env_map = {
        BudgetConfig: {
            "MAX_ITERATIONS": "MAX_ITERATIONS",
            "MAX_RETRY_ITERATIONS": "MAX_RETRY_ITERATIONS",
            "GRACE_ITERATIONS": "GRACE_ITERATIONS",
            "WARNING_THRESHOLD": "BUDGET_WARNING_THRESHOLD",
        },
        CompressionConfig: {
            "MODEL_CONTEXT_WINDOW": "MODEL_CONTEXT_WINDOW",
            "COMPRESSION_THRESHOLD": "COMPRESSION_THRESHOLD",
            "KEEP_RECENT_MESSAGES": "KEEP_RECENT_MESSAGES",
            "TARGET_MIDDLE_TOKENS": "TARGET_MIDDLE_TOKENS",
        },
        RetrievalConfig: {
            "RETRIEVE_TOP_K": "RETRIEVE_TOP_K",
            "CANDIDATE_MULTIPLIER": "CANDIDATE_MULTIPLIER",
            "RERANK_ENABLED": "RERANK_ENABLED",
            "RERANK_TOP_K_MULTIPLIER": "RERANK_TOP_K_MULTIPLIER",
            "ENTITY_BOOST_FACTOR": "ENTITY_BOOST_FACTOR",
            "RERANK_TIMEOUT": "RERANK_TIMEOUT",
            "MAX_SUBQUERY_CONCURRENCY": "MAX_SUBQUERY_CONCURRENCY",
        },
        ConfidenceConfig: {
            "CONFIDENCE_ANSWER_THRESHOLD": "CONFIDENCE_ANSWER_THRESHOLD",
            "SUFFICIENCY_ANSWER_TRUNCATE": "SUFFICIENCY_ANSWER_TRUNCATE",
        },
        MemoryConfig: {
            "SESSION_TTL": "MEMORY_SESSION_TTL",
            "SHORT_TERM_DAYS": "MEMORY_SHORT_TERM_DAYS",
            "LONG_TERM_ENABLED": "MEMORY_LONG_TERM_ENABLED",
        },
        GuardrailConfig: {
            "QUERY_ENABLED": "GUARDRAIL_QUERY_ENABLED",
            "OUTPUT_ENABLED": "GUARDRAIL_OUTPUT_ENABLED",
            "MAX_QUERY_LENGTH": "GUARDRAIL_MAX_QUERY_LENGTH",
            "LLM_ENABLED": "GUARDRAIL_LLM_ENABLED",
        },
        AgentConfig: {
            "MODEL_CONTEXT_WINDOW": "MODEL_CONTEXT_WINDOW",
            "AGENT_MAX_ITERATIONS": "AGENT_MAX_ITERATIONS",
            "COMPRESSION_THRESHOLD_AGENT": "COMPRESSION_THRESHOLD_AGENT",
        },
    }

    values = {}
    mapping = env_map.get(cls, {})
    for field_name, env_key in mapping.items():
        raw = os.getenv(env_key)
        if raw is not None:
            # 类型转换
            field_info = cls.model_fields.get(field_name)
            if field_info:
                ft = field_info.annotation
                if ft is bool:
                    values[field_name] = raw.lower() == "true"
                elif ft is int:
                    try:
                        values[field_name] = int(raw)
                    except ValueError:
                        logger.warning("Invalid int for %s=%r, using default", env_key, raw)
                elif ft is float:
                    try:
                        values[field_name] = float(raw)
                    except ValueError:
                        logger.warning("Invalid float for %s=%r, using default", env_key, raw)
                else:
                    values[field_name] = raw

    if overrides:
        values.update(overrides)

    return cls(**values)


def validate_model_config():
    """Validate that lightweight models are configured separately in production."""
    main_model = os.getenv("MODEL", "")
    grade_model = os.getenv("GRADE_MODEL", "")
    fast_model = os.getenv("FAST_MODEL", "")

    warnings = []
    if not grade_model:
        warnings.append("GRADE_MODEL not set — will fall back to main model (higher cost)")
    elif grade_model == main_model:
        warnings.append("GRADE_MODEL equals MODEL — consider using a lighter model for grading")

    if not fast_model:
        warnings.append("FAST_MODEL not set — will fall back to main model (higher cost)")
    elif fast_model == main_model:
        warnings.append("FAST_MODEL equals MODEL — consider using a lighter model for fast decisions")

    return warnings


# 便捷引用（从环境变量构建，支持 Pydantic 校验）
budget = _build_config(BudgetConfig)
compression = _build_config(CompressionConfig)
retrieval = _build_config(RetrievalConfig)
confidence = _build_config(ConfidenceConfig)
memory = _build_config(MemoryConfig)
guardrail = _build_config(GuardrailConfig)
agent = _build_config(AgentConfig)
