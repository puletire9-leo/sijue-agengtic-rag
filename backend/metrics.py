"""Prometheus metrics for SuperMew."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

# HTTP metrics
REQUEST_COUNT = Counter(
    "supermew_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "supermew_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# LLM metrics
LLM_CALLS = Counter(
    "supermew_llm_calls_total",
    "Total LLM API calls",
    ["model", "purpose"]
)
LLM_LATENCY = Histogram(
    "supermew_llm_call_duration_seconds",
    "LLM call latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)
LLM_TOKENS = Counter(
    "supermew_llm_tokens_total",
    "Total tokens consumed",
    ["model", "type"]  # type: prompt/completion
)

# RAG metrics
RAG_RETRIEVAL_COUNT = Counter(
    "supermew_rag_retrievals_total",
    "Total RAG retrievals",
    ["strategy", "result"]  # result: success/fallback/empty
)
RAG_RETRIEVAL_LATENCY = Histogram(
    "supermew_rag_retrieval_duration_seconds",
    "RAG retrieval latency"
)

# Budget metrics
BUDGET_EXHAUSTIONS = Counter(
    "supermew_budget_exhaustions_total",
    "Total budget exhaustions"
)

# Document metrics
DOCUMENT_UPLOADS = Counter(
    "supermew_document_uploads_total",
    "Total document uploads",
    ["status"]  # success/failed
)

# Memory metrics
MEMORY_RECALLS = Counter(
    "supermew_memory_recalls_total",
    "Total memory recalls",
    ["layer"]  # session/short_term/long_term
)

# Guardrail metrics
GUARDRAIL_BLOCKS = Counter(
    "supermew_guardrail_blocks_total",
    "Total guardrail blocks",
    ["type", "severity"]  # type: query/output, severity: low/high
)

# Active connections
ACTIVE_STREAMS = Gauge(
    "supermew_active_streams",
    "Number of active SSE streams"
)

# Rate limiter
RATE_LIMIT_HITS = Counter(
    "supermew_rate_limit_hits_total",
    "Total rate limit rejections",
    ["limiter"]  # sliding_window/token_bucket
)


def get_metrics():
    """Generate latest metrics in Prometheus text format."""
    return generate_latest(REGISTRY)
