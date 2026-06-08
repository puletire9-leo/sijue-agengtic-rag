"""Agentic RAG 完整状态定义 v4.7。

所有字段标注生命周期:
  - @producer: 写入该字段的节点
  - @consumer: 读取该字段的节点
"""

from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgenticRAGState(TypedDict):
    """Agentic RAG 完整状态定义 v4.7。

    相比 v4.6 的变更:
    - confidence + confidence_result → confidence_assessment (ConfidenceAssessment)
    - iteration_budget (dict) → iteration_budget (Budget value object 序列化)
    - 所有字段增加 @producer / @consumer 生命周期标注
    - 新增 is_degraded_answer 降级标记
    """

    # ═══ 输入 ═══
    # @producer: 外部调用者 (invoke)
    # @consumer: fast_path, inject_memory, generate_answer
    question: str

    # @producer: 外部调用者
    # @consumer: post_process, checkpointer
    session_id: str

    # @producer: 外部调用者
    # @consumer: inject_memory, post_process, guardrail_query
    user_id: str

    # @producer: 外部调用者
    # @consumer: hybrid_retrieve（按知识库过滤检索结果）
    kb_ids: List[str]

    # ═══ 消息列表（add_messages Reducer）═══
    # @producer: 外部调用者, generate_answer, compress_check
    # @consumer: 所有需要对话上下文的节点
    messages: Annotated[List[BaseMessage], add_messages]

    # ═══ 快速路径 ═══
    # @producer: fast_path
    # @consumer: direct_answer
    fast_path_result: Optional[Dict]

    # ═══ 记忆系统 ═══
    # @producer: inject_memory
    # @consumer: generate_answer, post_process
    memory_context: Optional[str]

    # @producer: inject_memory
    # @consumer: post_process
    session_memory: List[Dict]

    # @producer: inject_memory
    # @consumer: post_process (追踪用)
    memory_sources: List[str]

    # ═══ 检索决策 ═══
    # @producer: fast_path
    # @consumer: decide_retrieval, plan_retrieval
    needs_retrieval: bool

    # @producer: fast_path
    # @consumer: decide_retrieval
    retrieval_confidence: float

    # @producer: fast_path
    # @consumer: trace_steps (追踪用)
    decision_source: str

    # ═══ 检索策略选择（v4.7 新增: decide_retrieval → plan_retrieval）═══
    # @producer: decide_retrieval
    # @consumer: plan_retrieval, hybrid_retrieve, progressive_rerank
    retrieval_strategy: Optional[str]

    # @producer: decide_retrieval
    # @consumer: trace_steps (追踪用)
    retrieval_strategy_reason: Optional[str]

    # @producer: decide_retrieval
    # @consumer: plan_retrieval
    retrieval_strategy_confidence: float

    # @producer: plan_retrieval
    # @consumer: hybrid_retrieve, progressive_rerank, rewrite_expand
    retrieval_plan: Optional[Dict]

    # ═══ 智能路由 (v4.9) ═══
    # @producer: route_documents
    # @consumer: hybrid_retrieve (章节级过滤)
    routed_chapters: Optional[List[str]]

    # @producer: route_documents
    # @consumer: hybrid_retrieve, trace_steps
    route_applied: bool

    # @producer: route_documents
    # @consumer: grade_documents (语义回退判断)
    search_target: Optional[str]

    # @producer: route_documents
    # @consumer: hybrid_retrieve (路由失败时作为元数据过滤兜底)
    route_fallback_docs: Optional[List[str]]

    # ═══ 渐进式重排（v4.7 新增: progressive_rerank）═══
    # @producer: progressive_rerank
    # @consumer: grade_documents, generate_answer, post_process
    reranked_docs: List[Dict]

    # @producer: progressive_rerank
    # @consumer: post_process (追踪用)
    rerank_trace: Dict

    # ═══ 查询改写 ═══
    # @producer: transform_query, rewrite_expand
    # @consumer: hybrid_retrieve, rerank, grade_documents
    query: str

    # @producer: rewrite_expand
    # @consumer: trace_steps (追踪用)
    rewrite_strategy: Optional[str]

    # @producer: rewrite_expand
    # @consumer: rewrite_expand (策略轮换去重)
    tried_strategies: Optional[List[str]]

    # @producer: transform_query
    # @consumer: rewrite_expand (对比用)
    original_question: Optional[str]

    # ═══ v2: 子问题分解 ═══
    # @producer: rewrite_expand
    # @consumer: hybrid_retrieve
    _sub_queries: Optional[List[str]]

    # @producer: hybrid_retrieve
    # @consumer: post_process (情节记忆记录)
    _retrieved_filenames: Optional[List[str]]

    # ═══ 检索结果 ═══
    # @producer: hybrid_retrieve
    # @consumer: progressive_rerank, grade_documents, generate_answer, evaluate_confidence
    retrieved_docs: List[Dict]

    # @producer: hybrid_retrieve
    # @consumer: compress_check, trace_steps
    retrieval_metadata: Dict

    # @producer: hybrid_retrieve
    # @consumer: compress_check
    _needs_compression: bool

    # ═══ 门控评估 ═══
    # @producer: grade_documents
    # @consumer: generate_answer, rewrite_expand (路由决策)
    relevance_grade: Optional[str]

    # @producer: grade_documents
    # @consumer: generate_answer
    filtered_docs: List[Dict]

    # @producer: rewrite_expand
    # @consumer: rewrite_expand (循环控制), budget_check
    retry_count: int

    # @producer: grade_documents (覆盖检查失败时递增)
    # @consumer: grade_documents (限制覆盖重试次数)
    coverage_retry_count: int

    # ═══ 迭代预算（Budget value object 序列化）═══
    # @producer: init_budget
    # @consumer: budget_check, rewrite_expand, evaluate_confidence
    iteration_budget: Dict  # {"max": N, "used": N, "grace": N}

    # @producer: budget_check
    # @consumer: 条件边路由
    budget_exhausted: bool

    # ═══ 后台审查 ═══
    # @producer: post_process
    # @consumer: post_process (周期判断)
    turns_since_review: int

    # @producer: post_process
    # @consumer: 外部审查系统
    review_triggered: bool

    # ═══ 压缩后消息（L-01 修复: 避免 add_messages 追加导致上下文翻倍）═══
    # @producer: compress_check
    # @consumer: generate_answer
    compressed_messages: Optional[list]

    # ═══ 生成 ═══
    # @producer: generate_answer
    # @consumer: guardrail_output, evaluate_confidence, post_process
    answer: Optional[str]

    # @producer: generate_answer
    # @consumer: guardrail_output, post_process
    citations: List[Dict]

    # @producer: evaluate_confidence
    # @consumer: 条件边路由, post_process, handle_exhaustion
    confidence_assessment: Optional[Dict]  # 序列化的 ConfidenceAssessment

    # ═══ 降级标记（v4.7 新增）═══
    # @producer: handle_exhaustion
    # @consumer: guardrail_output, post_process, 外部调用者
    is_degraded_answer: bool

    # ═══ 护栏 ═══
    # @producer: guardrail_query
    # @consumer: 条件边路由
    _query_blocked: bool

    # @producer: guardrail_query
    # @consumer: direct_answer (展示原因)
    _block_reason: Optional[str]

    # @producer: guardrail_output
    # @consumer: post_process (审计日志)
    _redactions: List[str]

    # ═══ 追踪 ═══
    # @producer: 所有节点
    # @consumer: post_process, 外部监控系统
    trace_steps: List[Dict]

    # @producer: post_process
    # @consumer: 外部监控系统
    latency_ms: int

    # @producer: post_process
    # @consumer: 外部计费系统
    cost_usd: float


def create_initial_state(
    question: str,
    session_id: str = "default",
    user_id: str = "default",
    max_iterations: int = None,
    kb_ids: list[str] | None = None,
) -> dict:
    """创建初始状态，提供所有必需字段的默认值。

    Args:
        question: 用户问题
        session_id: 会话 ID
        user_id: 用户 ID
        max_iterations: 预算上限（默认从 BudgetConfig 读取，通常为 5）
        kb_ids: 可访问的知识库 ID 列表

    Returns:
        包含所有 AgenticRAGState 字段的 dict
    """
    from agentic_rag.schemas import Budget
    from agentic_rag.config import budget as budget_cfg

    if max_iterations is None:
        max_iterations = budget_cfg.MAX_ITERATIONS

    budget = Budget(max=max_iterations, used=0, grace=budget_cfg.GRACE_ITERATIONS)

    return {
        "question": question,
        "session_id": session_id,
        "user_id": user_id,
        "kb_ids": kb_ids or [],
        "messages": [],
        "fast_path_result": None,
        "memory_context": None,
        "session_memory": [],
        "memory_sources": [],
        "needs_retrieval": True,
        "retrieval_confidence": 1.0,
        "decision_source": "initial",
        "retrieval_strategy": None,
        "retrieval_strategy_reason": None,
        "retrieval_strategy_confidence": 0.5,
        "retrieval_plan": None,
        "routed_chapters": [],
        "route_applied": False,
        "search_target": "",
        "route_fallback_docs": [],
        "reranked_docs": [],
        "rerank_trace": {},
        "query": question,
        "rewrite_strategy": None,
        "tried_strategies": [],
        "original_question": None,
        "_sub_queries": None,
        "_retrieved_filenames": None,
        "retrieved_docs": [],
        "retrieval_metadata": {},
        "_needs_compression": False,
        "relevance_grade": None,
        "filtered_docs": [],
        "retry_count": 0,
        "coverage_retry_count": 0,
        "iteration_budget": budget.to_dict(),
        "budget_exhausted": False,
        "turns_since_review": 0,
        "review_triggered": False,
        "compressed_messages": None,
        "answer": None,
        "citations": [],
        "confidence_assessment": None,
        "is_degraded_answer": False,
        "_query_blocked": False,
        "_block_reason": None,
        "_redactions": [],
        "trace_steps": [],
        "latency_ms": 0,
        "cost_usd": 0.0,
    }


def get_required_state_fields() -> List[str]:
    """返回 AgenticRAGState 所有必需字段名（不含 Optional 字段）。"""
    return [
        "question",
        "session_id",
        "user_id",
        "messages",
        "needs_retrieval",
        "retrieval_confidence",
        "decision_source",
        "retrieval_strategy",
        "retrieval_strategy_reason",
        "retrieval_strategy_confidence",
        "retrieval_plan",
        "query",
        "retrieved_docs",
        "reranked_docs",
        "rerank_trace",
        "retrieval_metadata",
        "_needs_compression",
        "filtered_docs",
        "retry_count",
        "iteration_budget",
        "budget_exhausted",
        "turns_since_review",
        "review_triggered",
        "answer",
        "citations",
        "is_degraded_answer",
        "_query_blocked",
        "_redactions",
        "trace_steps",
        "latency_ms",
        "cost_usd",
        "session_memory",
        "memory_sources",
    ]
