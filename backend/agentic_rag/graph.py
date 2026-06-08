"""Agentic RAG 主图 — 基于 LangGraph StateGraph 的完整 RAG 管线。

图结构:
```
START → guardrail_query
  ├── blocked → direct_answer → guardrail_output → post_process → END
  └── ok → fast_path
       ├── no_retrieval → direct_answer → guardrail_output → post_process → END
       └── retrieval → inject_memory → decide_retrieval → plan_retrieval
            → transform_query → rewrite_expand → hybrid_retrieve → progressive_rerank
            → grade_documents
              ├── yes → compress_check → generate_answer → evaluate_confidence
              │    ├── answer → guardrail_output → post_process → END
              │    ├── partial → guardrail_output → post_process → END
              │    └── retry → budget_check
              │         ├── ok → rewrite_expand (loop)
              │         └── exhausted → handle_exhaustion → guardrail_output → post_process → END
              └── no → budget_check
                   ├── ok → rewrite_expand (loop)
                   └── exhausted → handle_exhaustion → guardrail_output → post_process → END
```
"""

import threading

from langgraph.graph import StateGraph, END

from agentic_rag.state import AgenticRAGState


# ═══ 条件路由函数 ═══


def route_guardrail(state: AgenticRAGState) -> str:
    """护栏路由。"""
    if state.get("_query_blocked"):
        return "direct_answer"
    return "fast_path"


def route_fast_path(state: AgenticRAGState) -> str:
    """快速路径路由。"""
    if not state.get("needs_retrieval", True):
        return "direct_answer"
    return "inject_memory"


def route_grade(state: AgenticRAGState) -> str:
    """文档评估路由。"""
    grade = state.get("relevance_grade")
    if grade in ("yes", "partial"):
        return "compress_check"
    return "budget_check"


def route_confidence(state: AgenticRAGState) -> str:
    """置信评估路由。"""
    ca = state.get("confidence_assessment")
    if isinstance(ca, dict):
        level = ca.get("level", "retry")
    else:
        level = "retry"

    if level == "retry":
        return "budget_check"
    return "guardrail_output"  # answer / partial → guardrail_output


def route_rewrite(state: AgenticRAGState) -> str:
    """改写节点路由：策略穷尽时跳过检索，直接进入预算检查。"""
    if state.get("budget_exhausted"):
        return "budget_check"
    return "hybrid_retrieve"


def route_budget(state: AgenticRAGState) -> str:
    """预算路由：耗尽 → handle_exhaustion，否则 → rewrite_expand。"""
    if state.get("budget_exhausted"):
        return "handle_exhaustion"
    return "rewrite_expand"


# ═══ 图构建 ═══


def build_agentic_rag_graph() -> StateGraph:
    """构建完整的 Agentic RAG StateGraph。

    节点函数是懒导入的，避免模块级触发重度依赖（如 HuggingFace Hub 下载）。
    """
    # ── 懒导入节点函数 ──
    from agentic_rag.nodes.guardrail_query_node import guardrail_query
    from agentic_rag.nodes.fast_path import fast_path
    from agentic_rag.nodes.direct_answer import direct_answer
    from agentic_rag.nodes.inject_memory import inject_memory
    from agentic_rag.nodes.decide_retrieval import decide_retrieval
    from agentic_rag.nodes.navigate_tree import route_documents
    from agentic_rag.nodes.plan_retrieval import plan_retrieval
    from agentic_rag.nodes.transform_query import transform_query
    from agentic_rag.nodes.rewrite_expand import rewrite_expand
    from agentic_rag.nodes.hybrid_retrieve import hybrid_retrieve
    from agentic_rag.nodes.progressive_rerank import progressive_rerank
    from agentic_rag.nodes.grade_documents import grade_documents
    from agentic_rag.nodes.compress_check import compress_check
    from agentic_rag.nodes.generate_answer import generate_answer
    from agentic_rag.nodes.evaluate_confidence import evaluate_confidence
    from agentic_rag.nodes.budget_check import budget_check
    from agentic_rag.nodes.handle_exhaustion import handle_exhaustion
    from agentic_rag.nodes.guardrail_output import guardrail_output
    from agentic_rag.nodes.post_process import post_process

    graph = StateGraph(AgenticRAGState)

    # ── 添加节点 ──
    graph.add_node("guardrail_query", guardrail_query)
    graph.add_node("fast_path", fast_path)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("inject_memory", inject_memory)
    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("route_documents", route_documents)
    graph.add_node("plan_retrieval", plan_retrieval)
    graph.add_node("transform_query", transform_query)
    graph.add_node("rewrite_expand", rewrite_expand)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("progressive_rerank", progressive_rerank)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("compress_check", compress_check)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("evaluate_confidence", evaluate_confidence)
    graph.add_node("budget_check", budget_check)
    graph.add_node("handle_exhaustion", handle_exhaustion)
    graph.add_node("guardrail_output", guardrail_output)
    graph.add_node("post_process", post_process)

    # ── 入口 ──
    graph.set_entry_point("guardrail_query")

    # ── 顺序边 ──
    graph.add_edge("inject_memory", "decide_retrieval")
    graph.add_edge("decide_retrieval", "route_documents")
    graph.add_edge("route_documents", "plan_retrieval")
    graph.add_edge("plan_retrieval", "transform_query")
    graph.add_edge("transform_query", "rewrite_expand")
    graph.add_conditional_edges(
        "rewrite_expand",
        route_rewrite,
        {"hybrid_retrieve": "hybrid_retrieve", "budget_check": "budget_check"},
    )
    graph.add_edge("hybrid_retrieve", "progressive_rerank")
    graph.add_edge("progressive_rerank", "grade_documents")
    graph.add_edge("compress_check", "generate_answer")
    graph.add_edge("generate_answer", "evaluate_confidence")
    graph.add_edge("direct_answer", "guardrail_output")
    graph.add_edge("handle_exhaustion", "guardrail_output")
    graph.add_edge("guardrail_output", "post_process")
    graph.add_edge("post_process", END)

    # ── 条件边 ──
    graph.add_conditional_edges(
        "guardrail_query",
        route_guardrail,
        {"direct_answer": "direct_answer", "fast_path": "fast_path"},
    )
    graph.add_conditional_edges(
        "fast_path",
        route_fast_path,
        {"direct_answer": "direct_answer", "inject_memory": "inject_memory"},
    )
    graph.add_conditional_edges(
        "grade_documents",
        route_grade,
        {"compress_check": "compress_check", "budget_check": "budget_check"},
    )
    graph.add_conditional_edges(
        "evaluate_confidence",
        route_confidence,
        {"guardrail_output": "guardrail_output", "budget_check": "budget_check"},
    )
    graph.add_conditional_edges(
        "budget_check",
        route_budget,
        {"rewrite_expand": "rewrite_expand", "handle_exhaustion": "handle_exhaustion"},
    )

    return graph.compile()


# ═══ 全局实例（懒加载，线程安全） ═══
_agentic_rag_graph = None
_graph_lock = threading.Lock()


def get_agentic_rag_graph():
    """获取全局 Agentic RAG 图实例（懒加载，线程安全）。"""
    global _agentic_rag_graph
    if _agentic_rag_graph is None:
        with _graph_lock:
            if _agentic_rag_graph is None:
                _agentic_rag_graph = build_agentic_rag_graph()
    return _agentic_rag_graph
