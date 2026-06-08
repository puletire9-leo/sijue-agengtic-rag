# Agentic RAG — 18 节点 LangGraph 状态机

## 图结构

```
START → guardrail_query
  ├─ blocked → direct_answer → guardrail_output → post_process → END
  └─ ok → fast_path
       ├─ no_retrieval → direct_answer → ...
       └─ retrieval → inject_memory → decide_retrieval → route_documents
            → plan_retrieval → transform_query → rewrite_expand
            → hybrid_retrieve → progressive_rerank → grade_documents
              ├─ yes → compress_check → generate_answer → evaluate_confidence
              │    ├─ answer/partial → guardrail_output → post_process → END
              │    └─ retry → budget_check
              │         ├─ ok → rewrite_expand (循环)
              │         └─ exhausted → handle_exhaustion → ...
              └─ no → budget_check
```

## 节点职责

| 节点 | 文件 | 功能 |
|------|------|------|
| guardrail_query | `nodes/guardrail_query_node.py` | 输入安全检查 |
| fast_path | `nodes/fast_path.py` | 快速路径判断（跳过检索）|
| inject_memory | `nodes/inject_memory.py` | 3 层记忆注入 |
| decide_retrieval | `nodes/decide_retrieval.py` | 检索策略选择 |
| route_documents | `nodes/navigate_tree.py` | 文档路由（章节过滤）|
| plan_retrieval | `nodes/plan_retrieval.py` | 检索参数规划 |
| transform_query | `nodes/transform_query.py` | 查询变换 |
| rewrite_expand | `nodes/rewrite_expand.py` | 查询重写（Step-Back/HyDE/Complex）|
| hybrid_retrieve | `nodes/hybrid_retrieve.py` | 混合检索 + 语义缓存 |
| progressive_rerank | `nodes/progressive_rerank.py` | 3 阶段精排 |
| grade_documents | `nodes/grade_documents.py` | 文档相关性评估 |
| compress_check | `nodes/compress_check.py` | 压缩检查 |
| generate_answer | `nodes/generate_answer.py` | 生成回答 + 缓存写入 |
| evaluate_confidence | `nodes/evaluate_confidence.py` | 置信评估 |
| budget_check | `nodes/budget_check.py` | 预算检查 |
| handle_exhaustion | `nodes/handle_exhaustion.py` | 预算耗尽处理 |
| guardrail_output | `nodes/guardrail_output.py` | 输出安全检查 |
| post_process | `nodes/post_process.py` | 后处理（追踪、记忆、成本）|

## 关键文件

- `config.py` — 7 个 Pydantic 配置类（budget/compression/retrieval/confidence/memory/guardrail/agent）
- `state.py` — AgenticRAGState（40+ 字段 TypedDict）
- `llm.py` — 3 层 LLM 路由（TrackedLLM + tenacity 重试）
- `graph.py` — StateGraph 构建 + 线程安全单例
- `runner.py` — 同步/异步执行入口
- `retrieval_v2.py` — 子问题分解、实体提取、渐进回填
- `prompt_builder.py` — Prompt 构建
- `token_estimator.py` — Token 估算

## 语义缓存

`hybrid_retrieve` 检索前检查语义缓存（cosine > 0.95），命中则跳过检索。
`generate_answer` 生成后将答案写入缓存。
实现：`core/semantic_cache.py`（基于 Milvus）

## 条件路由

5 个条件边：`route_guardrail`, `route_fast_path`, `route_grade`, `route_confidence`, `route_budget`
