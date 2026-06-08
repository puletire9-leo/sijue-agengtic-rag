# SuperMew RAG 本体功能设计

> 最后更新：2026-05-31（18 节点全通 + 4 策略真切换 + 3 阶段重排 + 3 层记忆注入 + 语义缓存）

---

## 一、RAG 层架构

RAG 层是 `search_knowledge_base` 工具的内部实现，Agent 调一次工具，图跑 18 个节点。

```
Agent 调用 search_knowledge_base(query)
  → tools.py → run_agentic_rag_sync(query)
    → agentic_rag/graph.py → 18 节点 LangGraph StateGraph
      → 返回 answer + citations + confidence + budget + rag_trace
```

技术栈：LangGraph `StateGraph` + `AgenticRAGState`（40+ 字段 TypedDict）

---

## 二、完整图结构（18 节点）

```
guardrail_query --blocked--> direct_answer -> guardrail_output -> post_process -> END
     |
     +--ok--> fast_path --no_retrieval--> direct_answer ...
                |
                +--retrieval--> inject_memory -> decide_retrieval -> plan_retrieval
                                  -> transform_query -> rewrite_expand
                                  -> hybrid_retrieve -> progressive_rerank
                                  -> grade_documents
                                    |
                                    +-- yes -> compress_check -> generate_answer
                                    |           -> evaluate_confidence
                                    |             |
                                    |             +-- answer -> guardrail_output -> post_process -> END
                                    |             +-- partial -> guardrail_output -> post_process -> END
                                    |             +-- retry -> budget_check
                                    |                  +-- ok -> rewrite_expand (loop)
                                    |                  +-- exhausted -> handle_exhaustion ...
                                    +-- no -> budget_check
                                         +-- ok -> rewrite_expand (loop)
                                         +-- exhausted -> handle_exhaustion ...
```

---

## 三、18 个节点清单

### 安全与路由

| # | 节点 | 职责 | 核心逻辑 |
|---|------|------|---------|
| 1 | `guardrail_query` | 输入安全检查 | QueryGuard 52 种危险模式，拦截 prompt 注入 |
| 2 | `fast_path` | 快速路径分类 | LLM 二分类：是否需要检索，闲聊直接回答 |
| 3 | `direct_answer` | 直接回答 | 不检索时直接生成，省 embedding + Milvus 调用 |
| 17 | `guardrail_output` | 输出安全检查 | OutputGuard 敏感词脱敏、格式校验 |
| 18 | `post_process` | 后处理 | 记忆同步、审查触发（每 10 轮）、延时/费用记录 |

### 记忆与策略

| # | 节点 | 职责 | 核心逻辑 |
|---|------|------|---------|
| 4 | `inject_memory` | 三层记忆注入 | Redis session → PG short_term → 对话偏好，合并 top 5 |
| 5 | `decide_retrieval` | 策略选择 | LLM NER+意图分类 → hybrid/dense_only/sparse_only/entity_boosted |
| 6 | `plan_retrieval` | 参数规划 | top_k、chunk_level、dense/sparse 权重、HyDE/step-back 开关、实体增强、年份/来源过滤 |

### 查询改写

| # | 节点 | 职责 | 核心逻辑 |
|---|------|------|---------|
| 7 | `transform_query` | 查询改写 | 关键词扩展、同义词对齐 |
| 8 | `rewrite_expand` | 查询扩展 | step_back / hyde / complex 轮换，消费 retrieval_plan 约束，子问题分解 |

### 检索与重排

| # | 节点 | 职责 | 核心逻辑 |
|---|------|------|---------|
| 9 | `hybrid_retrieve` | 混合检索 | Milvus dense+sparse RRF 融合，子问题并行，实体锚点，情节记忆提权，渐进回填 |
| 10 | `progressive_rerank` | 4 阶段渐进重排 | 归一化 → 加权融合 → Cross-Encoder → 上下文感知 |

### 评估与生成

| # | 节点 | 职责 | 核心逻辑 |
|---|------|------|---------|
| 11 | `grade_documents` | 相关性评估 | LLM 二分类 yes/no，不相关触发重试 |
| 12 | `compress_check` | 压缩检查 | tokens > 80000 或 docs > 8 → 五阶段压缩 |
| 13 | `generate_answer` | 生成回答 | 基于 reranked_docs + memory_context 生成 |
| 14 | `evaluate_confidence` | 置信评估 | LLM 评分 0~1 → answer/retry/partial 三路路由 |

### 预算与逃生

| # | 节点 | 职责 | 核心逻辑 |
|---|------|------|---------|
| 15 | `budget_check` | 双重预算检查 | retry_count > 3 或 Budget 耗尽 → 逃生 |
| 16 | `handle_exhaustion` | 预算逃生 | 生成降级回答，标记 is_degraded_answer=True |

---

## 四、纠错循环（双触发源）

```
触发 1: grade_documents → "no"
触发 2: evaluate_confidence → level="retry" (score < 0.3)

→ budget_check
  ├── retry_count ≤ 3 且 Budget 有剩余 → rewrite_expand（策略轮换）→ hybrid_retrieve → ...
  └── 超限 → handle_exhaustion → 降级回答
```

---

## 五、检索策略层

### 5.1 四种策略（真切换）

| 策略 | Milvus 方法 | 适用场景 | 触发条件 |
|------|-----------|---------|---------|
| `hybrid` | `hybrid_retrieve(dense+sparse+RRF)` | 大多数查询 | 默认 |
| `dense_only` | `dense_retrieve(HNSW)` | 概念解释、原理描述 | 语义型查询 |
| `sparse_only` | `sparse_retrieve(BM25)` | 精确数据、名称、日期 | 短查询+实体密集 |
| `entity_boosted` | `hybrid_retrieve` + 实体后处理 | 多命名实体查询 | 实体 ≥ 3 |

策略选择：LLM NER+意图分类（主路径）→ 正则启发式（降级）

### 5.2 策略真正切换（v3 实现）

```python
# rag_utils.py retrieve_documents() — 按策略路由到不同 Milvus 方法
if strategy == "sparse_only":
    retrieved = milvus.sparse_retrieve(sparse_embedding, ...)
elif strategy == "dense_only":
    retrieved = milvus.dense_retrieve(dense_embedding, ...)
else:
    retrieved = milvus.hybrid_retrieve(dense, sparse, ...)
```

并按需生成 embedding（sparse_only 不生成 dense，dense_only 不生成 sparse）。

### 5.3 渐进式重排（4 阶段）

```
Stage 1: Min-Max 归一化       → 统一 dense/BM25/总分到 [0,1]
Stage 2: 加权融合              → dense_weight × normed_dense + sparse_weight × normed_bm25
Stage 3: Cross-Encoder 精排    → Rerank API（BGE-Reranker / Jina），仅 rerank 一次
Stage 4: 上下文感知调整         → 同父块子块互相 +0.05/子块
```

`retrieve_documents(skip_rerank=True)` 避免重复 rerank——rerank 统一由 `progressive_rerank` 节点处理。

### 5.4 查询扩展策略

| 策略 | 功能 | 约束 |
|------|------|------|
| `step_back` | 生成退步抽象问题 + 回答，扩展查询范围 | retrieval_plan.use_step_back=True |
| `hyde` | 生成假设文档用于检索 | retrieval_plan.use_hyde=True |
| `complex` | 子问题分解（最多 4 个）+ step_back + hyde | 两者都启用时可用 |

策略受 `retrieval_plan` 约束（由 `plan_retrieval` 节点产出），重试时按允许列表轮换。

---

## 六、上下文压缩（5 阶段）

触发条件：消息 tokens > 80000，或检索文档 > 8 个（`_needs_compression` 标志）。

| 阶段 | 操作 | LLM 调用 | 降级 |
|------|------|---------|------|
| Stage 1: Pruning | MD5 去重 + 大型工具输出截断（>1000 字符） | 0 | 无 |
| Stage 2: Head Protect | SystemMessage 保护 + 前 3 轮对话保护 | 0 | 无 |
| Stage 3: Tail Protect | 从尾部保留最近约 20K tokens | 0 | 无 |
| Stage 4: Summarize | 14 字段结构化 LLM 摘要（中间部分） | 1（轻量 LLM） | 截断降级 |
| Stage 5: Repair | 修复孤立 tool_call/tool_result 对 | 0 | 无 |

每阶段后检查是否已达预算，提前退出。反抖动机制（连续 2 次低效压缩后跳过）+ 冷却期。

### 14 字段摘要模板

Active Task、Goal、Constraints & Preferences、Completed Actions、Active State、In Progress、Blocked、Key Decisions、Resolved Questions、Pending User Asks、Relevant Files、Remaining Work、Critical Context

---

## 七、预算控制（4 层）

| 层 | 机制 | 默认值 | 位置 |
|----|------|--------|------|
| Agent 总步数 | `recursion_limit` | 8 | `agent.py` |
| 每轮 RAG 调用 | `_KNOWLEDGE_TOOL_CALLS_THIS_TURN` | 1 | `tools.py` |
| 图内总迭代 | `Budget(max=5)` | 5 | `budget_check.py` |
| 改写重试上限 | `MAX_RETRY_ITERATIONS` | 3 | `config.py` |

`budget_check` 节点两道闸门（retry_count > 3 或 Budget 耗尽），任一触及即触发 `handle_exhaustion`。

```python
def budget_check(state):
    if state["retry_count"] > MAX_RETRY_ITERATIONS:  # > 3 → 逃生
        return {"budget_exhausted": True}
    budget = Budget.from_dict(state["iteration_budget"])
    if budget.is_exhausted:  # 耗尽 → 逃生
        return {"budget_exhausted": True}
    return {"budget_exhausted": False,
            "iteration_budget": budget.consume(1).to_dict()}
```

---

## 八、三级分块 + 自动合并

### 8.1 分块规格

| 层级 | 大小 | 重叠 | separator | 用途 |
|------|------|------|-----------|------|
| L1 | 1200 字 | 240 字 | `\n\n, \n, 。, ！, ？, ，` | 完整段落，最终回答用 |
| L2 | 600 字 | 120 字 | 同上 | 中间粒度 |
| L3 | 300 字 | 60 字 | 同上 | 检索精度最高 |

`RecursiveCharacterTextSplitter` 递归切分，保证语义边界。

### 8.2 自动合并

```
L3 检索 → _auto_merge_documents()
  Step 1: L3 按 parent_chunk_id 分组 → 命中 ≥ threshold(2) → 替换为 L2 父块
  Step 2: L2 继续合并 → 命中 ≥ threshold(2) → 替换为 L1 父块
  → 去重 → 按分数排序 → 截断 top_k
```

父块从 PostgreSQL `parent_chunks` 表 + Redis 缓存回填。

### 8.3 渐进式回填（v2 补充）

`hybrid_retrieve` 节点的 `_apply_progressive_backfill()` 支持更灵活的按需回填，与自动合并互补。

---

## 九、子问题分解

复杂问题自动拆为 2-4 个子问题，并行检索后合并：

```
rewrite_expand(strategy="complex")
  → decompose_query("华东区 Q3 毛利率和净利润相比 Q2 如何变化？")
    → ["华东区 Q3 毛利率", "华东区 Q3 净利润", "华东区 Q2 毛利率", "华东区 Q2 净利润"]
      → 4 路并行检索 → merge_sub_query_results → top_k 去重合并
```

---

## 十、端到端数据流

```
用户: "华东区2024年财报中的净利润是多少？"

guardrail_query        → 安全检查通过
fast_path              → needs_retrieval=True (confidence 0.95)
inject_memory          → Layer 0:1条 + Layer 1:2条 + Layer 2:0条 → 3条记忆注入
                          "[USER MEMORY] - [preference] 用户偏好对比去年同期"
decide_retrieval       → LLM分析: query_type=entity_dense, entities=[华东区,2024年,净利润]
                          strategy=entity_boosted (confidence 0.92)
plan_retrieval         → top_k=9, entity_boost=True, hyde=False,
                          step_back=True, metadata_filters={year: 2024}
transform_query        → "华东区 2024年 财报 净利润 利润 数据"
rewrite_expand         → step_back: "财报中关于财务数据的部分有哪些？"
                          query 扩展 + 退步回答拼接
hybrid_retrieve        → retrieve_documents(query, strategy="entity_boosted",
                          skip_rerank=True, top_k=9)
                          → Milvus hybrid(dense+sparse+RRF) → 36 候选
                          → entity_boost 后处理 → 9 文档
                          → auto-merge: l2_120 2子块命中 → L2 父块替换
progressive_rerank     → Stage1: 归一化 → Stage2: 融合(0.6/0.4)
                          → Stage3: Cross-Encoder 精排 → Stage4: 上下文提权
                          → reranked_docs (9个重排后的文档)
grade_documents        → LLM grader: "yes"（文档与净利润查询相关）
compress_check         → 3500 tokens < 80000 → 跳过压缩
generate_answer        → 基于 reranked_docs + memory_context 生成
                          "根据2024年报，华东区全年净利润为 3.2 亿元，
                           同比增长 15%。去年同期为 2.78 亿元。
                           [来源: 2024年报.pdf, Page 15]"
evaluate_confidence    → LLM 评估: score=0.88, level=answer
                          "有具体数据+来源引用"
guardrail_output       → 输出安全检查通过
post_process           → 记忆同步 + 审查检查(未触发) + 延时/费用记录
                          latency_ms=2340, cost_usd=0.012
```

---

## 十一、State 关键字段

```python
class AgenticRAGState(TypedDict):
    # 输入
    question: str
    session_id: str
    user_id: str
    messages: Annotated[List[BaseMessage], add_messages]

    # 快速路径
    needs_retrieval: bool

    # 记忆
    memory_context: Optional[str]
    session_memory: List[Dict]
    memory_sources: List[str]

    # 策略
    retrieval_strategy: Optional[str]      # hybrid / dense_only / sparse_only / entity_boosted
    retrieval_strategy_reason: Optional[str]
    retrieval_strategy_confidence: float
    retrieval_plan: Optional[Dict]

    # 查询
    query: str
    original_question: Optional[str]
    rewrite_strategy: Optional[str]
    _sub_queries: Optional[List[str]]

    # 检索
    retrieved_docs: List[Dict]
    reranked_docs: List[Dict]              # progressive_rerank 产出
    rerank_trace: Dict
    retrieval_metadata: Dict
    _needs_compression: bool
    _retrieved_filenames: Optional[List[str]]

    # 评估
    relevance_grade: Optional[str]         # yes / no
    filtered_docs: List[Dict]
    answer: Optional[str]
    citations: List[Dict]
    confidence_assessment: Optional[Dict]  # {score, level, reason}
    is_degraded_answer: bool

    # 预算
    iteration_budget: Dict                 # {max, used, grace}
    budget_exhausted: bool
    retry_count: int

    # 护栏
    _query_blocked: bool
    _block_reason: Optional[str]
    _redactions: List[str]

    # 追踪
    trace_steps: List[Dict]
    latency_ms: int
    cost_usd: float
    turns_since_review: int
    review_triggered: bool
```

---

## 十二、文件索引

### agentic_rag/ 包

| 文件 | 职责 |
|------|------|
| `graph.py` | 18 节点图构建 + 条件路由 |
| `state.py` | AgenticRAGState 完整定义 + create_initial_state |
| `schemas.py` | Budget（frozen dataclass，不可变）+ ConfidenceAssessment（Pydantic） |
| `config.py` | 阈值收敛：预算、压缩、检索、置信、记忆、护栏、Agent |
| `llm.py` | LLM 工厂：get_llm / get_lightweight_llm / get_grader |
| `token_estimator.py` | Token 估算工具 |
| `retrieval_v2.py` | 子问题分解、实体提取、渐进回填、上下文增强 |
| `prompt_builder.py` | Prompt 构建辅助 |
| `runner.py` | 同步/异步图运行入口 |

### agentic_rag/nodes/ 节点

| 文件 | 节点 |
|------|------|
| `guardrail_query_node.py` | guardrail_query |
| `fast_path.py` | fast_path |
| `direct_answer.py` | direct_answer |
| `inject_memory.py` | inject_memory（3 层记忆） |
| `decide_retrieval.py` | decide_retrieval（LLM 策略选择） |
| `plan_retrieval.py` | plan_retrieval（参数规划） |
| `transform_query.py` | transform_query |
| `rewrite_expand.py` | rewrite_expand（3 策略 + 子问题分解） |
| `hybrid_retrieve.py` | hybrid_retrieve |
| `progressive_rerank.py` | progressive_rerank（4 阶段） |
| `grade_documents.py` | grade_documents |
| `compress_check.py` | compress_check |
| `generate_answer.py` | generate_answer |
| `evaluate_confidence.py` | evaluate_confidence |
| `budget_check.py` | budget_check（双重检查） |
| `handle_exhaustion.py` | handle_exhaustion |
| `guardrail_output.py` | guardrail_output |
| `post_process.py` | post_process |

### 检索基础设施

| 文件 | 职责 |
|------|------|
| `rag_utils.py` | 检索核心：retrieve_documents（策略路由）、rerank、auto-merge、step-back、HyDE |
| `milvus_client.py` | Milvus 封装：hybrid_retrieve / dense_retrieve / sparse_retrieve |
| `embedding.py` | BGE-M3 嵌入 + BM25 稀疏向量（自实现，k1=1.5, b=0.75） |
| `document_loader.py` | 三级分块（L1/L2/L3）+ 嵌套关系 |
| `parent_chunk_store.py` | 父块 PostgreSQL + Redis 存取 |

### 核心机制

| 文件 | 职责 |
|------|------|
| `core/context_compressor.py` | 5 阶段上下文压缩 |
| `core/iteration_budget.py` | 线程安全 IterationBudget（Agent 层） |
| `core/budget_manager.py` | 父子 Agent 预算协调 |
| `core/loop_detector.py` | 死循环检测与恢复 |
| `core/episodic_memory.py` | 情节记忆（查询-命中记录） |
| `core/incremental_save.py` | 增量保存（MD5 去重） |
| `core/pagination.py` | 消息分页（SQL COUNT 注入） |
| `core/rate_limiter.py` | 速率限制 |
| `core/tool_guardrails.py` | 工具调用护栏 |

### 记忆系统

| 文件 | 职责 |
|------|------|
| `memory/memory_manager.py` | 三层记忆协调器 |
| `memory/providers/base.py` | MemoryProvider 抽象 |
| `memory/providers/session.py` | Layer 0: Redis session 记忆 |
| `memory/providers/short_term.py` | Layer 1: PostgreSQL FTS 短期记忆 |
| `memory/providers/long_term.py` | Layer 2: PostgreSQL + BGE-M3 向量长期记忆 |

### 安全护栏

| 文件 | 职责 |
|------|------|
| `guardrails/query_guard.py` | 输入护栏（52 种危险模式） |
| `guardrails/output_guard.py` | 输出护栏（敏感词脱敏） |
| `guardrails/rules.py` | 规则定义 |
