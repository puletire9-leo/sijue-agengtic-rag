## SuperMew RAG 检索与回答质量优化计划

> **基于**: "几何不确定性框架 vs Ensemble/MC Dropout" 查询案例分析  
> **日期**: 2026-06-01  
> **涉及节点**: rewrite_expand、grade_documents、handle_exhaustion、plan_retrieval、graph

---

## 问题诊断总结

通过对一次完整查询的端到端分析（31 秒思考 + 降级回答），发现 RAG 管线在检索重试、查询改写、文档评估和降级输出四个环节存在可优化空间。以下是 5 项修改的优先级排序和详细实施方案。

| 优先级 | 修改项 | 影响 | 涉及文件 | 预计工时 |
|:------:|--------|------|----------|:--------:|
| P1 | 查询改写策略太保守 | 检索命中率 | `rag_utils.py` | 0.5 天 |
| P2 | 策略穷尽后重复检索浪费预算 | 响应延迟 | `rewrite_expand.py`, `graph.py` | 0.5 天 |
| P3 | 文档评估粒度太粗（二元判断） | 有效文档丢弃 | `grade_documents.py`, `graph.py` | 1 天 |
| P4 | 降级回答未标注"非知识库来源" | 用户信任 | `handle_exhaustion.py` | 0.5 天 |
| P5 | `top_k=5` 对复杂比较类查询偏低 | 上下文不足 | `plan_retrieval.py` | 0.5 天 |

---

## P1：优化 Step-back 查询改写策略

### 问题分析

当前 `step_back_expand()` 的实现（`rag_utils.py` 第 288-303 行）将退步问题和答案**追加**到原查询后面：

```python
# rag_utils.py 第 292-296 行（当前实现）
expanded_query = (
    f"{query}\n\n"
    f"退步问题：{step_back_question}\n"
    f"退步问题答案：{step_back_answer}"
)
```

这导致改写后的查询是原查询的**超集**，核心关键词完全没变。以案例为例：

- 原查询: `几何不确定性框架 Ensemble Monte Carlo Dropout 对比效果 性能提升`
- 改写后: `几何不确定性框架 Ensemble Monte Carlo Dropout 对比效果 性能提升\n\n退步问题：不确定性估计\n退步问题答案：...`

检索引擎仍然以原始关键词为主导进行匹配，退步部分被稀释，几乎不改变检索结果。

### 修改方案

**文件**: `backend/rag_utils.py`，`step_back_expand()` 函数（第 288-303 行）

将查询拼接逻辑从"追加"改为"**替换式改写**"：用退步问题作为主查询，原始查询作为补充上下文。

```python
# 修改后
def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    if step_back_question:
        # 以退步问题为主查询，原始查询为上下文补充
        expanded_query = (
            f"{step_back_question}\n"
            f"（原始问题背景：{query}）"
        )
        if step_back_answer:
            expanded_query += f"\n参考背景：{step_back_answer}"
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }
```

**同时优化 prompt**，让 `_generate_step_back_question` 产出真正发散性的查询。

**文件**: `backend/rag_utils.py`，`_generate_step_back_question()` 函数（第 242-254 行）

```python
# 修改后 prompt
prompt = (
    "请将用户的具体问题改写为一个更高层次、不同角度的搜索查询，"
    "用于在知识库中检索相关的通用原理或背景知识。\n"
    "要求：\n"
    "- 不要包含原查询中的专有名词和具体术语\n"
    "- 使用不同的措辞和表达方式\n"
    "- 输出一句简短的搜索查询，不要解释\n\n"
    f"用户问题：{query}"
)
```

### 预期效果

以案例为例，改写后的查询应变为类似：

- `不确定性量化方法 模型置信度估计 技术对比`

而非原来的：

- `几何不确定性框架 Ensemble Monte Carlo Dropout 对比效果 性能提升\n退步问题：不确定性估计`

### 验证方法

1. 用 5 个不同类型的问题测试 step_back 改写输出，确认改写后的查询与原查询**关键词重叠度 < 50%**
2. 对相同知识库运行改写前后的检索，对比召回文档的**差异率**（期望 > 30% 新文档）
3. 监控 step_back 改写后 `grade_documents` 通过率是否提升

---

## P2：策略穷尽后跳过重复检索

### 问题分析

当 `rewrite_expand` 判定所有策略已穷尽（第 128-138 行），设置 `budget_exhausted: True` 并返回。但当前 graph 的边定义（`graph.py` 第 136 行）是：

```
rewrite_expand → hybrid_retrieve（硬边，无条件执行）
```

这意味着即使 `rewrite_expand` 返回 `budget_exhausted: True`，`hybrid_retrieve` 仍会用**相同的查询**再检索一次（因为 query 没有改变）。之后 `progressive_rerank` 和 `grade_documents` 也会依次执行，最终 `grade_documents` 大概率再次判"不相关"，然后 `budget_check` 才路由到 `handle_exhaustion`。

**整个过程中 `hybrid_retrieve → progressive_rerank → grade_documents` 这 3 个节点是纯浪费。**

### 修改方案

**方案 A（推荐）：在 `rewrite_expand` 中提前设置 `relevance_grade: "no"`**

**文件**: `backend/agentic_rag/nodes/rewrite_expand.py`，策略穷尽分支（第 128-138 行）

```python
# 修改后
if not untried:
    emit_rag_step("⚠️", "所有改写策略已穷尽", f"已尝试: {tried}")
    return {
        "query": query,
        "rewrite_strategy": current_strategy or allowed_strategies[0],
        "retry_count": retry_count + 1,
        "_sub_queries": None,
        "tried_strategies": tried,
        "budget_exhausted": True,
        # 关键：设置 relevance_grade 使 graph 跳过检索直接进入 budget_check
        # 但 graph.py 的硬边 rewrite_expand → hybrid_retrieve 仍会执行...
    }
```

这个方案单独不够——因为 `rewrite_expand → hybrid_retrieve` 是硬边。**需要配合方案 B。**

**方案 B（核心）：将 `rewrite_expand → hybrid_retrieve` 改为条件边**

**文件**: `backend/agentic_rag/graph.py`

将第 136 行的硬边：

```python
graph.add_edge("rewrite_expand", "hybrid_retrieve")
```

改为条件边：

```python
def route_rewrite(state: AgenticRAGState) -> str:
    """改写节点路由：策略穷尽时跳过检索，直接进入预算检查。"""
    if state.get("budget_exhausted"):
        return "budget_check"
    return "hybrid_retrieve"

# 替换硬边为条件边
graph.add_conditional_edges(
    "rewrite_expand",
    route_rewrite,
    {"hybrid_retrieve": "hybrid_retrieve", "budget_check": "budget_check"},
)
```

同时需要在 `route_budget` 中确保 `budget_exhausted=True` 时路由到 `handle_exhaustion`（当前逻辑已支持，无需修改）。

### 预期效果

策略穷尽时直接跳过 `hybrid_retrieve → progressive_rerank → grade_documents` 三个节点，进入 `budget_check → handle_exhaustion`。**节省约 5-8 秒延迟和 1 次 LLM + Cross-Encoder 调用。**

### 验证方法

1. 触发策略穷尽场景（连续 2 次 grade_documents 判"不相关"），检查日志确认第 3 轮不再出现 `🔍 正在检索知识库` 日志
2. 对比修改前后策略穷尽场景的**总响应时间**
3. 确认正常路径（策略未穷尽）不受影响

---

## P3：文档评估支持"部分相关"三级路由

### 问题分析

当前 `grade_documents`（`grade_documents.py` 第 57-107 行）是二元判断：LLM 评估整批文档（最多 5 篇，每篇前 1200 字符），返回 `{"relevant": true/false}`。全部相关 → 走生成路径；全部不相关 → 走重试路径。

这导致一个常见问题：5 篇文档中如果有 2-3 篇确实相关，但 LLM 因为其他 2-3 篇不相关而判 `false`，整批被丢弃。案例中 3 轮检索都判"不相关"，很可能就是这个原因。

### 修改方案

**文件**: `backend/agentic_rag/nodes/grade_documents.py`

将 prompt 从二元判断改为三级判断，返回 `full` / `partial` / `none`。

```python
# 修改后的 prompt
TIERED_GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved documents to a user question.\n"
    "用户问题是中文，检索文档也可能是中文。请基于语义相关性判断。\n\n"
    "Here are the retrieved documents:\n\n{context}\n\n"
    "Here is the user question: {question}\n\n"
    "请评估整批文档对回答问题的帮助程度，返回以下三个级别之一：\n"
    '- "full": 多数文档（3篇以上）直接相关，可以充分回答问题\n'
    '- "partial": 少数文档（1-2篇）部分相关，可以提供有限但有价值的回答\n'
    '- "none": 没有文档与问题相关，需要重新检索\n'
    'Return ONLY a JSON object: {{"level": "full/partial/none", "relevant_count": 数字, "reason": "简短理由(中文)"}}'
)
```

修改 `grade_documents()` 函数的返回逻辑：

```python
def grade_documents(state: AgenticRAGState) -> dict:
    # ... 前置检查不变 ...

    try:
        prompt = TIERED_GRADE_PROMPT.format(question=question, context=context)
        LLM_CALLS.labels(model="tier2", purpose="grade").inc()
        response = grader.invoke([{"role": "user", "content": prompt}])
        grade_result = _parse_tiered_response(response.content or "")
    except Exception:
        emit_rag_step("⚠️", "文档评估异常，保守视为相关")
        return {"relevance_grade": "yes", "filtered_docs": docs}

    level = grade_result["level"]
    count = grade_result["relevant_count"]

    if level == "full":
        emit_rag_step("✅", f"文档相关性评估通过 ({count}篇相关)")
        return {"relevance_grade": "yes", "filtered_docs": docs}
    elif level == "partial":
        emit_rag_step("🔶", f"部分文档相关 ({count}篇)，生成有限回答")
        return {"relevance_grade": "partial", "filtered_docs": docs}
    else:
        emit_rag_step("⚠️", "文档相关性不足，将重写查询重试")
        return {"relevance_grade": "no", "filtered_docs": docs}
```

新增响应解析函数：

```python
def _parse_tiered_response(content: str) -> dict:
    """Parse tiered grade response. Default: none (retry)."""
    text = content.strip()
    try:
        clean = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        level = parsed.get("level", "none")
        if level not in ("full", "partial", "none"):
            level = "none"
        return {"level": level, "relevant_count": parsed.get("relevant_count", 0)}
    except (json.JSONDecodeError, TypeError):
        # Fallback: 关键词匹配
        lower = text.lower()
        if "full" in lower or "充分" in lower:
            return {"level": "full", "relevant_count": 5}
        if "partial" in lower or "部分" in lower:
            return {"level": "partial", "relevant_count": 2}
        return {"level": "none", "relevant_count": 0}
```

**文件**: `backend/agentic_rag/graph.py`

修改 `route_grade` 函数，增加 `partial` 路由：

```python
def route_grade(state: AgenticRAGState) -> str:
    grade = state.get("relevance_grade")
    if grade == "yes":
        return "compress_check"
    elif grade == "partial":
        return "compress_check"  # partial 也走生成路径，但标记降级
    return "budget_check"
```

在 `compress_check` 或 `generate_answer` 中读取 `relevance_grade == "partial"` 并在回答中注明"基于部分相关文档"。

### 预期效果

以案例为例，如果 5 篇文档中有 2 篇涉及不确定性量化方法（但不是专门对比几何框架），系统会走 `partial` 路径生成"基于部分相关文档"的回答，而不是全部丢弃后重试 3 轮最终降级。

### 验证方法

1. 准备 10 个测试查询，其中 5 个预期"部分相关"，验证三级路由分布
2. 对比修改前后 `grade_documents` 的重试率和最终回答质量
3. 确认 `partial` 路径的回答中正确标注了"基于部分相关文档"

---

## P4：降级回答标注来源类型

### 问题分析

当前 `handle_exhaustion`（第 20-50 行）生成的降级回答有两种情况：

1. 有已有 answer 时：`EXHAUSTION_MESSAGE + answer`
2. 有 docs 但无 answer 时：`EXHAUSTION_MESSAGE + 文档列表`

但两种情况都没有告知用户"以下内容不是基于知识库的"。在实际案例中，LLM 用自身知识生成了高质量回答，但以"我专门检索了知识库"开头，给用户造成误导。

### 修改方案

**文件**: `backend/agentic_rag/nodes/handle_exhaustion.py`

修改 `EXHAUSTION_MESSAGE` 和降级逻辑：

```python
EXHAUSTION_MESSAGE = (
    "⚠️ **提示**: 经过多次检索，知识库中未找到与您问题直接相关的文档。"
    "以下回答基于模型的通用知识生成，仅供参考，建议查阅原始资料确认。\n\n"
    "如果您希望获得基于知识库的回答，可以尝试：\n"
    "1) 换一种方式描述问题\n"
    "2) 使用更简洁的关键词\n"
    "3) 确认相关文档已上传到知识库\n"
)
```

同时，当走到 `handle_exhaustion` 且需要 LLM 生成回答时（当前 `handle_exhaustion` 不主动调用 LLM，只复用已有 answer），考虑在降级路径中主动调用一次 LLM，并在 system prompt 中明确指示"知识库检索失败，请基于自身知识回答并明确标注"：

```python
def handle_exhaustion(state: AgenticRAGState) -> dict:
    emit_rag_step("⚠️", "预算耗尽，输出降级回答")

    answer = state.get("answer")
    docs = state.get("retrieved_docs") or state.get("filtered_docs", [])

    if not answer and docs:
        # 有检索到的文档但 grading 不通过 → 用这些文档生成有限回答
        from agentic_rag.nodes.generate_answer import generate_answer as _gen
        # 强制生成回答（跳过 grading）
        gen_result = _gen(state)
        answer = gen_result.get("answer", "")

    if answer:
        degraded = f"{EXHAUSTION_MESSAGE}\n\n---\n\n{answer}"
    elif docs:
        doc_summary = "\n".join([
            f"- {d.get('filename', 'Unknown')}" for d in docs[:5]
        ])
        degraded = (
            f"{EXHAUSTION_MESSAGE}\n\n"
            f"已检索到 {len(docs)} 条可能相关的文档：\n{doc_summary}\n\n"
            "请尝试用更精确的关键词重新提问。"
        )
    else:
        degraded = EXHAUSTION_MESSAGE

    return {
        "answer": degraded,
        "is_degraded_answer": True,
        "budget_exhausted": True,
        "citations": state.get("citations", []),
        "messages": [AIMessage(content=degraded)],
    }
```

### 预期效果

用户在降级回答开头看到醒目的 ⚠️ 提示，明确知道这是模型通用知识而非知识库内容。

### 验证方法

1. 触发降级场景，检查回答开头是否包含"未找到直接相关的文档"提示
2. 确认正常路径（非降级）不受影响

---

## P5：动态 top_k 适配查询复杂度

### 问题分析

当前 `plan_retrieval.py` 中 `hybrid` 策略的默认 `top_k=5`（第 47 行）。对于简单事实类查询（"X 是什么"）这足够，但对于比较类/综述类查询（"A 和 B 和 C 的区别"），5 个 chunk 的上下文窗口太窄。

当前只有"多实体查询"（5+ 个实体）才会动态提升 top_k（第 117-121 行），阈值太高。

### 修改方案

**文件**: `backend/agentic_rag/nodes/plan_retrieval.py`

在 `plan_retrieval()` 函数中增加**查询复杂度检测**，根据查询特征动态调整 top_k：

```python
def _detect_query_complexity(question: str) -> str:
    """检测查询复杂度类型。

    返回: "simple" | "comparison" | "survey" | "standard"
    """
    # 比较类关键词
    comparison_markers = [
        "对比", "比较", "区别", "差异", "vs", "VS", "和",
        "相比", "优于", "领先", "哪个好", "哪种更好",
    ]
    # 综述类关键词
    survey_markers = [
        "综述", "总结", "概述", "全面", "系统", "所有",
        "各种", "多种", "不同", "分类", "类型",
    ]

    has_comparison = sum(1 for m in comparison_markers if m in question) >= 2
    has_survey = any(m in question for m in survey_markers)

    if has_comparison:
        return "comparison"
    if has_survey:
        return "survey"
    if len(question) > 60:
        return "survey"  # 超长查询通常是复杂问题
    return "standard"


# 在 plan_retrieval() 中，entity_count 检查之后添加：
complexity = _detect_query_complexity(question)
if complexity == "comparison":
    plan["top_k"] = min(plan["top_k"] + 5, 15)
    plan["candidate_multiplier"] = min(plan["candidate_multiplier"] + 1, 5)
    emit_rag_step("📊", f"比较类查询，扩大检索: top_k={plan['top_k']}", "")
elif complexity == "survey":
    plan["top_k"] = min(plan["top_k"] + 3, 12)
    plan["candidate_multiplier"] = min(plan["candidate_multiplier"] + 1, 5)
    emit_rag_step("📚", f"综述类查询，扩大检索: top_k={plan['top_k']}", "")
```

### 预期效果

以案例查询（`这种几何不确定性框架和普通的 Ensemble 方法或 Monte Carlo Dropout 相比，实际效果领先多少？`）为例：

- 包含"相比"、"和" → 检测为 `comparison`
- top_k 从 5 提升到 10
- 召回更多文档，增加找到相关内容的概率

### 验证方法

1. 准备 20 个测试查询（10 简单 + 5 比较 + 5 综述），验证复杂度分类准确率
2. 对比修改前后比较类查询的 `grade_documents` 通过率
3. 监控 top_k 提升对响应延迟的影响（预期增加 1-3 秒）

---

## 修改文件清单

| 文件 | 修改类型 | 修改摘要 |
|------|----------|----------|
| `backend/rag_utils.py` | 优化 | step_back_expand 查询拼接逻辑 + prompt 优化 |
| `backend/agentic_rag/graph.py` | 结构变更 | rewrite_expand 硬边改条件边 + route_grade 增加 partial 路由 |
| `backend/agentic_rag/nodes/rewrite_expand.py` | 配合修改 | 策略穷尽时设置 budget_exhausted（已有，无需改） |
| `backend/agentic_rag/nodes/grade_documents.py` | 重构 | prompt 改三级判断 + 新增 _parse_tiered_response |
| `backend/agentic_rag/nodes/handle_exhaustion.py` | 优化 | EXHAUSTION_MESSAGE 改写 + 降级回答标注来源 |
| `backend/agentic_rag/nodes/plan_retrieval.py` | 增强 | 新增 _detect_query_complexity + 动态 top_k |

---

## 验证检查清单

| # | 验证项 | 方法 | 状态 |
|---|--------|------|:----:|
| 1 | step_back 改写后的查询与原查询关键词重叠度 < 50% | 5 个测试问题 | 待实施 |
| 2 | 策略穷尽时不触发额外检索 | 日志检查 | 待实施 |
| 3 | 策略穷尽时节省 5-8 秒延迟 | 对比计时 | 待实施 |
| 4 | 三级 grading 正确分类 full/partial/none | 10 个测试查询 | 待实施 |
| 5 | partial 路径生成"基于部分相关文档"的回答 | 人工检查 | 待实施 |
| 6 | 降级回答包含 ⚠️ 来源提示 | 触发降级场景 | 待实施 |
| 7 | 比较类查询 top_k 自动提升到 10 | 日志检查 | 待实施 |
| 8 | 正常路径（非降级、简单查询）不受影响 | 回归测试 | 待实施 |
| 9 | 端到端响应时间不退化 | 基线对比 | 待实施 |

---

## 实施顺序建议

建议按以下顺序实施，每项完成后进行验证再进入下一项：

**第一批（核心优化，影响最大）**: P1（step_back prompt）+ P2（条件边）+ P4（降级标注）。这三项互不依赖，可以并行实施，预计 1.5 天。

**第二批（结构优化，需要仔细测试）**: P3（三级 grading）。涉及 prompt 修改和 graph 路由变更，需要充分测试，预计 1 天。

**第三批（增量优化）**: P5（动态 top_k）。改动最小、风险最低，预计 0.5 天。
