# Evaluation — 评测体系

## 模块清单

| 文件 | 功能 |
|------|------|
| `metrics.py` | 自定义检索层指标（Hit Rate, MRR, Recall, Precision, NDCG）|
| `ragas_eval.py` | RAGAS 集成（Faithfulness, Answer Relevancy, Context Precision/Recall）|
| `dataset.py` | Golden Dataset CRUD（加载/验证/增删改/统计）|
| `runner.py` | 评测运行器（读取数据集 → 运行 RAG → 计算指标 → 输出报告）|

## 使用

```python
# 运行评测
from evaluation.runner import run_evaluation
report = run_evaluation(max_questions=50, run_ragas=True)

# 计算自定义指标
from evaluation.metrics import compute_all_retrieval_metrics
metrics = compute_all_retrieval_metrics(retrieved_ids, expected_ids, k_values=[3, 5])

# 管理数据集
from evaluation.dataset import load_dataset, add_item, save_dataset
ds = load_dataset()
add_item(ds, question="...", expected_answer="...", expected_sources=["doc.pdf"])
save_dataset(ds)
```

## Golden Dataset

`data/eval/golden_dataset.json` — 20 条种子 Q&A 数据，覆盖：
- 简单事实查询（3 条）
- 多文档综合（5 条）
- 多跳推理（7 条）
- 边界查询（5 条）

## 评测报告

保存在 `data/eval/reports/report_{run_id}.json`，含检索指标 + RAGAS 指标 + 逐题详情。
