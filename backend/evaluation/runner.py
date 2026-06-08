"""评测运行器 — 读取 Golden Dataset → 执行 RAG 管线 → 计算指标 → 输出报告。"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

EVAL_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "reports"


def _run_rag_pipeline(question: str, user_id: str = "eval") -> dict:
    """运行 RAG 管线，返回完整状态。

    使用 runner.run_agentic_rag_sync 同步执行。
    """
    from agentic_rag.runner import run_agentic_rag_sync

    return run_agentic_rag_sync(question=question, user_id=user_id)


def _extract_retrieved_ids(state: dict) -> List[str]:
    """从 RAG 状态中提取检索到的文档 ID 列表。"""
    docs = (
        state.get("reranked_docs")
        or state.get("filtered_docs")
        or state.get("retrieved_docs", [])
    )
    return [doc.get("chunk_id", doc.get("filename", "")) for doc in docs]


def _extract_answer(state: dict) -> str:
    """从 RAG 状态中提取生成的回答。"""
    return state.get("answer", "")


def _extract_contexts(state: dict) -> List[str]:
    """从 RAG 状态中提取检索上下文文本列表。"""
    docs = (
        state.get("reranked_docs")
        or state.get("filtered_docs")
        or state.get("retrieved_docs", [])
    )
    return [doc.get("text", "") for doc in docs]


def run_evaluation(
    dataset_path: Optional[str] = None,
    max_questions: int = 50,
    run_ragas: bool = True,
) -> dict:
    """运行完整评测。

    Args:
        dataset_path: Golden Dataset 文件路径，默认 data/eval/golden_dataset.json
        max_questions: 最大评测问题数
        run_ragas: 是否运行 RAGAS 评估（需要 LLM 调用，成本较高）

    Returns:
        评测报告字典
    """
    from evaluation.dataset import load_dataset

    if dataset_path is None:
        dataset_path = str(
            Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "golden_dataset.json"
        )

    dataset = load_dataset(dataset_path)
    if not dataset:
        return {"error": "Empty or invalid dataset"}

    dataset = dataset[:max_questions]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    # 逐题运行 RAG 管线
    results = []
    ragas_questions = []
    ragas_answers = []
    ragas_contexts = []
    ragas_ground_truths = []

    for item in dataset:
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        expected_sources = item.get("expected_sources", [])

        logger.info("Evaluating: %s", question[:60])

        try:
            state = _run_rag_pipeline(question)
            retrieved_ids = _extract_retrieved_ids(state)
            answer = _extract_answer(state)
            contexts = _extract_contexts(state)

            # 自定义检索指标
            from evaluation.metrics import compute_all_retrieval_metrics

            retrieval_metrics = compute_all_retrieval_metrics(retrieved_ids, expected_sources)

            result = {
                "id": item.get("id", ""),
                "question": question,
                "answer": answer,
                "expected_answer": expected_answer,
                "retrieved_ids": retrieved_ids,
                "expected_sources": expected_sources,
                "retrieval_metrics": retrieval_metrics,
                "confidence": state.get("confidence_assessment", {}),
                "budget_used": (state.get("iteration_budget") or {}).get("used", 0),
                "is_degraded": state.get("is_degraded_answer", False),
                "latency_ms": state.get("latency_ms", 0),
            }
            results.append(result)

            # 收集 RAGAS 评估数据
            if run_ragas and answer:
                ragas_questions.append(question)
                ragas_answers.append(answer)
                ragas_contexts.append(contexts)
                ragas_ground_truths.append(expected_answer)

        except Exception as e:
            logger.error("Evaluation failed for '%s': %s", question[:60], e)
            results.append({
                "id": item.get("id", ""),
                "question": question,
                "error": str(e),
            })

    # 汇总检索指标
    aggregated_retrieval = _aggregate_retrieval_metrics(results)

    # RAGAS 生成层指标
    ragas_metrics = {}
    if run_ragas and ragas_questions:
        try:
            from evaluation.ragas_eval import run_ragas_evaluation

            ragas_metrics = run_ragas_evaluation(
                questions=ragas_questions,
                answers=ragas_answers,
                contexts_list=ragas_contexts,
                ground_truths=ragas_ground_truths,
            )
        except Exception as e:
            logger.error("RAGAS evaluation failed: %s", e)
            ragas_metrics = {"error": str(e)}

    duration = time.time() - start_time

    report = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(dataset),
        "evaluated": len(results),
        "errors": sum(1 for r in results if "error" in r),
        "duration_seconds": round(duration, 2),
        "retrieval_metrics": aggregated_retrieval,
        "ragas_metrics": ragas_metrics,
        "results": results,
    }

    # 保存报告
    _save_report(report, run_id)

    return report


def _aggregate_retrieval_metrics(results: List[dict]) -> dict:
    """汇总检索层指标（取平均值）。"""
    from collections import defaultdict

    sums = defaultdict(float)
    counts = defaultdict(int)

    for r in results:
        metrics = r.get("retrieval_metrics", {})
        for k, v in metrics.items():
            sums[k] += v
            counts[k] += 1

    return {k: round(sums[k] / counts[k], 4) if counts[k] > 0 else 0.0 for k in sums}


def _save_report(report: dict, run_id: str):
    """保存评测报告到文件。"""
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_REPORTS_DIR / f"report_{run_id}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Evaluation report saved: %s", path)


def format_report_markdown(report: dict) -> str:
    """将评测报告格式化为 Markdown。"""
    lines = [
        f"# RAG 评测报告",
        f"",
        f"- **运行 ID**: {report['run_id']}",
        f"- **时间**: {report['timestamp']}",
        f"- **数据集大小**: {report['dataset_size']}",
        f"- **评测问题数**: {report['evaluated']}",
        f"- **错误数**: {report['errors']}",
        f"- **耗时**: {report['duration_seconds']}s",
        f"",
        f"## 检索层指标",
        f"",
        f"| 指标 | 分数 | 合格标准 | 状态 |",
        f"|------|------|---------|------|",
    ]

    thresholds = {
        "hit_rate@5": (0.95, ">="),
        "mrr@5": (0.8, ">="),
        "recall@5": (0.8, ">="),
        "precision@3": (0.7, ">="),
        "ndcg@5": (0.75, ">="),
    }

    for k, v in report.get("retrieval_metrics", {}).items():
        threshold, op = thresholds.get(k, (None, None))
        if threshold is not None:
            passed = v >= threshold if op == ">=" else v <= threshold
            status = "PASS" if passed else "FAIL"
        else:
            status = "-"
        lines.append(f"| {k} | {v:.4f} | {threshold or '-'} | {status} |")

    lines.extend([
        f"",
        f"## 生成层指标 (RAGAS)",
        f"",
        f"| 指标 | 分数 | 合格标准 | 状态 |",
        f"|------|------|---------|------|",
    ])

    ragas_thresholds = {
        "faithfulness": (0.9, ">="),
        "answer_relevancy": (0.8, ">="),
        "context_precision": (0.7, ">="),
        "context_recall": (0.7, ">="),
    }

    for k, v in report.get("ragas_metrics", {}).items():
        if isinstance(v, (int, float)):
            threshold, op = ragas_thresholds.get(k, (None, None))
            if threshold is not None:
                passed = v >= threshold if op == ">=" else v <= threshold
                status = "PASS" if passed else "FAIL"
            else:
                status = "-"
            lines.append(f"| {k} | {v:.4f} | {threshold or '-'} | {status} |")

    return "\n".join(lines)
