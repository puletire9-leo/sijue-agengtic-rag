"""RAGAS 评估集成 — Faithfulness, Answer Relevancy, Context Precision/Recall。

使用 RAGAS 库计算生成层指标。RAGAS 需要以下字段的 Dataset:
  - question: 用户问题
  - answer: 生成的回答
  - contexts: 检索到的上下文列表
  - ground_truth: 标准答案（用于 Context Recall）
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def build_ragas_dataset(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    ground_truths: Optional[List[str]] = None,
):
    """构建 RAGAS 评估所需的 Dataset。

    Args:
        questions: 用户问题列表
        answers: RAG 系统生成的回答列表
        contexts_list: 每个问题对应的检索上下文列表
        ground_truths: 标准答案列表（可选，用于 Context Recall）

    Returns:
        HuggingFace Dataset 对象
    """
    from datasets import Dataset

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    return Dataset.from_dict(data)


def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    ground_truths: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """运行 RAGAS 评估。

    Args:
        questions: 用户问题列表
        answers: RAG 系统生成的回答列表
        contexts_list: 每个问题对应的检索上下文列表
        ground_truths: 标准答案列表（可选）
        metrics: 要计算的指标列表，默认全部

    Returns:
        指标字典，如 {"faithfulness": 0.92, "answer_relevancy": 0.85, ...}
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError:
        logger.warning("ragas not installed, skipping RAGAS evaluation")
        return {}

    dataset = build_ragas_dataset(questions, answers, contexts_list, ground_truths)

    # 选择指标
    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    if metrics is None:
        selected = list(metric_map.values())
    else:
        selected = [metric_map[m] for m in metrics if m in metric_map]

    if not selected:
        logger.warning("No valid metrics selected")
        return {}

    try:
        result = evaluate(dataset=dataset, metrics=selected)
        return result
    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        return {}


def run_single_ragas_eval(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, float]:
    """对单个问答运行 RAGAS 评估。

    Args:
        question: 用户问题
        answer: RAG 系统生成的回答
        contexts: 检索到的上下文列表
        ground_truth: 标准答案（可选）

    Returns:
        指标字典
    """
    return run_ragas_evaluation(
        questions=[question],
        answers=[answer],
        contexts_list=[contexts],
        ground_truths=[ground_truth] if ground_truth else None,
    )
