"""自定义检索层指标 — RAGAS 不覆盖的指标（Hit Rate, MRR, NDCG 等）。"""

import math
from typing import List


def hit_rate_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """Hit Rate@k: 前 k 个结果中是否至少包含一个正确文档。

    Args:
        retrieved_ids: 检索返回的文档 ID 列表（按排名排序）
        expected_ids: 期望命中的文档 ID 列表
        k: 截断位置

    Returns:
        1.0 如果命中，0.0 如果未命中
    """
    top_k = set(retrieved_ids[:k])
    expected = set(expected_ids)
    return 1.0 if top_k & expected else 0.0


def mrr_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """MRR@k: 第一个正确结果的排名倒数。

    Args:
        retrieved_ids: 检索返回的文档 ID 列表（按排名排序）
        expected_ids: 期望命中的文档 ID 列表
        k: 截断位置

    Returns:
        第一个正确结果的排名倒数，未命中返回 0.0
    """
    expected = set(expected_ids)
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in expected:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """Recall@k: 检索到的相关文档占全部相关文档的比例。

    Args:
        retrieved_ids: 检索返回的文档 ID 列表（按排名排序）
        expected_ids: 期望命中的文档 ID 列表
        k: 截断位置

    Returns:
        召回率 [0, 1]
    """
    top_k = set(retrieved_ids[:k])
    expected = set(expected_ids)
    if not expected:
        return 1.0
    return len(top_k & expected) / len(expected)


def precision_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """Precision@k: 前 k 个结果中相关文档的比例。

    Args:
        retrieved_ids: 检索返回的文档 ID 列表（按排名排序）
        expected_ids: 期望命中的文档 ID 列表
        k: 截断位置

    Returns:
        精确率 [0, 1]
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    expected = set(expected_ids)
    return sum(1 for doc_id in top_k if doc_id in expected) / len(top_k)


def ndcg_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
    """NDCG@k: 归一化折损累计增益。

    Args:
        retrieved_ids: 检索返回的文档 ID 列表（按排名排序）
        expected_ids: 期望命中的文档 ID 列表
        k: 截断位置

    Returns:
        NDCG 分数 [0, 1]
    """
    expected = set(expected_ids)

    # DCG: 相关文档在位置 i 的增益为 1/log2(i+2)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in expected
    )

    # IDCG: 理想排序下所有相关文档都在最前面
    n_relevant = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def compute_all_retrieval_metrics(
    retrieved_ids: List[str],
    expected_ids: List[str],
    k_values: List[int] = None,
) -> dict:
    """计算所有检索层指标。

    Args:
        retrieved_ids: 检索返回的文档 ID 列表
        expected_ids: 期望命中的文档 ID 列表
        k_values: k 值列表，默认 [3, 5, 10]

    Returns:
        指标字典，如 {"hit_rate@3": 1.0, "mrr@5": 0.8, ...}
    """
    if k_values is None:
        k_values = [3, 5, 10]

    results = {}
    for k in k_values:
        results[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_ids, expected_ids, k)
        results[f"mrr@{k}"] = mrr_at_k(retrieved_ids, expected_ids, k)
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, expected_ids, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_ids, expected_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, expected_ids, k)

    return results
