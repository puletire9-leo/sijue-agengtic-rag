"""Golden Dataset 管理 — 加载、验证、CRUD。"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "golden_dataset.json"


def load_dataset(path: str = None) -> List[dict]:
    """加载 Golden Dataset。

    Args:
        path: 数据集文件路径，默认 data/eval/golden_dataset.json

    Returns:
        数据集条目列表
    """
    if path is None:
        path = str(DEFAULT_DATASET_PATH)

    p = Path(path)
    if not p.exists():
        logger.warning("Dataset not found: %s", path)
        return []

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            logger.error("Dataset must be a JSON array")
            return []
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load dataset: %s", e)
        return []


def save_dataset(dataset: List[dict], path: str = None):
    """保存 Golden Dataset。

    Args:
        dataset: 数据集条目列表
        path: 保存路径
    """
    if path is None:
        path = str(DEFAULT_DATASET_PATH)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Dataset saved: %s (%d items)", path, len(dataset))


def validate_item(item: dict) -> List[str]:
    """验证单条数据集条目。

    Returns:
        错误列表，空表示验证通过
    """
    errors = []

    if not item.get("question"):
        errors.append("missing 'question'")

    if not item.get("expected_answer"):
        errors.append("missing 'expected_answer'")

    if "expected_sources" not in item:
        errors.append("missing 'expected_sources'")

    difficulty = item.get("difficulty", "")
    if difficulty and difficulty not in ("easy", "medium", "hard"):
        errors.append(f"invalid difficulty: {difficulty}")

    category = item.get("category", "")
    if category and category not in ("factual", "analytical", "multi-hop", "boundary"):
        errors.append(f"invalid category: {category}")

    return errors


def add_item(
    dataset: List[dict],
    question: str,
    expected_answer: str,
    expected_sources: List[str],
    difficulty: str = "medium",
    category: str = "factual",
) -> dict:
    """添加一条数据集条目。

    Args:
        dataset: 现有数据集
        question: 用户问题
        expected_answer: 标准答案
        expected_sources: 期望命中的文档列表
        difficulty: 难度 (easy/medium/hard)
        category: 类别 (factual/analytical/multi-hop/boundary)

    Returns:
        新增的条目
    """
    item = {
        "id": f"q{uuid.uuid4().hex[:8]}",
        "question": question,
        "expected_answer": expected_answer,
        "expected_sources": expected_sources,
        "difficulty": difficulty,
        "category": category,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    dataset.append(item)
    return item


def update_item(dataset: List[dict], item_id: str, updates: dict) -> bool:
    """更新一条数据集条目。

    Args:
        dataset: 现有数据集
        item_id: 条目 ID
        updates: 要更新的字段

    Returns:
        是否找到并更新
    """
    for item in dataset:
        if item.get("id") == item_id:
            item.update(updates)
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            return True
    return False


def delete_item(dataset: List[dict], item_id: str) -> bool:
    """删除一条数据集条目。

    Returns:
        是否找到并删除
    """
    for i, item in enumerate(dataset):
        if item.get("id") == item_id:
            dataset.pop(i)
            return True
    return False


def get_statistics(dataset: List[dict]) -> dict:
    """获取数据集统计信息。"""
    if not dataset:
        return {"total": 0}

    difficulties = {}
    categories = {}
    for item in dataset:
        d = item.get("difficulty", "unknown")
        c = item.get("category", "unknown")
        difficulties[d] = difficulties.get(d, 0) + 1
        categories[c] = categories.get(c, 0) + 1

    return {
        "total": len(dataset),
        "by_difficulty": difficulties,
        "by_category": categories,
    }
