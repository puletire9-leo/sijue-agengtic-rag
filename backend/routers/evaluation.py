"""评测 API — Golden Dataset 管理 + 评测运行 + 报告查询。"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["evaluation"])


# ── 请求模型 ──


class DatasetItemCreate(BaseModel):
    question: str
    expected_answer: str
    expected_sources: list[str]
    difficulty: str = "medium"
    category: str = "factual"


class DatasetItemUpdate(BaseModel):
    question: Optional[str] = None
    expected_answer: Optional[str] = None
    expected_sources: Optional[list[str]] = None
    difficulty: Optional[str] = None
    category: Optional[str] = None


class EvalRunRequest(BaseModel):
    max_questions: int = 50
    run_ragas: bool = True
    dataset_path: Optional[str] = None


# ── Dataset 端点 ──


@router.get("/dataset")
async def get_dataset():
    """获取 Golden Dataset 列表。"""
    from evaluation.dataset import load_dataset, get_statistics

    dataset = load_dataset()
    stats = get_statistics(dataset)
    return {"items": dataset, "statistics": stats}


@router.post("/dataset")
async def add_dataset_item(item: DatasetItemCreate):
    """添加一条数据集条目。"""
    from evaluation.dataset import load_dataset, save_dataset, add_item, validate_item

    dataset = load_dataset()
    new_item = add_item(
        dataset,
        question=item.question,
        expected_answer=item.expected_answer,
        expected_sources=item.expected_sources,
        difficulty=item.difficulty,
        category=item.category,
    )
    errors = validate_item(new_item)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    save_dataset(dataset)
    return {"item": new_item, "total": len(dataset)}


@router.put("/dataset/{item_id}")
async def update_dataset_item(item_id: str, updates: DatasetItemUpdate):
    """更新一条数据集条目。"""
    from evaluation.dataset import load_dataset, save_dataset, update_item

    dataset = load_dataset()
    update_dict = updates.model_dump(exclude_none=True)
    if not update_dict:
        raise HTTPException(status_code=400, detail="No fields to update")

    if not update_item(dataset, item_id, update_dict):
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    save_dataset(dataset)
    return {"item_id": item_id, "updated_fields": list(update_dict.keys())}


@router.delete("/dataset/{item_id}")
async def delete_dataset_item(item_id: str):
    """删除一条数据集条目。"""
    from evaluation.dataset import load_dataset, save_dataset, delete_item

    dataset = load_dataset()
    if not delete_item(dataset, item_id):
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    save_dataset(dataset)
    return {"item_id": item_id, "total": len(dataset)}


# ── 评测运行端点 ──


@router.post("/run")
async def run_eval(request: EvalRunRequest):
    """运行完整评测（admin 权限）。"""
    from evaluation.runner import run_evaluation, format_report_markdown

    try:
        report = run_evaluation(
            dataset_path=request.dataset_path,
            max_questions=request.max_questions,
            run_ragas=request.run_ragas,
        )
    except Exception as e:
        log.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))

    if "error" in report:
        raise HTTPException(status_code=400, detail=report["error"])

    markdown = format_report_markdown(report)
    return {"report": report, "markdown": markdown}


@router.get("/report/{run_id}")
async def get_report(run_id: str):
    """获取评测报告。"""
    import json
    from pathlib import Path

    reports_dir = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "reports"
    report_path = reports_dir / f"report_{run_id}.json"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report {run_id} not found")

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        from evaluation.runner import format_report_markdown

        markdown = format_report_markdown(report)
        return {"report": report, "markdown": markdown}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_eval_history():
    """获取评测历史列表。"""
    import json
    from pathlib import Path

    reports_dir = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "reports"
    if not reports_dir.exists():
        return {"reports": []}

    reports = []
    for f in sorted(reports_dir.glob("report_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            reports.append({
                "run_id": data.get("run_id", f.stem),
                "timestamp": data.get("timestamp", ""),
                "dataset_size": data.get("dataset_size", 0),
                "evaluated": data.get("evaluated", 0),
                "errors": data.get("errors", 0),
                "duration_seconds": data.get("duration_seconds", 0),
                "retrieval_metrics": data.get("retrieval_metrics", {}),
                "ragas_metrics": {
                    k: v for k, v in data.get("ragas_metrics", {}).items()
                    if isinstance(v, (int, float))
                },
            })
        except Exception:
            continue

    return {"reports": reports}
