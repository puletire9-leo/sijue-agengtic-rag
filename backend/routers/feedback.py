"""反馈 API — 用户反馈收集 + Bad Case 分析。"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

log = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


# ── 请求模型 ──


class FeedbackCreate(BaseModel):
    session_id: str
    message_id: str
    rating: Optional[int] = Field(None, ge=1, le=5, description="评分 1-5")
    thumbs_up: Optional[bool] = None
    comment: Optional[str] = None


# ── 端点 ──


@router.post("")
async def submit_feedback(body: FeedbackCreate, request):
    """提交用户反馈。"""
    from auth import get_current_user
    from database import get_db
    from models import Feedback

    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    feedback_id = str(uuid.uuid4())
    db = next(get_db())
    try:
        fb = Feedback(
            id=feedback_id,
            user_id=user.id,
            session_id=body.session_id,
            message_id=body.message_id,
            rating=body.rating,
            thumbs_up=body.thumbs_up,
            comment=body.comment,
            created_at=datetime.now(timezone.utc),
        )
        db.add(fb)
        db.commit()
        log.info("Feedback submitted: user=%s rating=%s thumbs_up=%s", user.id, body.rating, body.thumbs_up)
        return {"id": feedback_id, "status": "ok"}
    except Exception as e:
        db.rollback()
        log.error("Failed to submit feedback: %s", e)
        raise HTTPException(status_code=500, detail="Failed to submit feedback")
    finally:
        db.close()


@router.get("/stats")
async def get_feedback_stats(request):
    """获取反馈统计（admin）。"""
    from auth import get_current_user, require_admin
    from database import get_db
    from models import Feedback
    from sqlalchemy import func

    user = await get_current_user(request)
    require_admin(user)

    db = next(get_db())
    try:
        total = db.query(func.count(Feedback.id)).scalar() or 0
        avg_rating = db.query(func.avg(Feedback.rating)).scalar()
        thumbs_up_count = db.query(func.count(Feedback.id)).filter(Feedback.thumbs_up == True).scalar() or 0
        thumbs_down_count = db.query(func.count(Feedback.id)).filter(Feedback.thumbs_up == False).scalar() or 0
        with_comment = db.query(func.count(Feedback.id)).filter(Feedback.comment.isnot(None)).scalar() or 0

        return {
            "total": total,
            "avg_rating": round(float(avg_rating), 2) if avg_rating else None,
            "thumbs_up": thumbs_up_count,
            "thumbs_down": thumbs_down_count,
            "satisfaction_rate": round(thumbs_up_count / total, 4) if total > 0 else 0,
            "with_comment": with_comment,
        }
    finally:
        db.close()


@router.get("/bad-cases")
async def get_bad_cases(request, limit: int = 50):
    """获取差评案例列表（admin）。"""
    from auth import get_current_user, require_admin
    from database import get_db
    from models import Feedback

    user = await get_current_user(request)
    require_admin(user)

    db = next(get_db())
    try:
        bad_cases = (
            db.query(Feedback)
            .filter(
                (Feedback.thumbs_up == False) | (Feedback.rating <= 2)
            )
            .order_by(Feedback.created_at.desc())
            .limit(limit)
            .all()
        )

        return {
            "bad_cases": [
                {
                    "id": fb.id,
                    "user_id": fb.user_id,
                    "session_id": fb.session_id,
                    "message_id": fb.message_id,
                    "rating": fb.rating,
                    "thumbs_up": fb.thumbs_up,
                    "comment": fb.comment,
                    "created_at": fb.created_at.isoformat() if fb.created_at else None,
                }
                for fb in bad_cases
            ],
            "total": len(bad_cases),
        }
    finally:
        db.close()
