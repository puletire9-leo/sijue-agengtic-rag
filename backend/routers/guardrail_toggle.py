"""护栏开关 API — 运行时切换，无需重启。"""
from fastapi import APIRouter, Depends
from auth import get_openwebui_user

router = APIRouter()

# 运行时状态
_guardrail_enabled = {"enabled": False}


@router.get("/guardrail/status")
async def get_guardrail_status(_=Depends(get_openwebui_user)):
    return {"enabled": _guardrail_enabled["enabled"]}


@router.post("/guardrail/toggle")
async def toggle_guardrail(_=Depends(get_openwebui_user)):
    _guardrail_enabled["enabled"] = not _guardrail_enabled["enabled"]
    return {"enabled": _guardrail_enabled["enabled"]}


def is_guardrail_enabled() -> bool:
    return _guardrail_enabled["enabled"]
