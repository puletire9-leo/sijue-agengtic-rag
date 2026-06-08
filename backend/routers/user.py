import asyncio
import logging
import os

import requests
from fastapi import APIRouter, Depends, HTTPException, Request

from auth import get_current_user
from core.rate_limiter import LocalRateLimiter
from models import User

logger = logging.getLogger(__name__)

router = APIRouter()

_balance_limiter = LocalRateLimiter(max_requests=10, window_seconds=60)


def _fetch_balance() -> dict:
    """同步查询 DeepSeek API 账号余额（在线程池中运行）。"""
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        return {"error": "API key not configured"}
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com")
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/user/balance",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        logger.error("Balance API returned HTTP %s", resp.status_code)
        return {"error": "查询余额失败，请稍后再试"}
    except Exception:
        logger.exception("Failed to fetch balance")
        return {"error": "查询余额失败，请稍后再试"}


@router.get("/user/balance")
async def get_balance(request: Request, _: User = Depends(get_current_user)):
    """查询 DeepSeek API 账号余额"""
    client_ip = request.client.host if request.client else "unknown"
    if not _balance_limiter.is_allowed(f"balance:{client_ip}"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch_balance)
