import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth import authenticate_user, create_access_token, get_current_user, get_db, get_password_hash, resolve_role
from core.rate_limiter import LocalRateLimiter
from models import User
from schemas import (
    AuthResponse,
    CurrentUserResponse,
    LoginRequest,
    RegisterRequest,
)
from agentic_rag.config import budget as budget_cfg

router = APIRouter()

_auth_limiter = LocalRateLimiter(max_requests=10, window_seconds=60)


def _get_client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/auth/register", response_model=AuthResponse)
async def register(request: Request, body: RegisterRequest, db: Session = Depends(get_db)):
    client_ip = _get_client_ip(request)
    if not _auth_limiter.is_allowed(f"register:{client_ip}"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    username = (body.username or "").strip()
    password = body.password
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    role = resolve_role(body.role, body.admin_code)
    password_hash = await asyncio.get_running_loop().run_in_executor(None, get_password_hash, password)
    user = User(username=username, password_hash=password_hash, role=role)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="注册失败，请尝试其他用户名")

    token = create_access_token(username=username, role=role)
    return AuthResponse(access_token=token, username=username, role=role)


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    client_ip = _get_client_ip(request)
    if not _auth_limiter.is_allowed(f"login:{client_ip}"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    user = await asyncio.get_running_loop().run_in_executor(None, authenticate_user, db, body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(username=user.username, role=user.role)
    return AuthResponse(access_token=token, username=user.username, role=user.role)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return CurrentUserResponse(username=current_user.username, role=current_user.role, budget_max=budget_cfg.MAX_ITERATIONS)


@router.post("/auth/refresh", response_model=AuthResponse)
async def refresh_token(request: Request, current_user: User = Depends(get_current_user)):
    client_ip = _get_client_ip(request)
    if not _auth_limiter.is_allowed(f"refresh:{client_ip}"):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    token = create_access_token(username=current_user.username, role=current_user.role)
    return AuthResponse(access_token=token, username=current_user.username, role=current_user.role)
