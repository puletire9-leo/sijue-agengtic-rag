import os
import base64
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from database import SessionLocal
from models import User

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY environment variable is required")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
ADMIN_INVITE_CODE = os.getenv("ADMIN_INVITE_CODE", "")
PBKDF2_ROUNDS = min(1_000_000, max(100_000, int(os.getenv("PASSWORD_PBKDF2_ROUNDS", "600000"))))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# Sync SQLAlchemy session is intentionally used in async endpoints.
# Each auth request performs 1-2 row-level queries (microseconds); at the
# current single-user scale this does not meaningfully block the event loop.
# If the app grows to multi-user concurrent load, migrate to:
#   create_async_engine("postgresql+asyncpg://...") + AsyncSession.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password: str, password_hash: str) -> bool:
    if not plain_password or not password_hash:
        return False

    # New format: pbkdf2_sha256$<rounds>$<salt_b64>$<digest_b64>
    if password_hash.startswith("pbkdf2_sha256$"):
        try:
            _, rounds, salt_b64, digest_b64 = password_hash.split("$", 3)
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(digest_b64.encode("ascii"))
            calculated = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode("utf-8"),
                salt,
                int(rounds),
            )
            return hmac.compare_digest(calculated, expected)
        except Exception:
            return False

    # Backward compatibility for legacy passlib/bcrypt hashes.
    if password_hash.startswith("$2") or password_hash.startswith("$bcrypt"):
        try:
            from passlib.context import CryptContext

            legacy_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")
            return legacy_context.verify(plain_password, password_hash)
        except Exception:
            return False

    return False


def get_password_hash(password: str) -> str:
    if not password:
        raise ValueError("password is required")

    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ROUNDS,
    )
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt_b64}${digest_b64}"


_DUMMY_HASH = get_password_hash("dummy_password_placeholder")


def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        # Always run PBKDF2 with a dummy hash to prevent timing side-channel
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效或过期的认证令牌",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise credentials_exception
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="管理员权限不足")
    return current_user


# ── 统一认证：Open WebUI 为唯一身份源 ──

OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY", "")


async def get_openwebui_user(request: Request) -> SimpleNamespace:
    """从 Open WebUI 转发的 headers 中获取用户信息。

    认证方式：
    1. API Key + x-openwebui-user-* headers（Open WebUI 代理）
    2. SuperMew JWT（兼容直连）
    """
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth[7:].strip()

    if not token:
        raise HTTPException(status_code=401, detail="Empty token")

    # 方式1：API Key + headers（Open WebUI 代理）
    if OPENAI_COMPATIBLE_API_KEY and hmac.compare_digest(token, OPENAI_COMPATIBLE_API_KEY):
        return SimpleNamespace(
            id=request.headers.get("x-openwebui-user-id", ""),
            username=request.headers.get("x-openwebui-user-name", "openwebui_user"),
            role=request.headers.get("x-openwebui-user-role", "user"),
        )

    # 方式2：SuperMew JWT（兼容直连）
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub", "")
        role = payload.get("role", "user")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return SimpleNamespace(id="", username=username, role=role)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")


def resolve_role(requested_role: str | None, admin_code: str | None) -> str:
    role = (requested_role or "user").strip().lower()
    if role != "admin":
        return "user"
    if ADMIN_INVITE_CODE and hmac.compare_digest(admin_code or "", ADMIN_INVITE_CODE):
        return "admin"
    raise HTTPException(status_code=403, detail="管理员邀请码错误")
