from fastapi import APIRouter

from routers.auth import router as auth_router
from routers.chat import router as chat_router
from routers.documents import router as documents_router
from routers.openai_compatible import router as openai_router
from routers.sessions import router as sessions_router
from routers.user import router as user_router

router = APIRouter()

router.include_router(auth_router)
router.include_router(chat_router)
router.include_router(documents_router)
router.include_router(openai_router)
router.include_router(sessions_router)
router.include_router(user_router)
