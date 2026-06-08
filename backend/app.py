from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import time

import api as api_module
from database import init_db
from metrics import REQUEST_COUNT, REQUEST_LATENCY, get_metrics
from telemetry import setup_tracing

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


def create_app() -> FastAPI:
    # 结构化日志（可通过 STRUCTURED_LOG=true 启用）
    if os.getenv("STRUCTURED_LOG", "false").lower() == "true":
        from core.structured_logging import setup_structured_logging
        setup_structured_logging()

    app = FastAPI(title="Cute Cat Bot API")

    @app.on_event("startup")
    async def _startup_init_db():
        setup_tracing()
        init_db()
        # 打印模型路由信息
        try:
            from agentic_rag.llm import get_model_routing_info, log_model_cost_info
            info = get_model_routing_info()
            print(f"[SuperMew] Model routing: {info['tiers_configured']}/3 tiers configured, "
                  f"{info['effective_models']} unique model(s)")
            print(f"  Tier 1 (powerful): {info['tier1_powerful']}")
            print(f"  Tier 2 (medium):   {info['tier2_medium']}")
            print(f"  Tier 3 (light):    {info['tier3_lightweight']}")
            log_model_cost_info()
        except Exception:
            pass

    # request_id 中间件（全链路串联）
    @app.middleware("http")
    async def request_id_middleware(request, call_next):
        import uuid
        rid = request.headers.get("x-request-id", uuid.uuid4().hex[:16])
        try:
            from core.structured_logging import set_request_id
            set_request_id(rid)
        except ImportError:
            pass
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # No-cache middleware for development
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        route = request.scope.get("route")
        path = route.path if route else "__unmatched__"
        if path != "/metrics":
            REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
            REQUEST_LATENCY.labels(endpoint=path).observe(duration)
        return response

    _metrics_token = os.getenv("METRICS_AUTH_TOKEN")

    @app.get("/metrics")
    async def metrics(request: Request):
        if _metrics_token:
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer ") or auth[7:].strip() != _metrics_token:
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Unauthorized")
        return PlainTextResponse(get_metrics(), media_type="text/plain")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.include_router(api_module.router)

    # 挂载评测和反馈路由
    from routers.evaluation import router as eval_router
    from routers.feedback import router as feedback_router
    from routers.guardrail_toggle import router as guardrail_router
    app.include_router(eval_router)
    app.include_router(feedback_router)
    app.include_router(guardrail_router)

    # serve frontend static files at root
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
