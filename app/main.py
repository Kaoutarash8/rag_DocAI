import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy import text

from app.api import admin, ask, auth, documents
from app.core import metrics
from app.core.config import get_settings
from app.core.errors import LLMError, ProfileError
from app.core.logs import setup_logging
from app.database import SessionLocal, init_db
from app.services import processing

settings = get_settings()
setup_logging()
log = logging.getLogger("app")
STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.auto_create_tables:
        init_db()
    metrics.register_queue_collector()
    processing.start()  # sans effet avec QUEUE_BACKEND=celery (les workers s'en chargent)
    if settings.warmup_models:
        from app.services import embeddings, profiles
        for sp in profiles.index_specs():  # un modèle par collection indexée
            threading.Thread(target=embeddings.get_model, args=(sp.embedding_model,), daemon=True).start()
    yield
    processing.stop()


is_prod = settings.env == "production"
app = FastAPI(
    title="Assistant Documentaire IA", version="2.3.0", lifespan=lifespan,
    docs_url=None if is_prod else "/docs", redoc_url=None, openapi_url=None if is_prod else "/openapi.json",
)

if settings.cors_list:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list,
                       allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Total-Count"])


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("Erreur non gérée [%s] %s %s", rid, request.method, request.url.path)
        metrics.ERRORS.labels("internal").inc()
        response = JSONResponse({"detail": "Erreur interne du serveur", "request_id": rid}, 500)
    elapsed = time.perf_counter() - start
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    route = getattr(request.scope.get("route"), "path", None) or "unmatched"  # gabarit : cardinalité bornée
    if route != "/metrics":
        metrics.observe_http(request.method, route, response.status_code, elapsed)
    log.info("%s %s -> %s (%.0f ms) [%s]", request.method, request.url.path,
             response.status_code, elapsed * 1000, rid,
             extra={"request_id": rid, "user_id": getattr(request.state, "user_id", None),
                    "method": request.method, "path": request.url.path, "route": route,
                    "status": response.status_code, "duration_ms": round(elapsed * 1000)})
    return response


@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError):
    metrics.ERRORS.labels("llm").inc()
    return JSONResponse({"detail": str(exc)}, status_code=502)


@app.exception_handler(ProfileError)
async def profile_error_handler(_: Request, exc: ProfileError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(ask.router)
app.include_router(admin.router)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_INDEX)


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics(request: Request):
    """Métriques Prometheus. À NE PAS exposer publiquement (le Caddyfile fourni bloque /metrics)."""
    token = settings.metrics_token
    if token and request.headers.get("authorization") != f"Bearer {token}":
        return JSONResponse({"detail": "Non autorisé"}, status_code=401)
    body, content_type = metrics.render()
    return Response(body, media_type=content_type)


@app.get("/health", tags=["system"])
def health():
    """Liveness : le process répond."""
    return {"status": "healthy"}


@app.get("/ready", tags=["system"])
def ready():
    """Readiness : base de données et vectorstore accessibles."""
    from app.services import vectorstore
    checks = {}
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {type(e).__name__}"
    try:
        checks["vectorstore_chunks"] = vectorstore.ping()
    except Exception as e:
        checks["vectorstore"] = f"error: {type(e).__name__}"
    ok = checks.get("database") == "ok" and "vectorstore_chunks" in checks
    return JSONResponse({"status": "ready" if ok else "degraded", **checks},
                        status_code=200 if ok else 503)
