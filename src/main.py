"""
main.py
-------
Valid Data Plane — pure storage and retrieval layer.

Owns the database (Supabase). Provides documents CRUD, KG node/edge CRUD,
vector search (semantic + graph), health/stats, and data endpoints for
the agent service.

All compute-heavy logic (ingest, context generation, RAG, panel filtering)
lives in the agent service (valid-agents, port 8003).

Run with:
    uvicorn src.main:app --reload --port 8000
"""
from __future__ import annotations

import os

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

dotenv.load_dotenv()

# Configure structured logging before importing anything that logs
from src.logging_config import configure_logging, get_logger

configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

from src.exception_handlers import register as register_exception_handlers
from src.middleware.auth import AuthMiddleware
from src.middleware.rate_limiter import RateLimiterMiddleware
from src.middleware.request_id import RequestIdMiddleware
from src.routers.admin_router import router as admin_router
from src.routers.data_router import router as data_router
from src.routers.ingest_router import router as ingest_router
from src.routers.search_router import router as search_router

app = FastAPI(
    title="Valid Data Plane",
    description="Pure data plane: documents, KG, vector search, health/stats.",
    version="4.0.0",
)

# ── CORS (locked down) ───────────────────────────────────────────────────────
# No more wildcard. Must be explicitly configured via CORS_ORIGINS (comma-sep).
# In dev, default to localhost.
_cors_raw = os.environ.get("CORS_ORIGINS", "").strip()
if _cors_raw:
    _origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
else:
    logger.warning("cors_origins_not_set_defaulting_to_localhost")
    _origins = ["http://localhost", "http://localhost:3000", "http://localhost:8003"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
)

# ── Middleware chain (outermost first) ───────────────────────────────────────
# Starlette runs them bottom-up; so last-added = innermost.
# Order we want at runtime: RequestId -> Auth -> RateLimit -> routes
app.add_middleware(RateLimiterMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(RequestIdMiddleware)

# ── Exception handlers ───────────────────────────────────────────────────────
register_exception_handlers(app)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(admin_router)
app.include_router(search_router)
app.include_router(data_router)
app.include_router(ingest_router)


@app.get("/", tags=["root"])
def root():
    return {
        "service": "Valid Data Plane",
        "version": "4.0.0",
        "docs": "/docs",
        "health": "/admin/health",
    }
