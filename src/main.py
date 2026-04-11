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

import logging
import os

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

from src.routers.documents_router import router as documents_router
from src.routers.admin_router import router as admin_router
from src.routers.search_router import router as search_router
from src.routers.data_router import router as data_router
from src.routers.ingest_router import router as ingest_router

app = FastAPI(
    title="Valid Data Plane",
    description="Pure data plane: documents, KG, vector search, health/stats.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents_router)           # GET/DELETE /documents
app.include_router(admin_router)               # GET /admin/health, /admin/stats
app.include_router(search_router)              # POST /search/semantic, /search/graph
app.include_router(data_router)                # GET /data/* (for agent service)
app.include_router(ingest_router)              # POST /ingest/processed, /ingest/processed-web


@app.get("/", tags=["root"])
def root():
    return {
        "service": "Valid Data Plane",
        "version": "4.0.0",
        "docs": "/docs",
        "health": "/admin/health",
    }