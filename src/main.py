"""
main.py
-------
Core API — data plane.

Owns the database (Supabase). Handles ingestion, documents, KG, search,
context summaries, and panel participants.

Separate services:
  - valid-transcription (port 8001): audio/video transcription
  - valid-agents (port 8003): agents, surveys, analysis, enrichment

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

from src.routers.ingest_router import router as ingest_router
from src.routers.documents_router import router as documents_router
from src.routers.admin_router import router as admin_router
from src.routers.context_router import router as context_router
from src.routers.search_router import router as search_router
from src.routers.panel_participant_router import router as panel_participant_router
from src.routers.data_router import router as data_router

app = FastAPI(
    title="Valid Core API",
    description="Data plane: ingestion, KG, search, context, documents, panel participants.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest_router)              # POST /ingest/file, POST /ingest/web
app.include_router(documents_router)           # GET/PATCH/DELETE /documents
app.include_router(admin_router)               # GET /admin/health, /admin/stats
app.include_router(context_router)             # POST /context/build, GET /context/summary/*
app.include_router(search_router)              # POST /search/semantic, /search/graph, /search/ask
app.include_router(panel_participant_router)    # POST /panel/ingest, /panel/filter
app.include_router(data_router)                # GET /data/* (for agent service)


@app.get("/", tags=["root"])
def root():
    return {
        "service": "Valid Core API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/admin/health",
    }