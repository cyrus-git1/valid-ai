"""
main.py
-------
FastAPI application entrypoint.

Registers all routers and configures CORS, logging, and startup checks.

Run with:
    uvicorn main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
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
from src.routers.kg_router import router as kg_router
from src.routers.search_router import router as search_router
from src.routers.admin_router import router as admin_router
from src.routers.context_router import router as context_router
from src.routers.survey_router import router as survey_router
from src.routers.context_summary_router import router as context_summary_router
from src.routers.strategic_analysis_router import router as strategic_analysis_router

app = FastAPI(
    title="Knowledge Graph RAG API",
    description=(
        "Ingest PDFs, DOCX files, and websites into a Supabase-backed "
        "Knowledge Graph, then query it with vector search, graph expansion, "
        "and LLM-powered question answering."
    ),
    version="1.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Tighten allowed_origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest_router)      # POST /ingest/file, POST /ingest/web
app.include_router(documents_router)   # GET/PATCH/DELETE /documents
app.include_router(kg_router)          # POST /kg/build, POST /kg/prune, GET /kg/nodes
app.include_router(search_router)      # POST /search/semantic, /search/graph, /search/ask
app.include_router(admin_router)       # GET /admin/health, /admin/stats, POST /admin/reindex
app.include_router(context_router)     # POST /context/build, GET /context/status
app.include_router(survey_router)      # POST /survey/generate  (direct — no classification)
app.include_router(context_summary_router)  # POST /context-summary/generate, /get, DELETE
app.include_router(strategic_analysis_router)  # POST /strategic-analysis/generate

@app.get("/", tags=["root"])
def root():
    return {
        "service": "Knowledge Graph RAG API",
        "docs": "/docs",
        "health": "/admin/health",
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
