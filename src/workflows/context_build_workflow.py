"""
src/workflows/context_build_workflow.py
-----------------------------------------
LangGraph StateGraph that orchestrates the full context build pipeline:

  Input JSON → validate → ingest all sources → build KG → return Documents

The terminal output is state["documents"] — a List[Document] that agents
can use immediately for retrieval and answer generation.

Usage
-----
    from src.workflows.context_build_workflow import build_context_graph

    app = build_context_graph()
    result = app.invoke({
        "tenant_id": "...",
        "client_id": "...",
        "docs": ["path/to/file.pdf"],
        "weblinks": ["https://example.com"],
        "client_profile": {...},
    })
    documents = result["documents"]  # List[Document]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.services.ingest_service import IngestInput, IngestOutput, IngestService
from src.services.kg_service import KGBuildConfig, KGService
from src.services.kg_retriever_service import KGRetrieverService
from src.supabase.supabase_client import get_supabase

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────────────

class ContextBuildState(TypedDict, total=False):
    tenant_id: str
    client_id: str
    docs: List[str]
    weblinks: List[str]
    transcripts: List[str]
    client_profile: Dict[str, Any]
    ingest_results: List[Dict[str, Any]]
    kg_build_result: Dict[str, Any]
    documents: List[Document]
    status: str
    error: Optional[str]
    warnings: List[str]


# ── Nodes ────────────────────────────────────────────────────────────────────

def validate_input(state: ContextBuildState) -> ContextBuildState:
    """Validate required fields and normalize input."""
    warnings: List[str] = []

    if not state.get("tenant_id"):
        return {**state, "status": "failed", "error": "tenant_id is required"}
    if not state.get("client_id"):
        return {**state, "status": "failed", "error": "client_id is required"}

    docs = state.get("docs", [])
    weblinks = state.get("weblinks", [])
    transcripts = state.get("transcripts", [])

    if not docs and not weblinks and not transcripts:
        return {**state, "status": "failed", "error": "At least one doc, weblink, or transcript required"}

    # Validate doc paths exist
    valid_docs = []
    for doc_path in docs:
        p = Path(doc_path)
        if p.exists():
            ext = p.suffix.lower()
            if ext in (".pdf", ".docx"):
                valid_docs.append(doc_path)
            else:
                warnings.append(f"Skipping unsupported file type: {doc_path}")
        else:
            warnings.append(f"File not found: {doc_path}")

    # Validate transcript paths exist
    valid_transcripts = []
    for vtt_path in transcripts:
        p = Path(vtt_path)
        if p.exists():
            if p.suffix.lower() == ".vtt":
                valid_transcripts.append(vtt_path)
            else:
                warnings.append(f"Skipping non-VTT transcript: {vtt_path}")
        else:
            warnings.append(f"Transcript not found: {vtt_path}")

    return {
        **state,
        "docs": valid_docs,
        "weblinks": weblinks,
        "transcripts": valid_transcripts,
        "warnings": warnings,
        "status": "validated",
    }


def ingest_sources(state: ContextBuildState) -> ContextBuildState:
    """Ingest all documents and weblinks into Supabase."""
    sb = get_supabase()
    svc = IngestService(sb)
    tenant_id = UUID(state["tenant_id"])
    client_id = UUID(state["client_id"])
    warnings = list(state.get("warnings", []))
    ingest_results: List[Dict[str, Any]] = []

    # Ingest documents
    for doc_path in state.get("docs", []):
        try:
            p = Path(doc_path)
            result = svc.ingest(IngestInput(
                tenant_id=tenant_id,
                client_id=client_id,
                file_bytes=p.read_bytes(),
                file_name=p.name,
                title=p.stem,
            ))
            ingest_results.append({
                "source": doc_path,
                "source_type": result.source_type,
                "document_id": str(result.document_id),
                "chunks_upserted": result.chunks_upserted,
            })
            warnings.extend(result.warnings)
        except Exception as e:
            warnings.append(f"Failed to ingest {doc_path}: {e}")
            logger.error("Ingest failed for %s: %s", doc_path, e)

    # Ingest transcripts (.vtt)
    for vtt_path in state.get("transcripts", []):
        try:
            p = Path(vtt_path)
            result = svc.ingest(IngestInput(
                tenant_id=tenant_id,
                client_id=client_id,
                file_bytes=p.read_bytes(),
                file_name=p.name,
                title=p.stem,
            ))
            ingest_results.append({
                "source": vtt_path,
                "source_type": result.source_type,
                "document_id": str(result.document_id),
                "chunks_upserted": result.chunks_upserted,
            })
            warnings.extend(result.warnings)
        except Exception as e:
            warnings.append(f"Failed to ingest transcript {vtt_path}: {e}")
            logger.error("Transcript ingest failed for %s: %s", vtt_path, e)

    # Ingest weblinks
    for url in state.get("weblinks", []):
        try:
            result = svc.ingest(IngestInput(
                tenant_id=tenant_id,
                client_id=client_id,
                web_url=url,
            ))
            ingest_results.append({
                "source": url,
                "source_type": "web",
                "document_id": str(result.document_id),
                "chunks_upserted": result.chunks_upserted,
            })
            warnings.extend(result.warnings)
        except Exception as e:
            warnings.append(f"Failed to ingest {url}: {e}")
            logger.error("Web ingest failed for %s: %s", url, e)

    if not ingest_results:
        return {**state, "status": "failed", "error": "All sources failed to ingest", "warnings": warnings}

    return {
        **state,
        "ingest_results": ingest_results,
        "warnings": warnings,
        "status": "ingested",
    }


def build_kg(state: ContextBuildState) -> ContextBuildState:
    """Build the knowledge graph from the ingested chunks."""
    sb = get_supabase()
    kg_svc = KGService(sb)
    tenant_id = UUID(state["tenant_id"])
    client_id = UUID(state["client_id"])
    warnings = list(state.get("warnings", []))

    try:
        kg_result = kg_svc.build_kg_from_chunk_embeddings(
            tenant_id=tenant_id,
            client_id=client_id,
            config=KGBuildConfig(),
        )
    except Exception as e:
        warnings.append(f"KG build failed: {e}")
        logger.error("KG build failed: %s", e)
        kg_result = {}

    # Convert to LangChain Documents using the retriever
    documents: List[Document] = []
    try:
        retriever = KGRetrieverService.from_env(
            tenant_id=tenant_id,
            client_id=client_id,
            top_k=50,
            hop_limit=0,
        )
        # Fetch all nodes for this client via a broad query
        # We use the retriever's internal methods for batch fetching
        all_nodes = retriever._vector_search(retriever._embed_query(""))
        for node in all_nodes:
            documents.append(
                retriever._node_to_document(node, similarity=node.get("similarity"), source="context_build")
            )
    except Exception as e:
        warnings.append(f"Document conversion failed: {e}")
        logger.error("Document conversion failed: %s", e)

    return {
        **state,
        "kg_build_result": kg_result,
        "documents": documents,
        "warnings": warnings,
        "status": "complete",
    }


def handle_error(state: ContextBuildState) -> ContextBuildState:
    """Terminal error handler."""
    logger.error("Context build failed: %s", state.get("error"))
    return {**state, "status": "failed"}


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_validate(state: ContextBuildState) -> str:
    if state.get("status") == "failed":
        return "handle_error"
    return "ingest_sources"


def route_after_ingest(state: ContextBuildState) -> str:
    if state.get("status") == "failed":
        return "handle_error"
    return "build_kg"


# ── Graph ────────────────────────────────────────────────────────────────────

def build_context_graph() -> StateGraph:
    """Build and compile the context build LangGraph."""
    graph = StateGraph(ContextBuildState)

    graph.add_node("validate_input", validate_input)
    graph.add_node("ingest_sources", ingest_sources)
    graph.add_node("build_kg", build_kg)
    graph.add_node("handle_error", handle_error)

    graph.set_entry_point("validate_input")

    graph.add_conditional_edges("validate_input", route_after_validate)
    graph.add_conditional_edges("ingest_sources", route_after_ingest)
    graph.add_edge("build_kg", END)
    graph.add_edge("handle_error", END)

    return graph.compile()
