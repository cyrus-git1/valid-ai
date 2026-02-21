from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from supabase import Client

# ✅ Your existing modules
# Adjust these import paths as needed
from app.tokenization import document_bytes_to_chunks  # tokenization.py
from app.helpers import embed_texts  # helpers.py


JsonDict = Dict[str, Any]


@dataclass
class IngestInput:
    tenant_id: UUID
    client_id: UUID
    document_uri: str
    metadata: JsonDict

    # optional
    title: Optional[str] = None
    source_type: str = "bucket"  # "bucket" | "pdf" | "docx" (your choice)

    # behavior
    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64
    prune_after_ingest: bool = False


@dataclass
class IngestOutput:
    document_id: UUID
    chunks_upserted: int
    chunk_ids: List[UUID]
    warnings: List[str]
    prune_result: Optional[JsonDict] = None


class IngestController:
    """
    Route-free controller (call it from FastAPI route OR background worker).
    """

    def __init__(self, supabase: Client):
        self.sb = supabase

    # -------------------------
    # Storage helpers
    # -------------------------
    def _parse_bucket_uri(self, document_uri: str) -> Tuple[str, str, str]:
        """
        Accepts:
          - "bucket:mybucket/path/to/file.pdf"
          - "mybucket/path/to/file.pdf"
          - full Supabase Storage URL:
              .../storage/v1/object/public/<bucket>/<path>
              .../storage/v1/object/sign/<bucket>/<path>?token=...

        Returns: (bucket, path, file_type)
        """
        uri = document_uri.strip()

        if uri.startswith("bucket:"):
            uri = uri[len("bucket:") :]

        if "/storage/v1/object/" in uri:
            after = uri.split("/storage/v1/object/", 1)[1]
            parts = after.split("/", 2)
            if len(parts) < 3:
                raise ValueError(f"Unrecognized storage URL format: {document_uri}")
            bucket = parts[1]
            path = parts[2].split("?", 1)[0]
        else:
            if "/" not in uri:
                raise ValueError(
                    "document_uri must be 'bucket/path/to/file' or a Supabase storage URL"
                )
            bucket, path = uri.split("/", 1)

        file_type = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        return bucket, path, file_type

    def download_from_storage(self, document_uri: str) -> Tuple[bytes, str, str, str]:
        """
        Returns: (file_bytes, file_type, bucket, path)
        """
        bucket, path, file_type = self._parse_bucket_uri(document_uri)

        data = self.sb.storage.from_(bucket).download(path)
        if not isinstance(data, (bytes, bytearray)):
            raise RuntimeError(f"Unexpected storage download type: {type(data)}")

        return bytes(data), file_type, bucket, path

    # -------------------------
    # Documents
    # -------------------------
    def upsert_document(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        source_type: str,
        source_uri: str,
        title: Optional[str],
        metadata: JsonDict,
    ) -> UUID:
        """
        Idempotent upsert.
        Requires unique index on documents:
          (tenant_id, client_id, source_uri)
        """
        payload = {
            "tenant_id": str(tenant_id),
            "client_id": str(client_id),
            "source_type": source_type,
            "source_uri": source_uri,
            "title": title,
            "metadata": metadata or {},
        }

        res = self.sb.table("documents").upsert(
            payload,
            on_conflict="tenant_id,client_id,source_uri",
        ).execute()

        if not res.data:
            raise RuntimeError("documents upsert returned no rows")

        return UUID(res.data[0]["id"])

    # -------------------------
    # Embedding helpers
    # -------------------------
    def embed_in_batches(
        self,
        texts: List[str],
        *,
        model: str,
        batch_size: int,
    ) -> List[List[float]]:
        """
        Uses your helpers.py embed_texts() in batches to avoid timeouts.
        """
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = embed_texts(batch, model=model)
            out.extend(embs)
        return out

    # -------------------------
    # Chunks
    # -------------------------
    def upsert_chunk(
        self,
        *,
        tenant_id: UUID,
        document_id: UUID,
        chunk_index: int,
        start_page: Optional[int],
        end_page: Optional[int],
        text: str,
        token_count: Optional[int],
        metadata: JsonDict,
        embedding: Optional[List[float]],
    ) -> UUID:
        """
        Uses RPC upsert_chunk (recommended).
        """
        res = self.sb.rpc(
            "upsert_chunk",
            {
                "p_tenant_id": str(tenant_id),
                "p_document_id": str(document_id),
                "p_chunk_index": chunk_index,
                "p_page_start": start_page,
                "p_page_end": end_page,
                "p_content": text,
                "p_content_tokens": token_count,
                "p_metadata": metadata or {},
                "p_embedding": embedding,
            },
        ).execute()

        return UUID(str(res.data))

    # -------------------------
    # Optional pruning
    # -------------------------
    def prune_kg(self, *, tenant_id: UUID, client_id: UUID) -> JsonDict:
        res = self.sb.rpc(
            "prune_kg",
            {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_edge_stale_days": 90,
                "p_node_stale_days": 180,
                "p_min_degree": 3,
                "p_keep_edge_evidence": 5,
                "p_keep_node_evidence": 10,
            },
        ).execute()
        return res.data or {}

    # -------------------------
    # Main entry
    # -------------------------
    def ingest(self, inp: IngestInput) -> IngestOutput:
        warnings: List[str] = []
        chunk_ids: List[UUID] = []

        # 1) Download bytes
        file_bytes, file_type, bucket, path = self.download_from_storage(inp.document_uri)

        # 2) Validate file type
        if file_type not in {"pdf", "docx"}:
            raise ValueError(f"Unsupported file type '{file_type}'. Only pdf/docx supported right now.")

        # 3) Upsert document row
        # Store bucket/path in metadata so you can re-process later without parsing URL
        doc_meta = {
            **(inp.metadata or {}),
            "bucket": bucket,
            "object_path": path,
            "file_type": file_type,
        }
        document_id = self.upsert_document(
            tenant_id=inp.tenant_id,
            client_id=inp.client_id,
            source_type=inp.source_type,
            source_uri=inp.document_uri,
            title=inp.title,
            metadata=doc_meta,
        )

        # 4) Chunk using tokenization.py (your code)
        chunks = document_bytes_to_chunks(file_bytes, file_type=file_type)
        if not chunks:
            return IngestOutput(
                document_id=document_id,
                chunks_upserted=0,
                chunk_ids=[],
                warnings=["No chunks produced by tokenizer."],
            )

        # 5) Embed chunk texts in batches
        texts = [c["text"] for c in chunks]
        try:
            embeddings = self.embed_in_batches(
                texts,
                model=inp.embed_model,
                batch_size=inp.embed_batch_size,
            )
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(embeddings)} embeddings for {len(chunks)} chunks"
            )

        # 6) Upsert chunks with embeddings
        for idx, (chunk_data, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                chunk_id = self.upsert_chunk(
                    tenant_id=inp.tenant_id,
                    document_id=document_id,
                    chunk_index=idx,
                    start_page=chunk_data.get("start_page"),
                    end_page=chunk_data.get("end_page"),
                    text=chunk_data["text"],
                    token_count=chunk_data.get("token_count"),
                    metadata={
                        "document_uri": inp.document_uri,
                        "file_type": file_type,
                        "chunk_start_page": chunk_data.get("start_page"),
                        "chunk_end_page": chunk_data.get("end_page"),
                        **(inp.metadata or {}),
                    },
                    embedding=embedding,
                )
                chunk_ids.append(chunk_id)
            except Exception as e:
                warnings.append(f"chunk {idx} upsert failed: {e}")

        # 7) Optional prune
        prune_result = None
        if inp.prune_after_ingest:
            try:
                prune_result = self.prune_kg(tenant_id=inp.tenant_id, client_id=inp.client_id)
            except Exception as e:
                warnings.append(f"prune_kg failed: {e}")

        return IngestOutput(
            document_id=document_id,
            chunks_upserted=len(chunk_ids),
            chunk_ids=chunk_ids,
            warnings=warnings,
            prune_result=prune_result,
        )