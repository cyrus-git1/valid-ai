"""
/privacy router
---------------
PII vault operations:

  POST /privacy/reveal   — alias → original (heavily audited)
  DELETE /privacy/erase  — remove an alias from the vault + redact references

These endpoints are GATED to the `admin` scope. Both write an audit_log row
with action=REVEAL or action=ERASE and the caller-supplied `reason`.

Encryption / decryption itself is NOT performed in Postgres. The vault stores
ciphertext; the application layer holds the wrapped DEK and unwraps via KMS.
At the time of writing, the actual KMS integration is a stub — wire it up
to your KMS of choice (AWS KMS, GCP KMS, Vault Transit) by implementing
`_decrypt_with_tenant_dek` below.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.logging_config import get_logger
from src.services.audit_service import AuditService
from src.supabase.supabase_client import get_supabase

logger = get_logger(__name__)
router = APIRouter(prefix="/privacy", tags=["privacy"])


# ── Request / response models ───────────────────────────────────────────────


class RevealRequest(BaseModel):
    tenant_id: UUID
    alias: str = Field(min_length=1)
    reason: str = Field(min_length=4, description="Required for SOC 2 audit trail.")
    actor_id: Optional[str] = None


class RevealResponse(BaseModel):
    alias: str
    original: str
    pii_type: str
    first_seen_at: Optional[str] = None
    seen_count: int = 0


class EraseRequest(BaseModel):
    tenant_id: UUID
    alias: Optional[str] = None
    actor_id: Optional[str] = Field(
        default=None,
        description="If set, erases all aliases originally tied to this actor (subject erasure).",
    )
    reason: str = Field(min_length=4)


class EraseResponse(BaseModel):
    aliases_erased: int
    chunks_redacted: int


# ── Helpers ─────────────────────────────────────────────────────────────────


def _require_admin(request: Request) -> str:
    scopes = getattr(request.state, "scopes", []) or []
    if "admin" not in scopes:
        raise HTTPException(status_code=403, detail="admin scope required for /privacy operations")
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="authenticated tenant required")
    return str(tenant_id)


def _decrypt_with_tenant_dek(
    *, tenant_id: str, ciphertext: bytes, rotation_version: int
) -> str:
    """
    Decrypt PII ciphertext using the tenant's DEK.

    STUB: replace with your KMS integration.
      1. Look up tenant_keys row by (tenant_id, rotation_version)
      2. Call KMS to unwrap `wrapped_dek` -> raw DEK
      3. AES-GCM decrypt `ciphertext` with the raw DEK
      4. Zero out the raw DEK from memory
      5. Return UTF-8 plaintext

    Until that's wired up, this raises so we never accidentally serve fake
    plaintext.
    """
    raise HTTPException(
        status_code=501,
        detail="KMS integration not configured — implement _decrypt_with_tenant_dek before enabling /privacy/reveal in production.",
    )


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/reveal", response_model=RevealResponse)
def reveal(body: RevealRequest, request: Request) -> RevealResponse:
    """Reveal the original PII behind an alias. Heavily audited."""
    tenant_id = _require_admin(request)
    if str(body.tenant_id) != tenant_id:
        raise HTTPException(status_code=403, detail="Tenant mismatch")

    sb = get_supabase()

    # Fetch the encrypted row
    res = (
        sb.table("pii_vault")
        .select("alias, encrypted_original, pii_type, rotation_version, first_seen_at, seen_count")
        .eq("tenant_id", tenant_id)
        .eq("alias", body.alias)
        .limit(1).execute()
    )
    rows = res.data or []
    if not rows:
        raise HTTPException(status_code=404, detail=f"alias not found: {body.alias}")
    row = rows[0]

    ciphertext = row["encrypted_original"]
    if isinstance(ciphertext, str):
        # Supabase REST returns bytea as hex-prefixed string ("\\x...") — convert
        if ciphertext.startswith("\\x"):
            ciphertext = bytes.fromhex(ciphertext[2:])
        else:
            ciphertext = ciphertext.encode("utf-8")

    try:
        original = _decrypt_with_tenant_dek(
            tenant_id=tenant_id,
            ciphertext=ciphertext,
            rotation_version=row.get("rotation_version", 1),
        )
    except HTTPException:
        # Always audit reveal attempts, even failures
        AuditService(sb).record(
            request=request,
            action="REVEAL",
            resource_type="pii_alias",
            resource_id=body.alias,
            status="failure",
            metadata={"reason": body.reason, "error": "decrypt_unavailable"},
        )
        raise

    AuditService(sb).record(
        request=request,
        action="REVEAL",
        resource_type="pii_alias",
        resource_id=body.alias,
        status="success",
        metadata={"reason": body.reason, "actor_id": body.actor_id},
    )
    logger.info(
        "privacy.reveal tenant=%s alias=%s actor=%s",
        tenant_id, body.alias, body.actor_id,
    )

    return RevealResponse(
        alias=row["alias"],
        original=original,
        pii_type=row["pii_type"],
        first_seen_at=row.get("first_seen_at"),
        seen_count=row.get("seen_count", 0),
    )


@router.post("/erase", response_model=EraseResponse)
def erase(body: EraseRequest, request: Request) -> EraseResponse:
    """
    Erase a PII alias (or all aliases tied to an actor) and redact references
    in chunks.pii_annotations. Required for GDPR right-to-erasure.
    """
    tenant_id = _require_admin(request)
    if str(body.tenant_id) != tenant_id:
        raise HTTPException(status_code=403, detail="Tenant mismatch")
    if not body.alias and not body.actor_id:
        raise HTTPException(status_code=400, detail="alias or actor_id required")

    sb = get_supabase()

    # Determine the list of aliases to erase
    aliases_to_erase: List[str] = []
    if body.alias:
        aliases_to_erase = [body.alias]
    else:
        # Derive aliases from chunks.pii_annotations referencing this actor.
        # The agent's PII redaction layer is responsible for tagging annotations
        # with `subject_actor_id` when applicable.
        try:
            res = (
                sb.table("chunks")
                .select("pii_annotations")
                .filter(
                    "pii_annotations", "cs",
                    f'[{{"subject_actor_id":"{body.actor_id}"}}]'
                )
                .execute()
            )
            for r in res.data or []:
                for ann in r.get("pii_annotations") or []:
                    if isinstance(ann, dict) and ann.get("alias"):
                        aliases_to_erase.append(ann["alias"])
        except Exception as e:
            logger.warning("privacy.erase actor scan failed: %s", e)

    aliases_to_erase = list(set(aliases_to_erase))

    # 1. Delete vault rows
    aliases_erased = 0
    for alias in aliases_to_erase:
        try:
            sb.rpc("erase_pii_alias", {"p_tenant_id": tenant_id, "p_alias": alias}).execute()
            aliases_erased += 1
        except Exception as e:
            logger.warning("privacy.erase vault delete failed alias=%s: %s", alias, e)

    # 2. Redact chunks.pii_annotations (zero out alias/spans referencing erased aliases)
    chunks_redacted = 0
    if aliases_to_erase:
        try:
            res = sb.table("chunks").select("id, tenant_id, pii_annotations").execute()
            for r in res.data or []:
                ann = r.get("pii_annotations") or []
                if not ann:
                    continue
                new_ann = [
                    a for a in ann
                    if not (isinstance(a, dict) and a.get("alias") in aliases_to_erase)
                ]
                if len(new_ann) != len(ann):
                    sb.table("chunks").update({"pii_annotations": new_ann}).eq("id", r["id"]).execute()
                    chunks_redacted += 1
        except Exception as e:
            logger.warning("privacy.erase chunk redaction failed: %s", e)

    AuditService(sb).record(
        request=request,
        action="ERASE",
        resource_type="pii_alias",
        resource_id=",".join(aliases_to_erase) if aliases_to_erase else (body.actor_id or ""),
        status="success",
        metadata={
            "reason": body.reason,
            "actor_id": body.actor_id,
            "aliases_erased": aliases_erased,
            "chunks_redacted": chunks_redacted,
        },
    )
    logger.info(
        "privacy.erase tenant=%s aliases=%d chunks=%d",
        tenant_id, aliases_erased, chunks_redacted,
    )

    return EraseResponse(
        aliases_erased=aliases_erased,
        chunks_redacted=chunks_redacted,
    )
