"""
Secrets accessor with pluggable backends.

Backends (selected via SECRETS_BACKEND env var):
  env       — default, reads from os.environ (dev)
  doppler   — reads from Doppler CLI-injected env
  aws       — aws-secretsmanager (requires boto3 + IAM credentials)
  vault     — HashiCorp Vault (requires hvac + VAULT_ADDR/VAULT_TOKEN)

The goal is to keep application code oblivious to the backend. Anywhere you
would `os.environ["OPENAI_API_KEY"]`, use `get_secret("OPENAI_API_KEY")`
instead. Values are cached for the lifetime of the process unless
SECRETS_TTL_SECONDS is set, in which case they expire after that window.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

from src.logging_config import get_logger

logger = get_logger(__name__)

_cache: dict[str, tuple[float, str]] = {}
_cache_lock = threading.Lock()
_TTL = int(os.environ.get("SECRETS_TTL_SECONDS", "0"))  # 0 = cache forever


class SecretNotFound(Exception):
    pass


def _env_backend(name: str) -> Optional[str]:
    return os.environ.get(name)


def _aws_backend(name: str) -> Optional[str]:
    try:
        import boto3  # type: ignore
    except ImportError as e:
        raise RuntimeError("aws backend requires boto3") from e
    region = os.environ.get("AWS_REGION", "us-east-1")
    prefix = os.environ.get("SECRETS_AWS_PREFIX", "").rstrip("/")
    secret_id = f"{prefix}/{name}" if prefix else name
    client = boto3.client("secretsmanager", region_name=region)
    try:
        resp = client.get_secret_value(SecretId=secret_id)
    except client.exceptions.ResourceNotFoundException:
        return None
    return resp.get("SecretString")


def _doppler_backend(name: str) -> Optional[str]:
    # Doppler injects secrets as env vars; treat identical to env backend.
    return os.environ.get(name)


def _vault_backend(name: str) -> Optional[str]:
    try:
        import hvac  # type: ignore
    except ImportError as e:
        raise RuntimeError("vault backend requires hvac") from e
    addr = os.environ.get("VAULT_ADDR")
    token = os.environ.get("VAULT_TOKEN")
    if not addr or not token:
        return None
    client = hvac.Client(url=addr, token=token)
    mount = os.environ.get("SECRETS_VAULT_MOUNT", "secret")
    path = os.environ.get("SECRETS_VAULT_PATH", "app")
    try:
        resp = client.secrets.kv.v2.read_secret_version(mount_point=mount, path=path)
        return (resp.get("data", {}).get("data") or {}).get(name)
    except Exception:
        return None


_BACKENDS: dict[str, Callable[[str], Optional[str]]] = {
    "env": _env_backend,
    "doppler": _doppler_backend,
    "aws": _aws_backend,
    "vault": _vault_backend,
}


def _active_backend() -> Callable[[str], Optional[str]]:
    name = os.environ.get("SECRETS_BACKEND", "env").lower()
    fn = _BACKENDS.get(name)
    if fn is None:
        logger.warning("secrets_backend_unknown_fallback_env", backend=name)
        return _env_backend
    return fn


def get_secret(name: str, *, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Fetch a secret by name. Cached per-process for performance.

    If required=True and the secret is missing, raises SecretNotFound.
    """
    now = time.time()
    with _cache_lock:
        cached = _cache.get(name)
        if cached and (_TTL == 0 or now - cached[0] < _TTL):
            return cached[1]

    try:
        value = _active_backend()(name)
    except Exception as e:
        logger.warning("secrets_lookup_failed", name=name, error=str(e))
        value = None

    if value is None:
        value = default

    if value is None and required:
        raise SecretNotFound(f"Secret {name!r} not found and no default provided.")

    if value is not None:
        with _cache_lock:
            _cache[name] = (now, value)

    return value


def clear_cache() -> None:
    """Drop all cached secrets (e.g. after rotation)."""
    with _cache_lock:
        _cache.clear()
