"""Supabase client singleton — import get_supabase() anywhere."""
from __future__ import annotations

import os
from functools import lru_cache

import dotenv
from supabase import Client, create_client

dotenv.load_dotenv()


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    return create_client(url, key)
