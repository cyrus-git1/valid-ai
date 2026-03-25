"""
test_kg_evidence.py
-------------------
End-to-end test of KG evidence population and retrieval logging.

1. Rebuilds a small slice of the KG (max 10 chunks) to populate evidence tables
2. Verifies evidence rows were written
3. Runs a graph search and prints full result with evidence + retrieval reasons
"""
import json
import logging
import os
import sys
from uuid import UUID

import dotenv
dotenv.load_dotenv()

# Show retrieval logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from src.supabase.supabase_client import get_supabase
from src.models.domain.kg import KGBuildConfig
from src.services.kg_service import KGService
from src.services.search_service import SearchService

TENANT_ID = UUID("6d790e63-9263-45ec-8108-82eddf26311e")
CLIENT_ID = UUID("f01dea55-3a1f-4384-96f3-2dede8c36feb")


def main():
    sb = get_supabase()
    kg = KGService(sb)

    # -- Step 1: Rebuild a small slice to populate evidence --------------------
    print("\n" + "=" * 70)
    print("STEP 1: Rebuilding KG (max 10 chunks) to populate evidence tables")
    print("=" * 70)

    cfg = KGBuildConfig(max_chunks=10, batch_size=10, similarity_threshold=0.82)
    result = kg.build_kg_from_chunk_embeddings(
        tenant_id=TENANT_ID,
        client_id=CLIENT_ID,
        config=cfg,
    )
    print(f"\nBuild result: {json.dumps(result, indent=2)}")

    # -- Step 2: Verify evidence was written -----------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Checking evidence tables")
    print("=" * 70)

    ne = sb.table("kg_node_evidence").select("*", count="exact").eq("tenant_id", str(TENANT_ID)).limit(5).execute()
    ee = sb.table("kg_edge_evidence").select("*", count="exact").eq("tenant_id", str(TENANT_ID)).limit(5).execute()

    print(f"\nNode evidence rows: {ne.count}")
    if ne.data:
        for row in ne.data[:3]:
            print(f"  node={row['node_id'][:8]}...  chunk={row['chunk_id'][:8]}...  "
                  f"score={row['score']}  quote={row['quote'][:60] if row.get('quote') else 'N/A'}...")

    print(f"\nEdge evidence rows: {ee.count}")
    if ee.data:
        for row in ee.data[:3]:
            print(f"  edge={row['edge_id'][:8]}...  chunk={row['chunk_id'][:8]}...  "
                  f"score={row['score']}  quote={row['quote'][:60] if row.get('quote') else 'N/A'}...")

    # -- Step 3: Run a search and show evidence + retrieval reasons ------------
    print("\n" + "=" * 70)
    print("STEP 3: Graph search with evidence and retrieval logging")
    print("=" * 70)

    svc = SearchService(tenant_id=TENANT_ID, client_id=CLIENT_ID)
    query = "what services are available"
    print(f"\nQuery: {query!r}\n")

    docs = svc.graph_search(query, top_k=3, hop_limit=1)

    print(f"\n{'-' * 70}")
    print(f"Returned {len(docs)} documents")
    print(f"{'-' * 70}")

    for i, doc in enumerate(docs):
        m = doc.metadata
        print(f"\n[Result {i + 1}]")
        print(f"  node_id:          {m.get('node_id', 'N/A')[:12]}...")
        print(f"  node_key:         {m.get('node_key', 'N/A')}")
        print(f"  source:           {m.get('source', 'N/A')}")
        print(f"  similarity_score: {m.get('similarity_score', 'N/A')}")
        print(f"  retrieval_reason: {m.get('retrieval_reason', 'N/A')}")
        print(f"  evidence_quote:   {m.get('evidence_quote', 'N/A')[:80] if m.get('evidence_quote') else 'None'}...")
        print(f"  evidence_score:   {m.get('evidence_score', 'N/A')}")
        print(f"  evidence_count:   {m.get('evidence_count', 0)}")
        print(f"  content preview:  {doc.page_content[:100]}...")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
