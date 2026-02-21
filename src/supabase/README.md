# Supabase Database Setup

This directory contains SQL migration files and initialization scripts for the Knowledge Graph RAG API database.

## Files

### Migration Files (run in order)
- `00_extensions.sql` - PostgreSQL extensions (vector, pgcrypto)
- `01_types.sql` - Custom enum types
- `02_documents.sql` - Documents table
- `03_chunks.sql` - Chunks table with embeddings
- `04_kg_nodes.sql` - Knowledge graph nodes table
- `05_kg_edges.sql` - Knowledge graph edges table
- `06_evidence_node.sql` - Node evidence table
- `07_evidence_edge.sql` - Edge evidence table
- `08_triggers_updated_at.sql` - Auto-update triggers
- `09_upsert_rpc.sql` - Upsert functions for chunks, nodes, edges
- `09b_fetch_chunks_rpc.sql` - Fetch chunks with embeddings function
- `10_pruning.sql` - Pruning functions for stale data
- `11_search_kg_nodes_rpc.sql` - Vector search function for KG nodes

### Initialization Script
- `init.sql` - **Complete initialization script** that runs all migrations in order. Use this for fresh database setup.

## Setup Instructions

### Option 1: Run Complete Initialization (Recommended for New Databases)

1. Open your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Copy and paste the contents of `init.sql`
4. Click **Run** to execute the script
5. Verify all tables, functions, and indexes were created successfully

### Option 2: Run Migrations Individually

If you prefer to run migrations one at a time (useful for tracking changes):

1. Run each migration file in numerical order (00 → 11)
2. Verify each migration completes successfully before proceeding

### Post-Setup Steps

#### 1. Create Storage Buckets

The application requires a storage bucket for PDF files. Create it via:

**Supabase Dashboard:**
1. Go to **Storage** in your Supabase dashboard
2. Click **New bucket**
3. Name: `pdf`
4. Set as **Private** (recommended) or **Public** based on your needs
5. Click **Create bucket**

**Or via SQL:**
```sql
-- Note: Storage bucket creation typically requires Supabase admin API
-- Use the dashboard or Supabase client library instead
```

#### 2. Configure Environment Variables

Ensure your `.env` file contains:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
OPENAI_API_KEY=your-openai-api-key
```

#### 3. Verify Setup

Run a quick test query to verify the schema:
```sql
-- Check tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check functions exist
SELECT routine_name 
FROM information_schema.routines 
WHERE routine_schema = 'public' 
AND routine_type = 'FUNCTION'
ORDER BY routine_name;
```

## Schema Overview

### Core Tables

- **documents** - Metadata about ingested documents (PDFs, DOCX, web pages)
- **chunks** - Text chunks extracted from documents with vector embeddings
- **kg_nodes** - Knowledge graph nodes representing concepts/chunks
- **kg_edges** - Relationships between knowledge graph nodes
- **kg_node_evidence** - Links nodes to source chunks
- **kg_edge_evidence** - Links edges to source chunks

### Key Features

- **Multi-tenancy**: All tables support `tenant_id` and `client_id` for data isolation
- **Vector Search**: Uses pgvector extension for semantic similarity search
- **Automatic Timestamps**: `created_at` and `updated_at` managed automatically
- **Soft Deletes**: Nodes/edges use status flags (`is_active`, `status`) instead of hard deletes
- **Evidence Tracking**: Links graph elements back to source chunks for explainability

### RPC Functions

- `upsert_chunk()` - Upsert chunks with embeddings
- `upsert_kg_node()` - Upsert knowledge graph nodes
- `upsert_kg_edge()` - Upsert knowledge graph edges
- `fetch_chunks_with_embeddings()` - Fetch chunks with embeddings (server-side JOIN)
- `search_kg_nodes()` - Vector similarity search over KG nodes
- `prune_kg()` - Archive stale nodes/edges and trim evidence

## Troubleshooting

### Extension Errors
If you see errors about missing extensions:
- Ensure you have admin access to your Supabase project
- Extensions may need to be enabled via Supabase dashboard → Database → Extensions

### Permission Errors
- Ensure you're using the **service role key** (not anon key) for admin operations
- Check that your database user has necessary permissions

### Vector Index Errors
- HNSW indexes are created automatically but may take time on large datasets
- If index creation fails, check available disk space and memory

## Maintenance

### Regular Pruning
Run the pruning function periodically to archive stale data:
```sql
SELECT public.prune_kg(
  'your-tenant-id'::uuid,
  'your-client-id'::uuid,
  90,  -- edge_stale_days
  180, -- node_stale_days
  3,   -- min_degree
  5,   -- keep_edge_evidence
  10   -- keep_node_evidence
);
```

### Monitoring
Check table sizes and growth:
```sql
SELECT 
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```


