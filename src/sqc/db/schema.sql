-- SQC schema.
-- Tenant isolation is enforced by Postgres RLS, not by application WHERE clauses.
-- The app connects as sqc_app, which is NOT the table owner, and RLS is FORCEd
-- so that even the owner cannot bypass it. Default-deny: if app.tenant_id is
-- unset or blank, nullif(...) is NULL, every policy is false, zero rows return.
--
-- {{EMBEDDING_DIM}} is substituted at migration time from SQC_EMBEDDING_DIM.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------- tenants
CREATE TABLE IF NOT EXISTS tenants (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
-- No RLS on tenants: it is the tenant registry itself, admin-only.

-- -------------------------------------------------------------- documents
CREATE TABLE IF NOT EXISTS documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    filename        text NOT NULL,
    doc_type        text NOT NULL DEFAULT 'unknown',   -- policy | soc2 | dpa | questionnaire | other
    -- effective_date drives conflict resolution between an old and a new policy.
    -- Nullable on purpose: we must never invent a date we could not extract.
    effective_date  date,
    version_label   text,
    content_sha256  text NOT NULL,
    page_count      integer,
    status          text NOT NULL DEFAULT 'pending',   -- pending | indexed | failed
    error_message   text,
    uploaded_at     timestamptz NOT NULL DEFAULT now(),
    indexed_at      timestamptz,
    CONSTRAINT documents_status_ck CHECK (status IN ('pending', 'indexed', 'failed'))
);
-- Same file uploaded twice by one tenant is one document; two tenants may
-- legitimately hold the same file, so the constraint is per tenant.
CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_sha_uq
    ON documents (tenant_id, content_sha256);
CREATE INDEX IF NOT EXISTS documents_tenant_status_ix
    ON documents (tenant_id, status);

-- ----------------------------------------------------------------- chunks
-- Small-to-big: `text` is the embedded leaf chunk; `parent_text` is the wider
-- section handed to the LLM so qualifiers and exceptions are not lost.
CREATE TABLE IF NOT EXISTS chunks (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id        uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id      uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index      integer NOT NULL,

    text             text NOT NULL,          -- raw leaf text, shown in citations
    parent_text      text,                   -- enclosing section, sent to the LLM
    embed_text       text NOT NULL,          -- heading_path + text, what we embed

    heading_path     text[] NOT NULL DEFAULT '{}',
    -- heading_path joined to a scalar. Denormalised on purpose: array_to_string
    -- is STABLE, not IMMUTABLE, so Postgres rejects it inside a generated
    -- column. Ingestion writes both from the same source, kept in step by
    -- a CHECK that the scalar is empty exactly when the array is.
    heading_text     text NOT NULL DEFAULT '',
    section          text,
    page_start       integer,
    page_end         integer,
    token_count      integer NOT NULL DEFAULT 0,

    -- Set by the ingestion-time injection scanner. Suspect chunks are still
    -- stored (they are the customer's own data) but are excluded from evidence.
    injection_flags  text[] NOT NULL DEFAULT '{}',

    embedding        vector({{EMBEDDING_DIM}}),
    embedding_model  text,                   -- detect mismatched vectors, never compare across models

    -- Generated FTS column: heading path is weighted A, body text B, so a
    -- chunk under "Encryption at Rest" outranks a passing mention elsewhere.
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(heading_text, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(text, '')), 'B')
    ) STORED,

    created_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT chunks_doc_index_uq UNIQUE (document_id, chunk_index),
    CONSTRAINT chunks_heading_sync_ck
        CHECK ((heading_text = '') = (cardinality(heading_path) = 0))
);

CREATE INDEX IF NOT EXISTS chunks_tsv_ix       ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_tenant_ix    ON chunks (tenant_id);
CREATE INDEX IF NOT EXISTS chunks_document_ix  ON chunks (document_id);
-- HNSW over cosine distance. Built after bulk load in practice; harmless empty.
CREATE INDEX IF NOT EXISTS chunks_embedding_ix
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- ------------------------------------------------------------ answer runs
-- Audit trail. Every generated answer is reconstructable: what was asked,
-- what was retrieved, what the model said, what the validator decided,
-- and what a human did with it.
CREATE TABLE IF NOT EXISTS answer_runs (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id         uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    question          text NOT NULL,
    outcome           text NOT NULL,         -- answered | review_required | refused | error
    confidence_label  text,                  -- high | medium | review_required | refused
    confidence_score  double precision,
    answer_text       text,
    refusal_reason    text,

    retrieved_chunk_ids uuid[] NOT NULL DEFAULT '{}',
    citations           jsonb  NOT NULL DEFAULT '[]'::jsonb,
    signals             jsonb  NOT NULL DEFAULT '{}'::jsonb,  -- every confidence input, for tuning

    llm_model         text,
    embedding_model   text,
    prompt_version    text,
    latency_ms        integer,

    -- human review
    reviewed_by       text,
    reviewed_at       timestamptz,
    review_action     text,                  -- accepted | edited | rejected
    final_answer      text,

    created_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT answer_runs_outcome_ck
        CHECK (outcome IN ('answered', 'review_required', 'refused', 'error')),
    CONSTRAINT answer_runs_review_ck
        CHECK (review_action IS NULL OR review_action IN ('accepted', 'edited', 'rejected'))
);
CREATE INDEX IF NOT EXISTS answer_runs_tenant_created_ix
    ON answer_runs (tenant_id, created_at DESC);

-- ------------------------------------------------------------------- RLS
-- FORCE matters: without it the table owner silently bypasses every policy.
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['documents', 'chunks', 'answer_runs'] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
        EXECUTE format('DROP POLICY IF EXISTS %I_tenant_isolation ON %I', t, t);
        EXECUTE format($p$
            CREATE POLICY %I_tenant_isolation ON %I
            USING      (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
            WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
        $p$, t, t);
    END LOOP;
END $$;

-- --------------------------------------------------------------- app role
-- The sqc_app role is created once by scripts/bootstrap.sql, run by a
-- superuser. Migrations deliberately do NOT create roles: in production the
-- migration runner should not hold CREATEROLE.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sqc_app') THEN
        RAISE EXCEPTION
            'role sqc_app is missing - run scripts/bootstrap.sql as a superuser first';
    END IF;
END $$;

GRANT USAGE ON SCHEMA public TO sqc_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON documents, chunks, answer_runs TO sqc_app;
GRANT SELECT ON tenants TO sqc_app;
