# SQC — AI Security Questionnaire Copilot

Evidence-grounded answering engine for enterprise security questionnaires.
Core principle: **never guess.** Unsupported questions are refused, not answered.

## Status
- Step 1 complete: schema + forced-RLS tenant isolation, verified against Postgres 16 / pgvector.
- Step 2 complete: PDF / DOCX / TXT parsing and section-aware small-to-big chunking.
- Step 3 complete: embedding / rerank / LLM provider protocols, Voyage + Anthropic
  clients, and deterministic fakes so the full suite runs offline with no API keys.
- Next: step 4, ingestion into Postgres (first step needing a real embedding key).

## Providers
Set in `.env`. Defaults are `fake`, so `pytest` never needs a key or a network.
For real runs set `SQC_EMBEDDING_PROVIDER=voyage`, `SQC_RERANK_PROVIDER=voyage`,
`SQC_LLM_PROVIDER=anthropic` and supply `VOYAGE_API_KEY` and `ANTHROPIC_API_KEY`.
`SQC_EMBEDDING_DIM` must match the dimension the schema was migrated with;
changing it requires re-running `scripts/init_db.py` and re-embedding.

### Known gap
Every parser test runs against generated PDFs and DOCX files. Heading detection
has not yet been proven on a real vendor security policy, where headers, footers,
watermarks, two-column layouts and scanned pages appear. Drop a real policy PDF in
`tests/data/` and the ingestion smoke test should be extended to cover it before
step 4 loads anything into the database.

## Local setup
```bash
# 1. Postgres 16 + pgvector
sudo apt install postgresql-16 postgresql-16-pgvector

# 2. Roles (once, as superuser) and database
sudo -u postgres psql -f scripts/bootstrap.sql
sudo -u postgres createdb -O sqc sqc
sudo -u postgres psql -d sqc -c "ALTER SCHEMA public OWNER TO sqc"
sudo -u postgres psql -c "ALTER ROLE sqc_app PASSWORD 'devpass'"

# 3. Python
pip install -e ".[dev]"
cp .env.example .env        # then fill in keys

# 4. Schema
python scripts/init_db.py
pytest
```

## Security model
Tenant isolation is enforced by PostgreSQL row-level security with
`FORCE ROW LEVEL SECURITY`, so even the schema owner is subject to it.
The application connects as `sqc_app`, which owns nothing. There is no
code path anywhere that writes tenant data without a tenant scope.
`tenant_session()` is the only way in; without it, queries return zero rows.
