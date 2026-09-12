# SQC — AI Security Questionnaire Copilot

Evidence-grounded answering engine for enterprise security questionnaires.
Core principle: **never guess.** Unsupported questions are refused, not answered.

## Status
- Step 1 complete: schema + forced-RLS tenant isolation, verified against Postgres 16 / pgvector.
- Step 2 complete: PDF / DOCX / TXT parsing and section-aware small-to-big chunking.
- Step 3 complete: embedding / rerank / LLM provider protocols, Voyage + Anthropic
  clients, and deterministic fakes so the full suite runs offline with no API keys.
- Step 4 complete: ingestion pipeline into Postgres, plus CLI tools.
- Step 5 complete: hybrid retrieval (full-text + vector + RRF + rerank + evidence packing).
- Step 6 complete: evidence-grounded answering, deterministic validator, refusal states.
- Next: step 7, evaluation harness and golden dataset (then threshold calibration).

## Answering
```bash
python scripts/answer.py --tenant $TENANT "Is MFA required for all accounts?"
python scripts/answer.py --tenant $TENANT --audit "Is data encrypted at rest?"
```
Status is decided by the validator, never by the model: the schema has no
confidence field, and the model cites short handles (E1, E2) that the
validator maps back to real chunk ids, so it cannot invent a citation.

## Searching
```bash
python scripts/search.py --tenant $TENANT "Do you enforce MFA?"
python scripts/search.py --tenant $TENANT "Where is customer data stored?" --explain
```
`--explain` shows per-retriever scores and ranks, which is how to tell a
lexical hit from a dense one when tuning.

## Ingesting documents
```bash
TENANT=$(python scripts/create_tenant.py "Acme Corp")
python scripts/ingest.py --tenant $TENANT --doc-type policy tests/data/policy.pdf
python scripts/ingest.py --tenant $TENANT --list
python scripts/ingest.py --tenant $TENANT --show <document-id>
```
Runs offline with `SQC_EMBEDDING_PROVIDER=fake`. Set it to `voyage` with a
real key when measuring retrieval quality.

## Providers
Set in `.env`. Defaults are `fake`, so `pytest` never needs a key or a network.
For real runs set `SQC_EMBEDDING_PROVIDER=voyage`, `SQC_RERANK_PROVIDER=voyage`,
`SQC_LLM_PROVIDER=anthropic` and supply `VOYAGE_API_KEY` and `ANTHROPIC_API_KEY`.
`SQC_EMBEDDING_DIM` must match the dimension the schema was migrated with;
changing it requires re-running `scripts/init_db.py` and re-embedding.

### Known gap
Parser tests run against generated PDFs and DOCX files, which have no headers,
footers, watermarks, two-column layouts or scanned pages. Drop any real security
policy PDF into `tests/data/` and `test_real_policy_pdf_ingests_with_usable_structure`
activates automatically, asserting that heading detection, page numbers and chunk
sizes survive a real layout.

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
