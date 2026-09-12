# Real-world test documents

Drop real security policy PDFs here. `tests/test_ingestion_pipeline.py`
picks up every `*.pdf` in this folder automatically and asserts that
heading detection, page numbers and chunk sizes survive a real layout.

The test is skipped while this folder is empty, so the suite stays green
on a fresh clone. Do not commit customer documents; use publicly published
policies only.
