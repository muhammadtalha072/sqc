# Evaluation

`datasets/` holds version-controlled golden files. `fixtures/` holds the small
policies the edge-case dataset is written against - committed on purpose, since
evaluation needs a corpus that cannot drift and a customer PDF cannot be
committed.

`cassettes/` holds recorded model responses, keyed by a hash of the exact
prompt. Replay makes reruns free and deterministic, which separates two
questions a live eval conflates: did my code change, or did the model change?

## The rule this suite enforces

Coverage is reported, never optimised for. A change that answers more
questions while answering more of them wrongly is a regression. The gate is
asymmetric: any rise in false answers or hallucinations fails the build, while
a fall in coverage is reported and tolerated.

## Corpus isolation

Each dataset declares its own `documents:` and gets a tenant derived from its
name. `scripts/eval_setup.py` provisions exactly that corpus and removes
anything else already in the tenant; the runner refuses to score a tenant
whose contents do not match.

This exists because an earlier `--with-real` flag ingested every document on
disk into one shared tenant. Twelve cases written against four small fixtures
were scored while competing with 164 chunks of five unrelated university
policies, and the retrieval recall from that run described the setup script.
Isolated, the same dataset scores 100% retrieval recall rather than 87.5%.

## Replay

The cache key includes the model name. Replay must be given that name
explicitly, because it builds no provider to read it from; without it every
lookup keyed to "cassette" and missed everything just recorded.

    record  -> 12 recorded, 0 replayed
    replay  -> 12 replayed, 0 recorded, 0 provider calls

## Provider failures

Rates are computed only over cases whose model call completed. A run with 17
of 24 calls failing reported coverage of 16.7% and a false-refusal rate of
83.3%; both described a free tier returning 503 under load, not the model.
The report states how many calls failed and over how many cases the rates
were computed.

Re-running in record mode retries only what is still missing, because cached
responses are skipped. `--attempts` sets the retry budget per call, defaulting
to 8 for evaluation against the interactive default of 4.
