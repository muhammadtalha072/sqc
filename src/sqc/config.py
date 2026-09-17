"""Centralised configuration. Nothing else in the codebase reads os.environ."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SQC_", extra="ignore", case_sensitive=False
    )

    # database
    database_url: str = "postgresql+psycopg://sqc_app:changeme@localhost:5432/sqc"
    admin_database_url: str = "postgresql+psycopg://sqc:sqc@localhost:5432/sqc"

    # Provider credentials. These carry explicit aliases because they are
    # conventionally unprefixed: a user who writes GEMINI_API_KEY in .env
    # expects it to be read, and env_prefix would otherwise make the setting
    # SQC_GEMINI_API_KEY and silently ignore the line they actually wrote.
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    voyage_api_key: str = Field(default="", validation_alias="VOYAGE_API_KEY")

    # embeddings
    embedding_provider: str = "fake"
    embedding_model: str = "voyage-3"
    embedding_dim: int = 1024

    # Reranking defaults to none, not fake. Measured on a real policy, the
    # fake reranker demoted the chunk that answered the question from first
    # place to eighth, and the system refused a question it had the evidence
    # for. A stand-in that scores worse than no reranker makes the product
    # quietly worse than if the feature did not exist.
    rerank_provider: str = "none"
    rerank_model: str = "rerank-2"

    # Entailment checking: none | lexical | llm. Defaults to lexical
    # because it costs nothing and catches the failure that matters most,
    # a claim naming a figure or standard the evidence never mentions.
    entailment_provider: str = "lexical"

    # Provider retry budget. Kept small on purpose. Raising it to 8 with a
    # two-second base delay cost two minutes of sleeping per failing call
    # and recovered nothing across four evaluation runs, while consuming
    # eight requests per failure against a 250-request daily free tier.
    # Retries help with transient throttling; an exhausted quota now
    # raises ProviderQuotaError on the first response instead.
    llm_max_attempts: int = Field(default=4, ge=1, le=12)
    llm_base_delay: float = Field(default=0.5, ge=0.0, le=30.0)

    # answering
    llm_provider: str = "fake"
    llm_model: str = "claude-sonnet-5"

    # retrieval
    candidates_per_retriever: int = Field(default=40, ge=1, le=200)
    evidence_top_k: int = Field(default=6, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1)

    # Refusal thresholds. 0.0 disables the retrieval floor, which is the
    # deliberate default: rerank scores are not comparable across providers,
    # so a threshold tuned for one model silently refuses answerable
    # questions under another. Calibrate against the eval set, then set it.
    retrieval_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_high: float = Field(default=0.75, ge=0.0, le=1.0)
    confidence_medium: float = Field(default=0.50, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
