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

    # embeddings
    embedding_provider: str = "fake"
    embedding_model: str = "voyage-3"
    embedding_dim: int = 1024

    # reranking
    rerank_provider: str = "fake"
    rerank_model: str = "rerank-2"

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
