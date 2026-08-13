"""Loads environment and versioned repository configuration without hidden defaults."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime values that must remain outside committed configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    weaviate_url: str = "http://localhost:8080"
    weaviate_grpc_url: str = "localhost:50051"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "bge-m3"
    omniroute_base_url: str = "http://localhost:20128/v1"
    omniroute_api_key: str = ""
    direct_llm_base_url: str = ""
    direct_llm_api_key: str = ""
    rag_model: str = "auto"
    llm_timeout_seconds: int = 180
    rag_generation_provider: str = "local"
    gemini_generation_model: str = "gemini-3.1-flash-lite"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    openrouter_model: str = ""
    openrouter_supports_structured_outputs: bool = True
    cohere_api_key: str = ""
    google_api_key: str = ""
    # Gemini is accessed through its OpenAI-compatible endpoint because the native SDK
    # closes its HTTP client under this macOS runtime. The evaluated model is still Gemini.
    ragas_judge_provider: str = "gemini_openai_compatible"
    ragas_judge_model: str = "gemini-3.1-flash-lite"
    # The runtime can only retrieve from this reviewed, checksum-pinned corpus release.
    corpus_release_id: str = "governance-security-expanded-2026-07-30-r2"


@lru_cache
def settings() -> Settings:
    """Returns one validated settings instance per process."""

    return Settings()


@lru_cache
def rag_config() -> dict:
    """Reads the reviewed retrieval and quality settings from version control."""

    path = Path("config/rag.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache
def prompt_template(prompt_id: str) -> str:
    """Loads a reviewed prompt by ID instead of embedding mutable instructions in code."""

    prompts = yaml.safe_load(Path("config/prompts.yaml").read_text(encoding="utf-8"))["prompts"]
    if prompt_id not in prompts:
        raise KeyError(f"Unknown versioned prompt: {prompt_id}")
    return prompts[prompt_id]["template"]
