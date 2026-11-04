from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables (or a .env file). Nothing is required."""

    model_config = SettingsConfigDict(env_prefix="LITSCOUT_", env_file=".env", extra="ignore")

    db_path: Path = Path("data/litscout.db")
    # When set, the web UI requires this password. Leave empty for single-user local use.
    access_password: str = ""
    # Secret used to sign the session cookie. Generated per process if empty.
    secret_key: str = ""
    # "demo" = bundled offline sample corpus only (useful without internet)
    sources_enabled: str = "openalex,arxiv,crossref,demo"
    contact_email: str = ""  # sent to OpenAlex/Crossref "polite pool"; optional
    http_timeout: float = 20.0

    # Optional LLM for abstractive synthesis. Without a key, LitScout uses its offline extractive drafter.
    llm_provider: Literal["none", "anthropic", "openai"] = "none"
    llm_api_key: str = ""
    llm_model: str = ""
    # For OpenAI-compatible servers (OpenAI, Ollama, LM Studio, vLLM...)
    llm_base_url: str = "https://api.openai.com/v1"

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "none" and (bool(self.llm_api_key) or self.llm_provider == "openai")

    @property
    def sources(self) -> list[str]:
        return [s.strip() for s in self.sources_enabled.split(",") if s.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
