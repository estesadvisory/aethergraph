from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_config_dir() -> Path:
    override = os.environ.get("AETHERGRAPH_CONFIG_DIR")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs"
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "configs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AETHERGRAPH_", extra="ignore")

    config_dir: Path = Field(default_factory=default_config_dir)
    host: str = "0.0.0.0"
    port: int = 8080
    public_url: str = "http://127.0.0.1:8080"
    mode: str = "inprocess"
    policy: str = "balanced"
    log_level: str = "INFO"
    ollama_host: str = "http://127.0.0.1:11434"
    fabric: str = "auto"
    litellm_base_url: str | None = None
    litellm_api_key: str | None = None
    litellm_include_local: bool = False

    @property
    def models_path(self) -> Path:
        return self.config_dir / "models.yaml"

    @property
    def policies_path(self) -> Path:
        return self.config_dir / "policies.yaml"

    @property
    def agents_path(self) -> Path:
        return self.config_dir / "agents.yaml"


def load_settings() -> Settings:
    return Settings()
