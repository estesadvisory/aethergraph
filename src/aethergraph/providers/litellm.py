from __future__ import annotations

import os
from typing import Any

import httpx

from aethergraph.providers.base import ModelProvider
from aethergraph.types import CompletionRequest, CompletionResult, Locality, ModelProfile

_PREFIX = {
    "openai": "",
    "anthropic": "anthropic/",
    "google": "gemini/",
    "openai_compat": "",
    "ollama": "ollama/",
    "litellm": "",
}

_GROQ_HOST = "api.groq.com"


def litellm_model_name(profile: ModelProfile) -> str:
    if profile.litellm_model:
        return profile.litellm_model
    if profile.provider == "openai_compat" and profile.base_url and _GROQ_HOST in profile.base_url:
        return f"groq/{profile.model}"
    prefix = _PREFIX.get(profile.provider, "")
    if prefix and profile.model.startswith(prefix):
        return profile.model
    return f"{prefix}{profile.model}"


class LiteLLMProvider(ModelProvider):
    """OpenAI-compatible client for a LiteLLM proxy.

    LiteLLM owns vendor SDKs, keys, and retries. AetherGraph keeps
    classification, graph binding, and cost policy.
    """

    name = "litellm"
    fabric_providers = frozenset(
        {"openai", "anthropic", "google", "openai_compat", "litellm"}
    )

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        include_local: bool = False,
    ) -> None:
        self.base_url = (base_url or os.environ.get("LITELLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY") or "sk-litellm"
        self.include_local = include_local

    def configured(self) -> bool:
        return bool(self.base_url)

    def covers(self, profile: ModelProfile) -> bool:
        if profile.provider == "mock":
            return False
        if profile.provider in self.fabric_providers:
            return True
        if profile.provider == "ollama":
            return self.include_local or profile.locality is not Locality.LOCAL
        return False

    def available(self, profile: ModelProfile) -> bool:
        return self.configured() and self.covers(profile)

    async def complete(self, profile: ModelProfile, request: CompletionRequest) -> CompletionResult:
        if not self.base_url:
            raise RuntimeError("LITELLM_BASE_URL is not set")
        payload: dict[str, Any] = {
            "model": litellm_model_name(profile),
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_output_tokens,
        }
        if request.system:
            payload["messages"].insert(0, {"role": "system", "content": request.system})
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "content-type": "application/json",
                    "authorization": f"Bearer {self.api_key}",
                },
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        choice = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        return CompletionResult(
            text=choice or "",
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            model=litellm_model_name(profile),
            provider=self.name,
        )
