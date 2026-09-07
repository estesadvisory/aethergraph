from __future__ import annotations

from aethergraph.providers.base import ModelProvider
from aethergraph.providers.http import (
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAICompatProvider,
    OpenAIProvider,
)
from aethergraph.providers.litellm import LiteLLMProvider
from aethergraph.providers.mock import MockProvider
from aethergraph.types import CompletionRequest, CompletionResult, ModelProfile


class ProviderRegistry:
    def __init__(
        self,
        ollama_host: str = "http://127.0.0.1:11434",
        fabric: str = "direct",
        litellm_base_url: str | None = None,
        litellm_api_key: str | None = None,
        litellm_include_local: bool = False,
    ) -> None:
        self.fabric = fabric
        self.litellm = LiteLLMProvider(
            base_url=litellm_base_url,
            api_key=litellm_api_key,
            include_local=litellm_include_local,
        )
        if self.fabric == "auto" and self.litellm.configured():
            self.fabric = "litellm"
        providers: list[ModelProvider] = [
            MockProvider(),
            OpenAIProvider(),
            OpenAICompatProvider(),
            AnthropicProvider(),
            GoogleProvider(),
            OllamaProvider(host=ollama_host),
            self.litellm,
        ]
        self._providers = {provider.name: provider for provider in providers}

    def get(self, name: str) -> ModelProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider '{name}'") from exc

    def uses_litellm(self, profile: ModelProfile) -> bool:
        return self.fabric == "litellm" and self.litellm.covers(profile) and self.litellm.configured()

    def available(self, profile: ModelProfile) -> bool:
        if self.uses_litellm(profile):
            return True
        provider = self._providers.get(profile.provider)
        return bool(provider and provider.available(profile))

    async def complete(self, profile: ModelProfile, request: CompletionRequest) -> CompletionResult:
        if self.uses_litellm(profile):
            return await self.litellm.complete(profile, request)
        return await self.get(profile.provider).complete(profile, request)
