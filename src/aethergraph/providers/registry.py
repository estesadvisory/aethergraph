from __future__ import annotations

from aethergraph.providers.base import ModelProvider
from aethergraph.providers.http import (
    AnthropicProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAICompatProvider,
    OpenAIProvider,
)
from aethergraph.providers.mock import MockProvider
from aethergraph.types import CompletionRequest, CompletionResult, ModelProfile


class ProviderRegistry:
    def __init__(self, ollama_host: str = "http://127.0.0.1:11434") -> None:
        providers: list[ModelProvider] = [
            MockProvider(),
            OpenAIProvider(),
            OpenAICompatProvider(),
            AnthropicProvider(),
            GoogleProvider(),
            OllamaProvider(host=ollama_host),
        ]
        self._providers = {provider.name: provider for provider in providers}

    def get(self, name: str) -> ModelProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider '{name}'") from exc

    def available(self, profile: ModelProfile) -> bool:
        provider = self._providers.get(profile.provider)
        return bool(provider and provider.available(profile))

    async def complete(self, profile: ModelProfile, request: CompletionRequest) -> CompletionResult:
        return await self.get(profile.provider).complete(profile, request)
