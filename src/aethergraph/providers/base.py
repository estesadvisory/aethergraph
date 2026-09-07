from __future__ import annotations

from abc import ABC, abstractmethod

from aethergraph.types import CompletionRequest, CompletionResult, ModelProfile


class ModelProvider(ABC):
    name: str

    @abstractmethod
    def available(self, profile: ModelProfile) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def complete(self, profile: ModelProfile, request: CompletionRequest) -> CompletionResult:
        raise NotImplementedError
