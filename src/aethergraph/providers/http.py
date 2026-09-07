from __future__ import annotations

import os
from typing import Any

import httpx

from aethergraph.providers.base import ModelProvider
from aethergraph.types import CompletionRequest, CompletionResult, ModelProfile


def _headers(api_key: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


class OpenAICompatProvider(ModelProvider):
    name = "openai_compat"

    def __init__(self, default_base_url: str = "https://api.openai.com/v1") -> None:
        self.default_base_url = default_base_url

    def available(self, profile: ModelProfile) -> bool:
        env_name = profile.api_key_env or "OPENAI_API_KEY"
        return bool(os.environ.get(env_name))

    async def complete(self, profile: ModelProfile, request: CompletionRequest) -> CompletionResult:
        env_name = profile.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(env_name)
        base_url = (profile.base_url or self.default_base_url).rstrip("/")
        payload: dict[str, Any] = {
            "model": profile.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_output_tokens,
        }
        if request.system:
            payload["messages"].insert(0, {"role": "system", "content": request.system})
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=_headers(api_key),
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
            model=profile.model,
            provider=self.name,
        )


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"

    def __init__(self) -> None:
        super().__init__("https://api.openai.com/v1")

    def available(self, profile: ModelProfile) -> bool:
        return bool(os.environ.get(profile.api_key_env or "OPENAI_API_KEY"))


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def available(self, profile: ModelProfile) -> bool:
        return bool(os.environ.get(profile.api_key_env or "ANTHROPIC_API_KEY"))

    async def complete(self, profile: ModelProfile, request: CompletionRequest) -> CompletionResult:
        api_key = os.environ.get(profile.api_key_env or "ANTHROPIC_API_KEY")
        payload: dict[str, Any] = {
            "model": profile.model,
            "max_tokens": request.max_output_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            payload["system"] = request.system
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "content-type": "application/json",
                    "x-api-key": api_key or "",
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        )
        usage = body.get("usage") or {}
        return CompletionResult(
            text=text,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            model=profile.model,
            provider=self.name,
        )


class GoogleProvider(ModelProvider):
    name = "google"

    def available(self, profile: ModelProfile) -> bool:
        return bool(os.environ.get(profile.api_key_env or "GOOGLE_API_KEY"))

    async def complete(self, profile: ModelProfile, request: CompletionRequest) -> CompletionResult:
        api_key = os.environ.get(profile.api_key_env or "GOOGLE_API_KEY")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{profile.model}:generateContent"
        )
        payload = {
            "contents": [{"parts": [{"text": request.prompt}]}],
            "generationConfig": {"maxOutputTokens": request.max_output_tokens},
        }
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(url, params={"key": api_key}, json=payload)
            response.raise_for_status()
            body = response.json()
        candidates = body.get("candidates") or []
        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or []) if candidates else []
        text = "".join(part.get("text", "") for part in parts)
        usage = body.get("usageMetadata") or {}
        return CompletionResult(
            text=text,
            input_tokens=int(usage.get("promptTokenCount") or 0),
            output_tokens=int(usage.get("candidatesTokenCount") or 0),
            model=profile.model,
            provider=self.name,
        )


class OllamaProvider(ModelProvider):
    name = "ollama"

    def __init__(self, host: str = "http://127.0.0.1:11434") -> None:
        self.host = host.rstrip("/")
        self._models: set[str] | None = None

    def available(self, profile: ModelProfile) -> bool:
        try:
            if self._models is None:
                response = httpx.get(f"{self.host}/api/tags", timeout=1.5)
                response.raise_for_status()
                self._models = {
                    item.get("name", "")
                    for item in (response.json().get("models") or [])
                }
        except Exception:
            self._models = set()
        return profile.model in (self._models or set()) or bool(self._models)

    async def complete(self, profile: ModelProfile, request: CompletionRequest) -> CompletionResult:
        payload = {
            "model": profile.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
            "options": {"num_predict": request.max_output_tokens},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        text = ((body.get("message") or {}).get("content")) or ""
        return CompletionResult(
            text=text,
            input_tokens=int((body.get("prompt_eval_count") or 0)),
            output_tokens=int((body.get("eval_count") or 0)),
            model=profile.model,
            provider=self.name,
        )
