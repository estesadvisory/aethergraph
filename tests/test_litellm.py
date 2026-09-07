import httpx

from aethergraph.providers.litellm import LiteLLMProvider, litellm_model_name
from aethergraph.providers.registry import ProviderRegistry
from aethergraph.types import CompletionRequest, Locality, Modality, ModelProfile, TaskType


def _profile(**overrides) -> ModelProfile:
    base = dict(
        id="openai-gpt-4.1-mini",
        display_name="GPT-4.1 Mini",
        provider="openai",
        model="gpt-4.1-mini",
        locality=Locality.CLOUD,
        context_window=1000,
        input_cost_per_mtok=0.4,
        output_cost_per_mtok=1.6,
        latency_ms_p50=700,
        modalities=[Modality.TEXT],
        capabilities={TaskType.CLASSIFICATION: 8},
    )
    base.update(overrides)
    return ModelProfile(**base)


def test_litellm_model_prefixes():
    assert litellm_model_name(_profile()) == "gpt-4.1-mini"
    assert (
        litellm_model_name(_profile(provider="anthropic", model="claude-sonnet-4-5"))
        == "anthropic/claude-sonnet-4-5"
    )
    assert (
        litellm_model_name(_profile(provider="google", model="gemini-2.5-flash"))
        == "gemini/gemini-2.5-flash"
    )
    groq = _profile(
        provider="openai_compat",
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
    )
    assert litellm_model_name(groq) == "groq/llama-3.3-70b-versatile"
    assert litellm_model_name(_profile(litellm_model="azure/gpt-4.1")) == "azure/gpt-4.1"


def test_fabric_makes_cloud_models_available_without_vendor_keys():
    registry = ProviderRegistry(fabric="litellm", litellm_base_url="http://127.0.0.1:4000")
    assert registry.available(_profile())
    assert registry.uses_litellm(_profile())
    assert not registry.uses_litellm(_profile(provider="mock", model="mock-nano", locality=Locality.LOCAL))


def test_auto_fabric_enables_when_proxy_url_is_set():
    registry = ProviderRegistry(fabric="auto", litellm_base_url="http://127.0.0.1:4000")
    assert registry.fabric == "litellm"
    assert registry.uses_litellm(_profile())


def test_direct_fabric_ignores_litellm_url():
    registry = ProviderRegistry(fabric="direct", litellm_base_url="http://127.0.0.1:4000")
    assert not registry.uses_litellm(_profile())
    assert not registry.available(_profile())


async def test_complete_posts_openai_compat_payload(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "routed"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2},
            }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    provider = LiteLLMProvider(base_url="http://proxy:4000", api_key="sk-test")
    result = await provider.complete(
        _profile(provider="anthropic", model="claude-haiku-4-5"),
        CompletionRequest(model_id="x", prompt="hello", max_output_tokens=32),
    )
    assert result.text == "routed"
    assert result.provider == "litellm"
    assert captured["url"] == "http://proxy:4000/chat/completions"
    assert captured["json"]["model"] == "anthropic/claude-haiku-4-5"
    assert captured["headers"]["authorization"] == "Bearer sk-test"
