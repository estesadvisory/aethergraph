from __future__ import annotations

import os

from aethergraph.a2a.cards import build_gateway_card
from aethergraph.a2a.runtime import GatewayHandler
from aethergraph.a2a.server import create_a2a_app
from aethergraph.catalog import load_models, load_policy
from aethergraph.config import Settings, load_settings
from aethergraph.cost import CostLedger
from aethergraph.graph.engine import GraphEngine
from aethergraph.providers.registry import ProviderRegistry


class Runtime:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.models = load_models(self.settings.models_path)
        self.policy = load_policy(self.settings.policies_path)
        self.providers = ProviderRegistry(
            ollama_host=os.environ.get("OLLAMA_HOST", self.settings.ollama_host)
        )
        self.ledger = CostLedger()
        self.engine = GraphEngine(self.models, self.policy, self.providers, self.ledger)
        self.card = build_gateway_card(self.settings.public_url)
        self.handler = GatewayHandler(self.engine)

    def app(self):
        application = create_a2a_app(self.card, self.handler)
        attach_ops_routes(application, self)
        return application


def attach_ops_routes(app, runtime: Runtime) -> None:
    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "models": [
                {
                    "id": item.id,
                    "provider": item.provider,
                    "locality": item.locality.value,
                    "available": runtime.providers.available(item),
                    "input_cost_per_mtok": item.input_cost_per_mtok,
                    "capabilities": {key.value: value for key, value in item.capabilities.items()},
                }
                for item in runtime.models
            ]
        }

    @app.get("/v1/ledger")
    async def ledger() -> dict:
        return {
            "summary": runtime.ledger.summary(),
            "entries": [item.model_dump(mode="json") for item in runtime.ledger.entries()],
        }

    @app.get("/v1/policy")
    async def policy() -> dict:
        return runtime.policy.model_dump(mode="json")


def create_app() -> object:
    return Runtime().app()
