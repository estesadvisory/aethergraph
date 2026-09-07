from pathlib import Path

import pytest

from aethergraph.catalog import load_models, load_policy
from aethergraph.cost import CostLedger
from aethergraph.graph.engine import GraphEngine
from aethergraph.providers.registry import ProviderRegistry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def models():
    return load_models(ROOT / "configs" / "models.yaml")


@pytest.fixture
def policy():
    return load_policy(ROOT / "configs" / "policies.yaml")


@pytest.fixture
def engine(models, policy):
    return GraphEngine(models, policy, ProviderRegistry(), CostLedger())
