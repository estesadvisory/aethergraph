from aethergraph.cost import estimate_cost
from aethergraph.types import Locality, Modality, ModelProfile, TaskType


def test_estimate_cost_matches_per_million():
    profile = ModelProfile(
        id="x",
        display_name="x",
        provider="mock",
        model="x",
        locality=Locality.CLOUD,
        context_window=1000,
        input_cost_per_mtok=1.0,
        output_cost_per_mtok=2.0,
        latency_ms_p50=10,
        modalities=[Modality.TEXT],
        capabilities={TaskType.CONVERSATION: 5},
    )
    assert estimate_cost(profile, 1_000_000, 500_000) == 2.0
