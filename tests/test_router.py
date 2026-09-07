from aethergraph.router import RoutingError, route
from aethergraph.types import Locality, Modality, RouteRequest, TaskType


def test_cheap_classification_prefers_inexpensive_or_local(models, policy):
    decision = route(
        RouteRequest(
            prompt="Classify this ticket as billing, bug, or feature.",
            task_type=TaskType.CLASSIFICATION,
            complexity=1,
        ),
        models,
        policy,
        available=lambda profile: profile.provider == "mock",
    )
    assert decision.selected.model_id == "mock-nano"
    assert decision.selected.locality is Locality.LOCAL


def test_hard_reasoning_uses_stronger_model(models, policy):
    decision = route(
        RouteRequest(
            prompt="Prove the cost/latency Pareto frontier for this catalog.",
            task_type=TaskType.REASONING,
            complexity=5,
        ),
        models,
        policy,
        available=lambda profile: profile.provider == "mock",
    )
    assert decision.selected.model_id == "mock-reasoner"
    assert decision.required_capability >= 9


def test_missing_vision_is_rejected(models, policy):
    try:
        route(
            RouteRequest(
                prompt="Describe the image",
                task_type=TaskType.VISION,
                complexity=3,
                modalities=[Modality.IMAGE],
            ),
            models,
            policy,
            available=lambda profile: profile.provider == "mock",
        )
    except RoutingError as exc:
        assert "No model meets" in str(exc)
    else:
        raise AssertionError("expected RoutingError")


def test_cloud_can_be_disabled(models, policy):
    policy.allow_cloud = False
    decision = route(
        RouteRequest(prompt="Summarize this paragraph about routing.", task_type=TaskType.SUMMARIZATION, complexity=2),
        models,
        policy,
        available=lambda profile: profile.provider == "mock",
    )
    assert decision.selected.locality is Locality.LOCAL
