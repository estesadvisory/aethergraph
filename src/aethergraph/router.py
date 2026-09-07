from __future__ import annotations

from collections.abc import Callable

from aethergraph.classifier import classify, estimate_tokens
from aethergraph.cost import estimate_cost
from aethergraph.types import (
    Locality,
    ModelProfile,
    RouteCandidate,
    RouteDecision,
    RouteRequest,
    RoutingPolicy,
    TaskClassification,
    TaskType,
)


AvailabilityFn = Callable[[ModelProfile], bool]


def required_capability(
    policy: RoutingPolicy,
    task_type: TaskType,
    complexity: int,
    override: float | None = None,
) -> float:
    if override is not None:
        return override
    floors = policy.floors.get(task_type) or [5, 6, 7, 8, 9]
    index = max(0, min(len(floors) - 1, complexity - 1))
    return float(floors[index])


def _supports_modalities(profile: ModelProfile, request: RouteRequest) -> bool:
    supported = set(profile.modalities)
    return all(item in supported for item in request.modalities)


def score_candidate(
    profile: ModelProfile,
    request: RouteRequest,
    policy: RoutingPolicy,
    classification: TaskClassification,
    input_tokens: int,
    output_tokens: int,
) -> RouteCandidate:
    capability = float(profile.capabilities.get(classification.task_type, 0))
    estimated = estimate_cost(profile, input_tokens, output_tokens)
    latency_penalty = (profile.latency_ms_p50 / 1000) * policy.latency_weight_usd_per_second
    local_bonus = 0.0
    prefer_local = policy.prefer_local if request.prefer_local is None else request.prefer_local
    reasons = [
        f"capability {capability:.1f}",
        f"est_cost ${estimated:.6f}",
        f"latency {profile.latency_ms_p50}ms",
    ]
    if prefer_local and profile.locality is Locality.LOCAL:
        local_bonus = policy.local_bonus_usd
        reasons.append("local bonus")
    score = estimated + latency_penalty - local_bonus
    return RouteCandidate(
        model_id=profile.id,
        score_usd=round(score, 8),
        estimated_cost_usd=round(estimated, 8),
        latency_ms=profile.latency_ms_p50,
        capability=capability,
        locality=profile.locality,
        reasons=reasons,
    )


def route(
    request: RouteRequest,
    models: list[ModelProfile],
    policy: RoutingPolicy,
    available: AvailabilityFn | None = None,
) -> RouteDecision:
    inferred = classify(request.prompt, request.modalities)
    classification = inferred
    if request.task_type:
        classification = inferred.model_copy(
            update={
                "task_type": request.task_type,
                "rationale": f"caller override type={request.task_type.value}; {inferred.rationale}",
            }
        )
    if request.complexity:
        classification.complexity = request.complexity

    floor = required_capability(
        policy, classification.task_type, classification.complexity, request.required_capability
    )
    input_tokens = request.estimated_input_tokens or estimate_tokens(request.prompt)
    output_tokens = request.max_output_tokens or policy.default_output_tokens.get(
        classification.task_type, 400
    )
    budget = request.budget_usd if request.budget_usd is not None else policy.default_budget_usd

    rejected: list[str] = []
    candidates: list[RouteCandidate] = []
    for profile in models:
        if available and not available(profile):
            rejected.append(f"{profile.id}: unavailable")
            continue
        if profile.locality is Locality.CLOUD and not policy.allow_cloud:
            rejected.append(f"{profile.id}: cloud disabled")
            continue
        if profile.locality is Locality.LOCAL and not policy.allow_local:
            rejected.append(f"{profile.id}: local disabled")
            continue
        if not _supports_modalities(profile, request):
            rejected.append(f"{profile.id}: missing modality")
            continue
        if input_tokens > profile.context_window:
            rejected.append(f"{profile.id}: context window")
            continue
        capability = float(profile.capabilities.get(classification.task_type, 0))
        if capability < floor:
            rejected.append(f"{profile.id}: capability {capability:.1f} < {floor:.1f}")
            continue
        candidate = score_candidate(
            profile, request, policy, classification, input_tokens, output_tokens
        )
        if candidate.estimated_cost_usd > budget:
            rejected.append(f"{profile.id}: over budget ${candidate.estimated_cost_usd:.6f}")
            continue
        candidates.append(candidate)

    if not candidates:
        raise RoutingError(
            f"No model meets {classification.task_type.value} "
            f"complexity {classification.complexity} at floor {floor}."
        )

    ranked = sorted(candidates, key=lambda item: (item.score_usd, -item.capability, item.latency_ms))
    return RouteDecision(
        selected=ranked[0],
        classification=classification,
        required_capability=floor,
        alternates=ranked[1 : policy.shadow_log_alternates + 1],
        rejected=rejected,
    )


class RoutingError(RuntimeError):
    pass
