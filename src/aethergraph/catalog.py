from __future__ import annotations

from pathlib import Path

import yaml

from aethergraph.types import Locality, Modality, ModelProfile, RoutingPolicy, TaskType


def _task_map(raw: dict[str, float]) -> dict[TaskType, float]:
    return {TaskType(key): float(value) for key, value in raw.items()}


def load_models(path: Path) -> list[ModelProfile]:
    payload = yaml.safe_load(path.read_text()) or {}
    models: list[ModelProfile] = []
    for item in payload.get("models", []):
        models.append(
            ModelProfile(
                id=item["id"],
                display_name=item["display_name"],
                provider=item["provider"],
                model=item["model"],
                locality=Locality(item["locality"]),
                context_window=int(item["context_window"]),
                input_cost_per_mtok=float(item["input_cost_per_mtok"]),
                output_cost_per_mtok=float(item["output_cost_per_mtok"]),
                latency_ms_p50=int(item["latency_ms_p50"]),
                modalities=[Modality(m) for m in item.get("modalities", ["text"])],
                capabilities=_task_map(item.get("capabilities", {})),
                base_url=item.get("base_url"),
                api_key_env=item.get("api_key_env"),
                litellm_model=item.get("litellm_model"),
            )
        )
    return models


def load_policy(path: Path) -> RoutingPolicy:
    payload = yaml.safe_load(path.read_text()) or {}
    floors = {
        TaskType(key): [int(v) for v in values]
        for key, values in (payload.get("floors") or {}).items()
    }
    default_output_tokens = {
        TaskType(key): int(value)
        for key, value in (payload.get("default_output_tokens") or {}).items()
    }
    return RoutingPolicy(
        name=payload.get("name", "balanced"),
        prefer_local=bool(payload.get("prefer_local", True)),
        local_bonus_usd=float(payload.get("local_bonus_usd", 0.0008)),
        latency_weight_usd_per_second=float(payload.get("latency_weight_usd_per_second", 0.0004)),
        max_escalations=int(payload.get("max_escalations", 1)),
        default_budget_usd=float(payload.get("default_budget_usd", 0.25)),
        allow_cloud=bool(payload.get("allow_cloud", True)),
        allow_local=bool(payload.get("allow_local", True)),
        shadow_log_alternates=int(payload.get("shadow_log_alternates", 3)),
        floors=floors,
        default_output_tokens=default_output_tokens,
        quality_gates=dict(payload.get("quality_gates") or {}),
    )
