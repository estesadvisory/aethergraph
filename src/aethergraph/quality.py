from __future__ import annotations

import json

from aethergraph.types import RoutingPolicy, TaskType


def passes_quality_gate(text: str, task_type: TaskType, policy: RoutingPolicy) -> tuple[bool, str]:
    gates = policy.quality_gates
    if gates.get("require_nonempty", True) and not text.strip():
        return False, "empty output"
    if task_type is TaskType.EXTRACTION and gates.get("json_for_extraction", True):
        snippet = _extract_json(text)
        if snippet is None:
            return False, "extraction did not return JSON"
    return True, "ok"


def _extract_json(text: str) -> object | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.split("\n", 1)[-1]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
