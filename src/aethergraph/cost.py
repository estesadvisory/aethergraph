from __future__ import annotations

from threading import Lock

from pydantic import BaseModel

from aethergraph.types import ModelProfile


class LedgerEntry(BaseModel):
    graph_id: str
    node_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    task_type: str = ""


class CostLedger:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._lock = Lock()

    def record(self, entry: LedgerEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def entries(self) -> list[LedgerEntry]:
        with self._lock:
            return list(self._entries)

    def total_usd(self, graph_id: str | None = None) -> float:
        with self._lock:
            return sum(
                item.cost_usd
                for item in self._entries
                if graph_id is None or item.graph_id == graph_id
            )

    def summary(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "calls": len(self._entries),
                "total_usd": round(sum(item.cost_usd for item in self._entries), 6),
                "input_tokens": sum(item.input_tokens for item in self._entries),
                "output_tokens": sum(item.output_tokens for item in self._entries),
            }


def estimate_cost(
    profile: ModelProfile,
    input_tokens: int,
    output_tokens: int,
) -> float:
    return (
        input_tokens * profile.input_cost_per_mtok
        + output_tokens * profile.output_cost_per_mtok
    ) / 1_000_000
