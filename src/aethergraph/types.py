from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    CONVERSATION = "conversation"
    WRITING = "writing"
    TRANSLATION = "translation"
    PLANNING = "planning"
    CODE = "code"
    REASONING = "reasoning"
    VISION = "vision"
    EMBEDDING = "embedding"


class Locality(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    ESCALATED = "escalated"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class TaskClassification(BaseModel):
    task_type: TaskType
    complexity: int = Field(ge=1, le=5)
    modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    scores: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class ModelProfile(BaseModel):
    id: str
    display_name: str
    provider: str
    model: str
    locality: Locality
    context_window: int
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    latency_ms_p50: int
    modalities: list[Modality]
    capabilities: dict[TaskType, float]
    base_url: str | None = None
    api_key_env: str | None = None


class RoutingPolicy(BaseModel):
    name: str = "balanced"
    prefer_local: bool = True
    local_bonus_usd: float = 0.0008
    latency_weight_usd_per_second: float = 0.0004
    max_escalations: int = 1
    default_budget_usd: float = 0.25
    allow_cloud: bool = True
    allow_local: bool = True
    shadow_log_alternates: int = 3
    floors: dict[TaskType, list[int]] = Field(default_factory=dict)
    default_output_tokens: dict[TaskType, int] = Field(default_factory=dict)
    quality_gates: dict[str, bool] = Field(default_factory=dict)


class RouteRequest(BaseModel):
    prompt: str
    task_type: TaskType | None = None
    complexity: int | None = Field(default=None, ge=1, le=5)
    modalities: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    estimated_input_tokens: int | None = None
    max_output_tokens: int | None = None
    required_capability: float | None = None
    budget_usd: float | None = None
    prefer_local: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteCandidate(BaseModel):
    model_id: str
    score_usd: float
    estimated_cost_usd: float
    latency_ms: int
    capability: float
    locality: Locality
    reasons: list[str] = Field(default_factory=list)


class RouteDecision(BaseModel):
    selected: RouteCandidate
    classification: TaskClassification
    required_capability: float
    alternates: list[RouteCandidate] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    task_type: TaskType | None = None
    prompt: str
    depends_on: list[str] = Field(default_factory=list)
    max_output_tokens: int | None = None
    required_capability: float | None = None
    model_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphSpec(BaseModel):
    id: str
    nodes: list[GraphNode]
    input: str = ""
    budget_usd: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeResult(BaseModel):
    node_id: str
    status: NodeStatus
    model_id: str | None = None
    output: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    escalated_from: str | None = None
    error: str | None = None
    route: RouteDecision | None = None


class GraphResult(BaseModel):
    graph_id: str
    status: NodeStatus
    nodes: dict[str, NodeResult]
    total_cost_usd: float = 0.0
    output: str = ""


class CompletionRequest(BaseModel):
    model_id: str
    prompt: str
    system: str | None = None
    max_output_tokens: int = 800
    images: list[str] = Field(default_factory=list)


class CompletionResult(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""
