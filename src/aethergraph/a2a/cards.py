from __future__ import annotations

from aethergraph.a2a.protocol import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)
from aethergraph.types import TaskType

_SKILL_COPY: dict[str, tuple[str, str, list[str]]] = {
    TaskType.EXTRACTION.value: (
        "Structured Extraction",
        "Pull facts, fields, and JSON from unstructured text.",
        ["extract", "json", "schema"],
    ),
    TaskType.CLASSIFICATION.value: (
        "Classification",
        "Label or categorize short inputs at low cost.",
        ["classify", "label"],
    ),
    TaskType.SUMMARIZATION.value: (
        "Summarization",
        "Compress long documents into key points.",
        ["summarize", "tldr"],
    ),
    TaskType.REASONING.value: (
        "Reasoning",
        "Hard analysis, tradeoffs, and multi-step logic.",
        ["analyze", "why", "tradeoff"],
    ),
    TaskType.PLANNING.value: (
        "Planning",
        "Decompose a goal into an executable graph.",
        ["plan", "roadmap"],
    ),
    TaskType.CODE.value: (
        "Code Engineering",
        "Generate, review, or refactor source code.",
        ["code", "review", "implement"],
    ),
    TaskType.WRITING.value: (
        "Writing",
        "Draft long-form prose from structured notes.",
        ["write", "draft"],
    ),
    TaskType.CONVERSATION.value: (
        "Conversation",
        "General assistant replies.",
        ["chat"],
    ),
    TaskType.TRANSLATION.value: (
        "Translation",
        "Translate text between languages.",
        ["translate"],
    ),
    TaskType.VISION.value: (
        "Vision",
        "Describe or extract from images.",
        ["image", "screenshot"],
    ),
    TaskType.EMBEDDING.value: (
        "Embeddings",
        "Create vectors for retrieval.",
        ["embed", "vector"],
    ),
}


def skill_for(task_type: str) -> AgentSkill:
    name, description, tags = _SKILL_COPY.get(
        task_type, (task_type.title(), f"Handle {task_type} tasks.", [task_type])
    )
    return AgentSkill(id=task_type, name=name, description=description, tags=tags)


def build_gateway_card(public_url: str) -> AgentCard:
    return AgentCard(
        name="AetherGraph Gateway",
        description=(
            "Plans work as an AI graph and routes each node to the cheapest "
            "capable cloud or local model."
        ),
        supportedInterfaces=[
            AgentInterface(url=f"{public_url}/", protocolBinding="JSONRPC", protocolVersion="1.0"),
            AgentInterface(
                url=f"{public_url}/a2a", protocolBinding="HTTP+JSON", protocolVersion="1.0"
            ),
        ],
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        skills=[
            AgentSkill(
                id="execute-graph",
                name="Execute Graph",
                description="Run a GraphSpec DAG with cost-aware model binding.",
                tags=["graph", "routing", "a2a"],
                examples=['{"id":"demo","nodes":[{"id":"n1","task_type":"extraction","prompt":"..."}]}'],
            ),
            AgentSkill(
                id="route-task",
                name="Route Task",
                description="Classify a prompt and select the cheapest capable model.",
                tags=["routing", "cost"],
            ),
            AgentSkill(
                id="plan-graph",
                name="Plan Graph",
                description="Turn a free-form goal into an executable DAG.",
                tags=["planning", "graph"],
            ),
        ],
        provider=AgentProvider(organization="AetherGraph", url="https://github.com/estesadvisory/aethergraph"),
        documentationUrl="https://github.com/estesadvisory/aethergraph",
    )


def build_worker_card(name: str, description: str, public_url: str, skills: list[str]) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        supportedInterfaces=[
            AgentInterface(url=f"{public_url}/", protocolBinding="JSONRPC", protocolVersion="1.0"),
        ],
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill_for(skill) for skill in skills],
        provider=AgentProvider(organization="AetherGraph", url="https://github.com/estesadvisory/aethergraph"),
    )
