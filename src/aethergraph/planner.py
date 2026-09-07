from __future__ import annotations

import re
import uuid

from aethergraph.classifier import classify
from aethergraph.types import GraphNode, GraphSpec, TaskType


def plan_graph(goal: str, graph_id: str | None = None) -> GraphSpec:
    """Turn a free-form goal into a small practical DAG.

    This is deliberately heuristic. A production deployment can replace it
    with a planner model while keeping the same GraphSpec contract.
    """
    classification = classify(goal)
    graph_id = graph_id or f"graph-{uuid.uuid4().hex[:10]}"
    lowered = goal.lower()

    multi_step = bool(
        re.search(r"\b(then|after that|and then|finally|followed by)\b", lowered)
        or (re.search(r"\b(extract|summarize|analyze|write|review)\b", lowered) and len(goal) > 180)
    )

    if classification.task_type is TaskType.PLANNING or re.search(r"\b(roadmap|decompose)\b", lowered):
        return GraphSpec(
            id=graph_id,
            input=goal,
            nodes=[
                GraphNode(
                    id="plan",
                    task_type=TaskType.PLANNING,
                    prompt=f"Decompose this goal into a concrete work plan:\n\n{goal}",
                ),
                GraphNode(
                    id="critique",
                    task_type=TaskType.REASONING,
                    depends_on=["plan"],
                    prompt=(
                        "Review this plan for missing dependencies and risk. "
                        "Return a revised plan.\n\n{{plan.output}}"
                    ),
                ),
            ],
        )

    if classification.task_type is TaskType.CODE and re.search(r"\b(review|audit|refactor)\b", lowered):
        return GraphSpec(
            id=graph_id,
            input=goal,
            nodes=[
                GraphNode(id="review", task_type=TaskType.CODE, prompt=goal),
                GraphNode(
                    id="fix",
                    task_type=TaskType.CODE,
                    depends_on=["review"],
                    prompt="Apply the highest-value fixes from this review:\n\n{{review.output}}",
                ),
            ],
        )

    if multi_step or (
        classification.task_type in {TaskType.WRITING, TaskType.REASONING} and len(goal) > 240
    ):
        return GraphSpec(
            id=graph_id,
            input=goal,
            nodes=[
                GraphNode(
                    id="extract",
                    task_type=TaskType.EXTRACTION,
                    prompt=(
                        "Extract the facts, constraints, and desired outputs as JSON "
                        f"with keys facts, constraints, deliverable.\n\n{goal}"
                    ),
                ),
                GraphNode(
                    id="analyze",
                    task_type=TaskType.REASONING,
                    depends_on=["extract"],
                    prompt="Analyze this structured brief and decide the answer:\n\n{{extract.output}}",
                ),
                GraphNode(
                    id="write",
                    task_type=TaskType.WRITING,
                    depends_on=["analyze"],
                    prompt="Write the final deliverable from this analysis:\n\n{{analyze.output}}",
                ),
            ],
        )

    return GraphSpec(
        id=graph_id,
        input=goal,
        nodes=[
            GraphNode(
                id="task",
                task_type=classification.task_type,
                prompt=goal,
            )
        ],
    )
