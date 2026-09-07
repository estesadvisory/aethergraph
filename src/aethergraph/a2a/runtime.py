from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from aethergraph.a2a.protocol import (
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    message_text,
)
from aethergraph.graph.engine import GraphEngine
from aethergraph.planner import plan_graph
from aethergraph.router import route
from aethergraph.types import GraphSpec, RouteRequest


Handler = Callable[[Task, Message], Awaitable[Task]]


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def put(self, task: Task) -> Task:
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list(self, context_id: str | None = None, status: str | None = None) -> list[Task]:
        items = list(self._tasks.values())
        if context_id:
            items = [item for item in items if item.context_id == context_id]
        if status:
            items = [item for item in items if item.status.state.value == status]
        return items


class GatewayHandler:
    def __init__(self, engine: GraphEngine) -> None:
        self.engine = engine

    async def handle(self, task: Task, message: Message) -> Task:
        text = message_text(message)
        data = next((part.data for part in message.parts if part.data), None) or {}
        skill = (message.metadata or {}).get("skill") or data.get("skill") or "execute-graph"

        if skill == "route-task":
            decision = route(
                RouteRequest(prompt=text or data.get("prompt", "")),
                self.engine.model_list,
                self.engine.policy,
                available=self.engine._available,
            )
            payload = decision.model_dump(mode="json")
            return _complete(task, message, text=decision.selected.model_id, data=payload)

        if skill == "plan-graph":
            spec = plan_graph(text or data.get("goal", ""))
            return _complete(task, message, text=spec.model_dump_json(), data=spec.model_dump(mode="json"))

        spec = _graph_from_message(text, data)
        result = await self.engine.run(spec)
        return _complete(
            task,
            message,
            text=result.output,
            data=result.model_dump(mode="json"),
            artifact_name="graph-result",
        )


def _graph_from_message(text: str, data: dict[str, Any]) -> GraphSpec:
    if data.get("nodes"):
        return GraphSpec.model_validate(data)
    if data.get("graph"):
        return GraphSpec.model_validate(data["graph"])
    if text.strip().startswith("{"):
        return GraphSpec.model_validate_json(text)
    return plan_graph(text)


def _complete(
    task: Task,
    user_message: Message,
    text: str,
    data: dict[str, Any] | None = None,
    artifact_name: str = "result",
) -> Task:
    agent_message = Message(
        role=Role.AGENT,
        parts=[Part(text=text, data=data)],
        task_id=task.id,
        context_id=task.context_id,
    )
    task.status = TaskStatus(state=TaskState.COMPLETED, message=agent_message)
    task.history = [user_message, agent_message]
    task.artifacts = [Artifact(name=artifact_name, parts=agent_message.parts)]
    return task


def new_task(message: Message) -> Task:
    task_id = message.task_id or uuid4().hex
    context_id = message.context_id or uuid4().hex
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=TaskState.WORKING),
        history=[message],
    )
