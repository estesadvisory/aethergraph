from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from aethergraph.a2a.protocol import (
    AgentCard,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    TaskState,
)
from aethergraph.a2a.runtime import TaskStore, new_task

Handler = Callable[..., Awaitable]


def create_a2a_app(card: AgentCard, handler: Handler, store: TaskStore | None = None) -> FastAPI:
    store = store or TaskStore()
    app = FastAPI(title=card.name, version=card.version)
    app.state.agent_card = card
    app.state.handler = handler
    app.state.store = store

    @app.get("/.well-known/agent-card.json")
    async def agent_card() -> dict:
        return card.model_dump(mode="json", exclude_none=True)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "agent": card.name}

    @app.post("/")
    async def jsonrpc(request: Request) -> JSONResponse:
        payload = await request.json()
        rpc = JsonRpcRequest.model_validate(payload)
        try:
            result = await dispatch(rpc, card, handler, store)
            return JSONResponse(JsonRpcResponse(id=rpc.id, result=result).model_dump(mode="json"))
        except JsonRpcException as exc:
            return JSONResponse(
                JsonRpcResponse(
                    id=rpc.id, error=JsonRpcError(code=exc.code, message=exc.message, data=exc.data)
                ).model_dump(mode="json")
            )

    @app.post("/message:send")
    async def rest_send(payload: dict) -> dict:
        message = Message.model_validate(payload.get("message") or payload)
        task = await execute(message, handler, store)
        return {"task": task.model_dump(mode="json")}

    @app.get("/tasks/{task_id}")
    async def rest_get(task_id: str) -> dict:
        task = store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task.model_dump(mode="json")

    @app.get("/tasks")
    async def rest_list(contextId: str | None = None, status: str | None = None) -> dict:
        return {"tasks": [item.model_dump(mode="json") for item in store.list(contextId, status)]}

    return app


async def dispatch(rpc: JsonRpcRequest, card: AgentCard, handler: Handler, store: TaskStore) -> dict:
    method = rpc.method
    params = rpc.params or {}
    if method == "SendMessage":
        message = Message.model_validate(params.get("message") or params)
        task = await execute(message, handler, store)
        return {"task": task.model_dump(mode="json")}
    if method == "GetTask":
        task = store.get(str(params.get("id")))
        if not task:
            raise JsonRpcException(-32001, "Task not found")
        return task.model_dump(mode="json")
    if method == "ListTasks":
        return {
            "tasks": [
                item.model_dump(mode="json")
                for item in store.list(params.get("contextId"), params.get("status"))
            ]
        }
    if method == "CancelTask":
        task = store.get(str(params.get("id")))
        if not task:
            raise JsonRpcException(-32001, "Task not found")
        if task.status.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED}:
            raise JsonRpcException(-32002, "Task is not cancelable")
        task.status.state = TaskState.CANCELED
        store.put(task)
        return task.model_dump(mode="json")
    if method == "GetExtendedAgentCard":
        raise JsonRpcException(-32004, "Extended agent card is not supported")
    raise JsonRpcException(-32601, f"Method not found: {method}")


async def execute(message: Message, handler: Handler, store: TaskStore):
    task = new_task(message)
    store.put(task)
    completed = await handler.handle(task, message)
    return store.put(completed)


class JsonRpcException(Exception):
    def __init__(self, code: int, message: str, data=None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
