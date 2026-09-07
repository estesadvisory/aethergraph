from __future__ import annotations

from typing import Any

import httpx

from aethergraph.a2a.protocol import AgentCard, Message, Task, new_text_message


class A2AClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get_agent_card(self) -> AgentCard:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/.well-known/agent-card.json")
            response.raise_for_status()
            return AgentCard.model_validate(response.json())

    async def send_message(self, message: Message | str, metadata: dict[str, Any] | None = None) -> Task:
        if isinstance(message, str):
            message = new_text_message(message)
        if metadata:
            message.metadata.update(metadata)
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {"message": message.model_dump(mode="json")},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url + "/", json=payload)
            response.raise_for_status()
            body = response.json()
        if body.get("error"):
            raise RuntimeError(body["error"])
        return Task.model_validate(body["result"]["task"])
