from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskState(str, Enum):
    SUBMITTED = "TASK_STATE_SUBMITTED"
    WORKING = "TASK_STATE_WORKING"
    COMPLETED = "TASK_STATE_COMPLETED"
    FAILED = "TASK_STATE_FAILED"
    CANCELED = "TASK_STATE_CANCELED"
    REJECTED = "TASK_STATE_REJECTED"
    INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"


class Role(str, Enum):
    USER = "ROLE_USER"
    AGENT = "ROLE_AGENT"


class Part(BaseModel):
    text: str | None = None
    data: dict[str, Any] | None = None
    url: str | None = None
    media_type: str | None = None


class Message(BaseModel):
    role: Role
    parts: list[Part]
    message_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str | None = None
    context_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStatus(BaseModel):
    state: TaskState
    message: Message | None = None


class Artifact(BaseModel):
    name: str
    parts: list[Part]


class Task(BaseModel):
    id: str
    context_id: str
    status: TaskStatus
    history: list[Message] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentProvider(BaseModel):
    organization: str
    url: str


class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    extendedAgentCard: bool = False


class AgentInterface(BaseModel):
    url: str
    protocolBinding: str
    protocolVersion: str = "1.0"


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str]
    examples: list[str] = Field(default_factory=list)
    inputModes: list[str] = Field(default_factory=lambda: ["text/plain", "application/json"])
    outputModes: list[str] = Field(default_factory=lambda: ["text/plain", "application/json"])


class AgentCard(BaseModel):
    name: str
    description: str
    supportedInterfaces: list[AgentInterface]
    version: str
    capabilities: AgentCapabilities
    defaultInputModes: list[str] = Field(default_factory=lambda: ["text/plain", "application/json"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["text/plain", "application/json"])
    skills: list[AgentSkill]
    provider: AgentProvider | None = None
    documentationUrl: str | None = None


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Any | None = None


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int | None = None
    result: Any | None = None
    error: JsonRpcError | None = None


def message_text(message: Message) -> str:
    return "\n".join(part.text for part in message.parts if part.text)


def new_text_message(text: str, role: Role = Role.USER) -> Message:
    return Message(role=role, parts=[Part(text=text)])
