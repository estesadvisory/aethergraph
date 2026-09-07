from fastapi.testclient import TestClient

from aethergraph.a2a.cards import build_gateway_card
from aethergraph.a2a.protocol import new_text_message
from aethergraph.factory import Runtime


def test_agent_card_is_a2a_v1():
    card = build_gateway_card("http://127.0.0.1:8080")
    payload = card.model_dump(mode="json")
    assert payload["supportedInterfaces"][0]["protocolVersion"] == "1.0"
    assert payload["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"
    assert {skill["id"] for skill in payload["skills"]} >= {"execute-graph", "route-task", "plan-graph"}


def test_jsonrpc_send_message_routes_and_completes():
    app = Runtime().app()
    client = TestClient(app)

    card = client.get("/.well-known/agent-card.json")
    assert card.status_code == 200
    assert card.json()["name"] == "AetherGraph Gateway"

    response = client.post(
        "/",
        json={
            "jsonrpc": "2.0",
            "id": 7,
            "method": "SendMessage",
            "params": {
                "message": new_text_message("Classify this ticket as billing or bug.").model_dump(mode="json")
                | {"metadata": {"skill": "route-task"}}
            },
        },
    )
    body = response.json()
    assert body["id"] == 7
    task = body["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["data"]["selected"]["model_id"] == "mock-nano"


def test_unknown_method_is_jsonrpc_error():
    app = Runtime().app()
    client = TestClient(app)
    response = client.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "Nope", "params": {}})
    assert response.json()["error"]["code"] == -32601
