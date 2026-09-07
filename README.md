# AetherGraph

Cost-aware **A2A v1.0** graph engineering for AI work. AetherGraph compiles a request into a DAG, then binds each node to the cheapest model that can do that kind of work — cloud APIs or local Ollama/vLLM.

It is meant to be used, not just diagrammed:

- Agent Cards at `/.well-known/agent-card.json`
- JSON-RPC `SendMessage` / `GetTask` / `ListTasks` / `CancelTask`
- A model catalog with capability scores and USD/million-token prices
- A router that applies capability floors, latency, locality, and budget
- A graph engine with parallel layers and one-step escalation

See the repository files for the full implementation, architecture, and tests.
