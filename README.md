# AetherGraph

Cost-aware **A2A v1.0** graph engineering for AI work. AetherGraph compiles a request into a DAG, then binds each node to the cheapest model that can do that kind of work — cloud APIs or local Ollama/vLLM.

It is meant to be used, not just diagrammed:

- Agent Cards at `/.well-known/agent-card.json`
- JSON-RPC `SendMessage` / `GetTask` / `ListTasks` / `CancelTask`
- A model catalog with capability scores and USD/million-token prices
- A router that applies capability floors, latency, locality, and budget
- A graph engine with parallel layers and one-step escalation

## Why this exists

Frontier models are excellent at hard reasoning and messy code. They are a waste of money on classification, extraction, and summaries. Local 8B–14B models flip that: cheap (or free on your GPU), good enough for structured work, weak on deep analysis.

AetherGraph makes that split explicit. A research brief becomes:

1. **extract** → Gemini Flash Lite / local 8B / mock-nano
2. **analyze** → o3 / Sonnet / local 70B
3. **write** → mid-tier cloud or local 70B

The expensive model never sees the cheap node.

## Quick start

```bash
cd aethergraph
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Route a prompt without calling a paid API (mock models are always available):

```bash
aethergraph route "Classify this ticket as billing, bug, or feature."
aethergraph plan "Review this FastAPI service and then draft a fix plan."
aethergraph run examples/research_brief.json
```

Serve the A2A gateway:

```bash
aethergraph serve --port 8080
curl -s http://127.0.0.1:8080/.well-known/agent-card.json | python3 -m json.tool
```

Send a task:

```bash
curl -s http://127.0.0.1:8080/ \
  -H 'content-type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "SendMessage",
    "params": {
      "message": {
        "role": "ROLE_USER",
        "parts": [{"text": "Summarize why local models should win classification."}],
        "metadata": {"skill": "execute-graph"}
      }
    }
  }'
```

## Cloud and local models

Copy `.env.example` and set the keys you have. Missing keys simply mark that provider unavailable; the router skips those models.

| Provider | Env | Notes |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | GPT-4.1 family, o3, embeddings |
| Anthropic | `ANTHROPIC_API_KEY` | Haiku / Sonnet / Opus |
| Google | `GOOGLE_API_KEY` | Gemini 2.5 Flash / Pro |
| Groq | `GROQ_API_KEY` | OpenAI-compatible, low latency |
| Ollama | `OLLAMA_HOST` | Discovers local tags via `/api/tags` |
| LiteLLM | `LITELLM_BASE_URL` | Optional fabric: vendor keys live on the proxy |
| Mock | none | Always on for tests and dry runs |

If `LITELLM_BASE_URL` is set (or `AETHERGRAPH_FABRIC=litellm`), cloud completions go through a [LiteLLM](https://github.com/BerriAI/litellm) proxy. AetherGraph still owns task typing, graph binding, capability floors, and cost policy. LiteLLM owns vendor SDKs, retries, and key vaults. Direct adapters stay as the fallback when no proxy is configured.

Edit `configs/models.yaml` to change prices, capability scores, or add a vLLM endpoint (`provider: openai_compat` + `base_url`). Use `litellm_model` when the LiteLLM slug differs from the native model id. Edit `configs/policies.yaml` to raise/lower capability floors or the local-preference bonus.

## Library use

```python
from aethergraph.factory import Runtime
from aethergraph.planner import plan_graph

runtime = Runtime()
spec = plan_graph("Extract risks from this RFC, then recommend a rollout.")
result = await runtime.engine.run(spec)
print(result.output, result.total_cost_usd)
```

## Design

See [ARCHITECTURE.md](ARCHITECTURE.md) for the A2A mapping, routing math, catalog contract, and extension points.

## Status

Working control plane: A2A surface, graph router, direct adapters, and an optional LiteLLM fabric. Still in-memory for tasks and the cost ledger. Next production steps are auth on Agent Cards, durable stores, SSE streaming, and split worker processes.
