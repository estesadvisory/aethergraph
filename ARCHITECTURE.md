# AetherGraph Architecture

AetherGraph is a practical Agent2Agent (A2A) v1.0 system for **graph-shaped AI work** and **cost-aware model routing**. It treats models as replaceable workers, not as the application.

The design goal is not a research scheduler. It is an operations-friendly control plane that can:

- accept work from people or other agents over A2A
- compile that work into a DAG
- bind each node to the cheapest model that can actually do the job
- run cloud and local models behind the same contract
- escalate only when a cheap attempt fails a quality gate

## Why A2A

[A2A v1.0](https://a2a-protocol.org/latest/specification/) is the horizontal protocol for agents: discovery, task lifecycle, and message exchange across vendors and frameworks. MCP remains the vertical protocol for tools. AetherGraph uses A2A as the *inter-agent* fabric and keeps model providers behind a thin adapter layer.

Each process publishes an Agent Card at `/.well-known/agent-card.json` and speaks JSON-RPC 2.0 methods from the spec:

| Operation | JSON-RPC method |
| --- | --- |
| Send a message / start work | `SendMessage` |
| Inspect a task | `GetTask` |
| List tasks | `ListTasks` |
| Cancel work | `CancelTask` |

Task states use the v1.0 names: `TASK_STATE_SUBMITTED`, `TASK_STATE_WORKING`, `TASK_STATE_COMPLETED`, `TASK_STATE_FAILED`, `TASK_STATE_CANCELED`.

That means a LangGraph agent, a Google ADK agent, or another AetherGraph cluster can delegate work without a proprietary SDK.

## System shape

```
 Client / peer agent
        |  A2A JSON-RPC 1.0
        v
 Gateway Agent Card
   plan-graph | route-task | execute-graph
        |
        +-- Classifier (task type + complexity)
        +-- Planner    (goal -> GraphSpec DAG)
        +-- Router     (capability floor + cost + latency + locality)
        +-- Graph engine (parallel layers, artifact passing, escalation)
        |
        v
 Specialist worker cards          Provider fabric
   extract | reason | code   -->  OpenAI, Anthropic, Gemini,
   write   | vision          -->  Groq, Ollama, vLLM, mock
```

In single-node mode the workers are logical skills on the gateway. In distributed mode they are independent A2A servers with the same cards. The graph engine does not care which transport is used.

## Graph IR

Work is a `GraphSpec`, not a prompt soup.

```json
{
  "id": "research-brief",
  "input": "Compare local 70B vs Sonnet for code review",
  "budget_usd": 0.05,
  "nodes": [
    {"id": "extract", "task_type": "extraction", "prompt": "... {{input}}"},
    {"id": "analyze", "task_type": "reasoning", "depends_on": ["extract"], "prompt": "... {{extract.output}}"},
    {"id": "write", "task_type": "writing", "depends_on": ["analyze"], "prompt": "... {{analyze.output}}"}
  ]
}
```

Rules:

- Edges are `depends_on`. Ready nodes in a layer run concurrently.
- Prompts interpolate `{{input}}` and `{{node.output}}`.
- Each node is routed independently. Extraction can land on a flash-lite or local 8B model while reasoning lands on a frontier model.
- A graph budget is a hard cap distributed as nodes complete.

This is the difference between "an LLM app" and graph engineering: the expensive model only sees the node that needs it.

## Task taxonomy

The classifier assigns one of:

`classification`, `extraction`, `summarization`, `conversation`, `writing`, `translation`, `planning`, `code`, `reasoning`, `vision`, `embedding`

plus a complexity score from 1 to 5.

The default classifier is heuristic (keywords, code fences, length, modalities). That is intentional: classification itself should be free or nearly free. A production site can swap in a cheap-model second pass without changing the router.

## Routing

Routing is a constrained optimization, not a lookup table.

1. Determine `required_capability` from policy floors `[task_type][complexity]`.
2. Drop models that are unhealthy, missing credentials, missing a modality, over context, over budget, or below the capability floor.
3. Score survivors:

   `score = estimated_usd + latency_s * latency_weight - local_bonus`

4. Pick the lowest score. Keep a short alternate list for escalation and shadow logs.

Estimated USD uses catalog prices:

` (input_tokens * input_usd_per_mtok + output_tokens * output_usd_per_mtok) / 1e6 `

Local models are priced at $0. The local bonus exists because "free tokens" are not free if a 70B box is saturated. Tune `local_bonus_usd` and `latency_weight_usd_per_second` until the policy matches how you value GPU time versus API invoices.

### Capability floors

Floors keep cheap models off work they will fail. Example from `configs/policies.yaml`:

| Complexity | Classification | Code | Reasoning |
| --- | --- | --- | --- |
| 1 | 4 | 5 | 5 |
| 3 | 6 | 7 | 7 |
| 5 | 8 | 9 | 9 |

A "classify this ticket" task should never wake Opus. A five-constraint architecture review should never land on an 8B chat model.

### Cascade

If a quality gate fails (empty output, extraction that is not JSON) the engine retries once with the next higher-capability alternate. That is the main cost lever: start cheap, pay more only on failure.

## Model catalog

`configs/models.yaml` is the source of truth. Each profile has:

- provider + model id
- locality (`local` | `cloud`)
- context window
- input/output USD per million tokens
- p50 latency
- modalities
- capability scores 0-10 per task type

Availability is runtime, not config:

- `openai` / `openai_compat` / `anthropic` / `google`: API key present
- `ollama`: `/api/tags` reachable and the model is listed
- `mock`: always on, used for tests and dry runs

Add Groq, Together, Fireworks, or vLLM by pointing `openai_compat` at their base URL. Add a new Ollama tag by copying a local profile.

## Providers

Adapters speak HTTP. There is no vendor SDK in the hot path.

| Provider | Endpoint style |
| --- | --- |
| `openai` | OpenAI chat completions |
| `openai_compat` | Same shape (Groq, vLLM, Together) |
| `anthropic` | Anthropic messages |
| `google` | Gemini `generateContent` |
| `ollama` | Ollama `/api/chat` |
| `litellm` | Optional fabric: OpenAI-compatible LiteLLM proxy |
| `mock` | Deterministic local stub |

Set `LITELLM_BASE_URL` (fabric `auto` or `litellm`) to send OpenAI, Anthropic, Gemini, and Groq calls through LiteLLM. Mock stays in-process. Ollama stays direct unless `AETHERGRAPH_LITELLM_INCLUDE_LOCAL=true`. Model slugs are mapped (`anthropic/…`, `gemini/…`, `groq/…`) or overridden with `litellm_model` on the catalog entry.

Embeddings are catalogued so the router can send `embedding` nodes to `text-embedding-3-small` or `nomic-embed-text` instead of a chat model.

## A2A agent roles

| Agent | Skills | Typical models |
| --- | --- | --- |
| Gateway | `plan-graph`, `route-task`, `execute-graph` | cheap planner, then delegates |
| Extract | extraction, classification, summarization, embedding | flash-lite, Haiku, local 8B |
| Reason | reasoning, planning | o3, Opus, Sonnet, local 70B |
| Code | code | Sonnet, GPT-4.1, Qwen2.5-Coder |
| Write | writing, conversation, translation | mid-tier cloud or local 70B |
| Vision | vision | Gemini Flash/Pro, LLaVA |

The gateway is the only agent most clients need to know. Workers can be split out later without changing GraphSpec.

## Execution path

```
SendMessage
  -> classify or honor caller task_type
  -> plan GraphSpec if the body is not already a graph
  -> for each ready layer:
        route node
        call provider
        record ledger
        quality gate
        optional escalate
  -> return A2A Task + artifacts
```

The cost ledger is in-memory in this release. Swap `CostLedger` for Postgres or a metrics backend when you need multi-process totals.

## What this is not

- Not a training orchestrator.
- Not a replacement for Kubernetes. Run the gateway as a normal service and scale workers independently.
- Not a guarantee of official `a2a-sdk` class compatibility. The wire format follows A2A v1.0 Agent Cards and JSON-RPC methods so official clients can still discover and call it.

## Extension points

Keep these interfaces stable and you can grow the system without a rewrite:

| Seam | Replace with |
| --- | --- |
| `classifier.classify` | cheap-model or fine-tuned classifier |
| `planner.plan_graph` | frontier planner that emits GraphSpec |
| `router.route` | linear program, bandit, or learned policy |
| `ProviderRegistry` | LiteLLM fabric, extra vendors, batch APIs, on-prem vLLM |
| `CostLedger` | durable store, FinOps export |
| `GatewayHandler` | auth, tenancy, policy packs per team |

The practical production path is: start with the heuristic classifier and YAML catalog, measure actual cost and quality for two weeks, then tighten floors and add a learned router only where the heuristics are wrong.
