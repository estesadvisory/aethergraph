from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict, deque

from aethergraph.cost import CostLedger, LedgerEntry, estimate_cost
from aethergraph.providers.registry import ProviderRegistry
from aethergraph.quality import passes_quality_gate
from aethergraph.router import RoutingError, route
from aethergraph.types import (
    CompletionRequest,
    GraphResult,
    GraphSpec,
    ModelProfile,
    NodeResult,
    NodeStatus,
    RouteRequest,
    RoutingPolicy,
    TaskType,
)


_REF = re.compile(r"\{\{([a-zA-Z0-9_-]+)\.output\}\}")


def topological_layers(spec: GraphSpec) -> list[list[str]]:
    ids = {node.id for node in spec.nodes}
    incoming: dict[str, int] = {node.id: 0 for node in spec.nodes}
    edges: dict[str, list[str]] = defaultdict(list)
    for node in spec.nodes:
        for dep in node.depends_on:
            if dep not in ids:
                raise ValueError(f"Node {node.id} depends on unknown node {dep}")
            incoming[node.id] += 1
            edges[dep].append(node.id)
    ready = deque(node_id for node_id, count in incoming.items() if count == 0)
    layers: list[list[str]] = []
    seen = 0
    while ready:
        layer = list(ready)
        ready.clear()
        layers.append(layer)
        for node_id in layer:
            seen += 1
            for child in edges[node_id]:
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
    if seen != len(spec.nodes):
        raise ValueError("Graph contains a cycle")
    return layers


def render_prompt(template: str, results: dict[str, NodeResult], graph_input: str) -> str:
    rendered = template.replace("{{input}}", graph_input)
    def replace(match: re.Match[str]) -> str:
        node_id = match.group(1)
        result = results.get(node_id)
        return result.output if result else ""
    return _REF.sub(replace, rendered)


class GraphEngine:
    def __init__(
        self,
        models: list[ModelProfile],
        policy: RoutingPolicy,
        providers: ProviderRegistry,
        ledger: CostLedger | None = None,
    ) -> None:
        self.models = {item.id: item for item in models}
        self.model_list = models
        self.policy = policy
        self.providers = providers
        self.ledger = ledger or CostLedger()

    def _available(self, profile: ModelProfile) -> bool:
        return self.providers.available(profile)

    async def run(self, spec: GraphSpec) -> GraphResult:
        results: dict[str, NodeResult] = {}
        nodes = {node.id: node for node in spec.nodes}
        remaining_budget = spec.budget_usd or self.policy.default_budget_usd
        try:
            for layer in topological_layers(spec):
                layer_results = await asyncio.gather(
                    *[
                        self._run_node(nodes[node_id], spec, results, remaining_budget)
                        for node_id in layer
                    ]
                )
                for result in layer_results:
                    results[result.node_id] = result
                    remaining_budget = max(0.0, remaining_budget - result.cost_usd)
                    if result.status is NodeStatus.FAILED:
                        return GraphResult(
                            graph_id=spec.id,
                            status=NodeStatus.FAILED,
                            nodes=results,
                            total_cost_usd=round(sum(item.cost_usd for item in results.values()), 6),
                            output=result.error or "",
                        )
        except Exception as exc:
            return GraphResult(
                graph_id=spec.id,
                status=NodeStatus.FAILED,
                nodes=results,
                total_cost_usd=round(sum(item.cost_usd for item in results.values()), 6),
                output=str(exc),
            )

        terminal = [node.id for node in spec.nodes if not any(node.id in other.depends_on for other in spec.nodes)]
        output = "\n\n".join(results[node_id].output for node_id in terminal if node_id in results)
        return GraphResult(
            graph_id=spec.id,
            status=NodeStatus.COMPLETED,
            nodes=results,
            total_cost_usd=round(sum(item.cost_usd for item in results.values()), 6),
            output=output,
        )

    async def _run_node(
        self,
        node,
        spec: GraphSpec,
        prior: dict[str, NodeResult],
        budget: float,
    ) -> NodeResult:
        prompt = render_prompt(node.prompt, prior, spec.input)
        task_type = node.task_type or TaskType.CONVERSATION
        request = RouteRequest(
            prompt=prompt,
            task_type=task_type,
            max_output_tokens=node.max_output_tokens,
            required_capability=node.required_capability,
            budget_usd=budget,
            metadata=node.metadata,
        )
        try:
            decision = route(request, self.model_list, self.policy, available=self._available)
        except RoutingError as exc:
            return NodeResult(node_id=node.id, status=NodeStatus.FAILED, error=str(exc))

        selected_id = node.model_id or decision.selected.model_id
        result = await self._complete(node.id, spec.id, selected_id, prompt, request, task_type, decision)
        if result.status is NodeStatus.COMPLETED:
            ok, reason = passes_quality_gate(result.output, task_type, self.policy)
            if ok:
                return result
            if not self.policy.quality_gates.get("escalate_on_gate_failure", True):
                result.status = NodeStatus.FAILED
                result.error = reason
                return result
            upgrade = next((item for item in decision.alternates if item.capability > decision.selected.capability), None)
            if upgrade is None:
                result.status = NodeStatus.FAILED
                result.error = f"quality gate failed: {reason}"
                return result
            escalated = await self._complete(
                node.id, spec.id, upgrade.model_id, prompt, request, task_type, decision
            )
            escalated.escalated_from = selected_id
            return escalated
        return result

    async def _complete(
        self,
        node_id: str,
        graph_id: str,
        model_id: str,
        prompt: str,
        request: RouteRequest,
        task_type: TaskType,
        decision,
    ) -> NodeResult:
        profile = self.models[model_id]
        started = time.perf_counter()
        try:
            completion = await self.providers.complete(
                profile,
                CompletionRequest(
                    model_id=model_id,
                    prompt=prompt,
                    max_output_tokens=request.max_output_tokens
                    or self.policy.default_output_tokens.get(task_type, 400),
                ),
            )
        except Exception as exc:
            return NodeResult(
                node_id=node_id,
                status=NodeStatus.FAILED,
                model_id=model_id,
                error=str(exc),
                route=decision,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        cost = estimate_cost(profile, completion.input_tokens, completion.output_tokens)
        self.ledger.record(
            LedgerEntry(
                graph_id=graph_id,
                node_id=node_id,
                model_id=model_id,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=cost,
                task_type=task_type.value,
            )
        )
        return NodeResult(
            node_id=node_id,
            status=NodeStatus.COMPLETED,
            model_id=model_id,
            output=completion.text,
            cost_usd=round(cost, 8),
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=latency_ms,
            route=decision,
        )
