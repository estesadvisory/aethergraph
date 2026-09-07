from aethergraph.graph.engine import topological_layers
from aethergraph.planner import plan_graph
from aethergraph.types import GraphNode, GraphSpec, NodeStatus, TaskType


def test_topological_layers_are_parallel_safe():
    spec = GraphSpec(
        id="g",
        nodes=[
            GraphNode(id="a", prompt="a"),
            GraphNode(id="b", prompt="b"),
            GraphNode(id="c", prompt="c", depends_on=["a", "b"]),
        ],
    )
    layers = topological_layers(spec)
    assert set(layers[0]) == {"a", "b"}
    assert layers[1] == ["c"]


async def test_engine_runs_multi_node_graph(engine):
    spec = GraphSpec(
        id="brief",
        input="Need a routing policy for code review.",
        nodes=[
            GraphNode(
                id="extract",
                task_type=TaskType.EXTRACTION,
                prompt="Extract JSON with facts, constraints, deliverable from: {{input}}",
            ),
            GraphNode(
                id="write",
                task_type=TaskType.WRITING,
                depends_on=["extract"],
                prompt="Write a short policy from {{extract.output}}",
            ),
        ],
    )
    result = await engine.run(spec)
    assert result.status is NodeStatus.COMPLETED
    assert "extract" in result.nodes
    assert "write" in result.nodes
    assert result.nodes["extract"].output
    assert "mock fact" in result.nodes["write"].output or result.nodes["write"].output
    assert result.nodes["extract"].model_id in {"mock-nano", "mock-reasoner"}
    assert result.total_cost_usd >= 0


def test_planner_builds_single_node_for_simple_code():
    spec = plan_graph("Implement a binary heap in Python.")
    assert len(spec.nodes) == 1
    assert spec.nodes[0].task_type is TaskType.CODE
