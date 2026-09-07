from __future__ import annotations

import argparse
import json
import sys

import uvicorn

from aethergraph.factory import Runtime
from aethergraph.planner import plan_graph
from aethergraph.router import route
from aethergraph.types import GraphSpec, RouteRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aethergraph", description="A2A graph router")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the A2A gateway")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    route_cmd = sub.add_parser("route", help="Classify and route a prompt")
    route_cmd.add_argument("prompt")

    plan_cmd = sub.add_parser("plan", help="Compile a prompt into a graph")
    plan_cmd.add_argument("prompt")

    run_cmd = sub.add_parser("run", help="Execute a prompt or GraphSpec JSON file")
    run_cmd.add_argument("target")

    args = parser.parse_args(argv)
    runtime = Runtime()

    if args.command == "serve":
        app = runtime.app()
        uvicorn.run(
            app,
            host=args.host or runtime.settings.host,
            port=args.port or runtime.settings.port,
        )
        return 0

    if args.command == "route":
        decision = route(
            RouteRequest(prompt=args.prompt),
            runtime.models,
            runtime.policy,
            available=runtime.engine._available,
        )
        print(json.dumps(decision.model_dump(mode="json"), indent=2))
        return 0

    if args.command == "plan":
        spec = plan_graph(args.prompt)
        print(spec.model_dump_json(indent=2))
        return 0

    if args.command == "run":
        spec = _load_target(args.target)
        result = _run(runtime, spec)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "completed" else 1

    parser.error("unknown command")
    return 2


def _load_target(target: str) -> GraphSpec:
    if target.endswith(".json"):
        return GraphSpec.model_validate_json(open(target, encoding="utf-8").read())
    if target.endswith(".yaml") or target.endswith(".yml"):
        import yaml

        return GraphSpec.model_validate(yaml.safe_load(open(target, encoding="utf-8")))
    return plan_graph(target)


def _run(runtime: Runtime, spec: GraphSpec) -> dict:
    import asyncio

    result = asyncio.run(runtime.engine.run(spec))
    return result.model_dump(mode="json")


if __name__ == "__main__":
    sys.exit(main())
