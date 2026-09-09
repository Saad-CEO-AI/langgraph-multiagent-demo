from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from src.graph.graph import build_graph
from src.graph.state import GraphState

DEFAULT_MOCK_QUERY = "What is the capital of France and why is it historically significant?"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Research -> Critique -> Revise multi-agent LangGraph flow."
    )
    parser.add_argument(
        "--query",
        required=False,
        help="The question to answer. Defaults to a sample question when --mock is set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream each agent's step, including any revision loop, as it runs.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline with a scripted fake model -- no API key, no network call. "
        "Deterministic: always triggers one revision loop before approving, so you "
        "can see the whole graph (including the loop) execute end to end.",
    )
    args = parser.parse_args(argv)
    if not args.query and not args.mock:
        parser.error("--query is required unless --mock is set")
    return args


def initial_state(query: str) -> GraphState:
    return GraphState(
        query=query,
        draft="",
        critique="",
        verdict="needs_revision",
        revision_count=0,
        final_answer="",
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.mock:
        os.environ["LLM_PROVIDER"] = "mock"
        print("[mock mode: no API key, no network call -- responses are scripted]", file=sys.stderr)

    graph = build_graph()
    state = initial_state(args.query or DEFAULT_MOCK_QUERY)

    try:
        if args.verbose:
            final_state: dict = {}
            for step in graph.stream(state, stream_mode="updates"):
                for node_name, node_output in step.items():
                    print(f"--- {node_name} ---", file=sys.stderr)
                    for key, value in node_output.items():
                        print(f"{key}: {value}", file=sys.stderr)
                    final_state.update(node_output)
        else:
            final_state = graph.invoke(state)
    except Exception as exc:  # noqa: BLE001 - CLI boundary, report and exit non-zero
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(final_state["final_answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
