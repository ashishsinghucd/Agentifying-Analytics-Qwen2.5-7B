"""Command line entry point.

  python -m agentic.cli build-data
  python -m agentic.cli ask "Which site had the highest energy intensity in Q2 2026?"
  python -m agentic.cli ask "Does this chart match our data for pellets by site in 2025?" --image dash.png
  python -m agentic.cli export-dax > measures.dax
"""
from __future__ import annotations

import argparse
import json

from agentic.config import Settings


def main() -> None:
    p = argparse.ArgumentParser(prog="agentic")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-data", help="Create the synthetic DuckDB warehouse")
    ask = sub.add_parser("ask", help="Ask the analytics agent a question")
    ask.add_argument("question")
    ask.add_argument("--image", help="Optional chart image to read and reconcile")
    ask.add_argument("--json", action="store_true", help="Print the full evidence pack")
    sub.add_parser("export-dax", help="Print the semantic model as Power BI DAX measures")
    args = p.parse_args()

    if args.cmd == "build-data":
        from agentic.data.generate_synthetic import build
        print(build(Settings.from_env().warehouse_path))
    elif args.cmd == "export-dax":
        from agentic.semantic.layer import SemanticLayer
        print(SemanticLayer(Settings.from_env().metrics_path).to_dax())
    else:
        from agentic.agent.orchestrator import AnalyticsAgent
        resp = AnalyticsAgent().ask(args.question, image_path=args.image)
        if args.json:
            print(json.dumps(resp.to_dict(), indent=2, ensure_ascii=False, default=str))
        else:
            print(resp.answer)
            for c in resp.evidence["charts"]:
                print(f"  chart: {c}")
            for v in resp.evidence["verification"]:
                print(f"  verification: {v['status']} (score {v['score']}, reader {v['reader']})")
            print(f"  trace: {resp.trace_path}")


if __name__ == "__main__":
    main()
