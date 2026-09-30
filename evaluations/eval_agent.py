"""Evaluate the analytics agent on a golden question set.

For each question it checks
  * routing   - did the planner pick the right governed metric and dimensions?
  * accuracy  - does the agent's top answer match an independently written SQL ground truth?
  * verified  - did the chart read-back (VLM) agree with the data?

  python evaluations/eval_agent.py --min-accuracy 0.9
Set LLM_BASE_URL to evaluate the LLM planner, VLM_BACKEND=qwen for the real chart reader.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic.agent.orchestrator import AnalyticsAgent  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default=str(Path(__file__).with_name("golden_questions.jsonl")))
    p.add_argument("--min-accuracy", type=float, default=0.0)
    p.add_argument("--out", default="outputs/agent_eval.json")
    args = p.parse_args()

    agent = AnalyticsAgent()
    con = duckdb.connect(agent.s.warehouse_path, read_only=True)
    cases = [json.loads(line) for line in open(args.golden, encoding="utf-8") if line.strip()]
    results = []
    for c in cases:
        resp = agent.ask(c["question"])
        truth_label, truth_value = con.execute(c["truth_sql"]).fetchone()
        q = resp.evidence["queries"][0] if resp.evidence["queries"] else None
        spec_ok = accurate = False
        if q and q["rows"]:
            trace = [json.loads(line) for line in open(resp.trace_path, encoding="utf-8")]
            call = next(t for t in trace if t.get("tool") == "query_metric")
            spec_ok = call["args"]["metric"] == c["metric"] and set(call["args"]["group_by"]) == set(c["group_by"])
            pick = min if call["args"].get("order") == "asc" else max
            best = pick(q["rows"], key=lambda r: r["value"])
            label_ok = truth_label is None or str(truth_label) in map(str, best.values())
            value_ok = abs(best["value"] - truth_value) <= 1e-3 * max(1.0, abs(truth_value))
            accurate = label_ok and value_ok
        verified = bool(resp.evidence["verification"]) and resp.evidence["verification"][0]["status"] == "passed"
        results.append({"id": c["id"], "question": c["question"], "routing_ok": spec_ok,
                        "accurate": accurate, "verified": verified, "answer": resp.answer})
        print(f"{c['id']}  routing={'OK ' if spec_ok else 'ERR'}  accurate={'OK ' if accurate else 'ERR'}  "
              f"verified={'OK ' if verified else 'ERR'}  {c['question']}")

    n = len(results)
    summary = {"n": n, "planner": agent.planner.name, "chart_reader": agent.reader.name,
               "routing_accuracy": sum(r["routing_ok"] for r in results) / n,
               "answer_accuracy": sum(r["accurate"] for r in results) / n,
               "verification_pass_rate": sum(r["verified"] for r in results) / n}
    print(json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "cases": results}, indent=2, ensure_ascii=False))
    if summary["answer_accuracy"] < args.min_accuracy:
        sys.exit(f"Answer accuracy {summary['answer_accuracy']:.2f} below threshold {args.min_accuracy}")


if __name__ == "__main__":
    main()
