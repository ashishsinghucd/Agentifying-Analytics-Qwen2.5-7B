"""Chart <-> data consistency checks.

The same routine is used for
  * self-verification: the agent's own chart, read back by the VLM, must match the SQL result;
  * reconciliation: a chart from an existing dashboard or PDF report is read by the VLM
    and compared with the governed number in the warehouse.
"""
from __future__ import annotations

import unicodedata
from typing import Any

from agentic.tools.chart_reader import ChartReader, parse_number
from agentic.tools.warehouse import QueryResult


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s)).casefold().strip()


def expected_extremes(result: QueryResult) -> tuple[str | None, float]:
    dims = [c for c in result.columns if c not in ("period", "value")]
    best = max(result.rows, key=lambda r: r["value"])
    label = str(best[dims[0]]) if dims else None
    return label, float(best["value"])


def verify_chart(reader: ChartReader, image_path: str, result: QueryResult,
                 tolerance: float = 0.05) -> dict[str, Any]:
    if not result.rows:
        return {"status": "skipped", "reason": "empty result", "checks": []}
    label, value = expected_extremes(result)
    checks: list[dict[str, Any]] = []

    answer = reader.ask(image_path, "What is the highest value shown in the chart? Answer with the number only.")
    read = parse_number(answer)
    rel_err = abs(read - value) / abs(value) if (read is not None and value) else None
    checks.append({"check": "max_value", "expected": value, "vlm_answer": answer,
                   "parsed": read, "relative_error": rel_err,
                   "passed": None if rel_err is None else rel_err <= tolerance})

    dims = [c for c in result.columns if c not in ("period", "value")]
    categories = {str(r[dims[0]]) for r in result.rows} if dims else set()
    if label is not None and len(categories) > 1:
        answer = reader.ask(image_path, f"Which {dims[0]} has the highest value in the chart? "
                                        "Answer with the label only.")
        unreadable = not answer.strip() or _norm(answer) == "unknown"
        ok = None if unreadable else (_norm(label) in _norm(answer) or _norm(answer) in _norm(label))
        checks.append({"check": "top_label", "expected": label, "vlm_answer": answer, "passed": ok})

    outcomes = [c["passed"] for c in checks if c["passed"] is not None]
    if False in outcomes:
        status = "failed"
    elif outcomes and len(outcomes) == len(checks):
        status = "passed"
    else:
        status = "inconclusive"  # the chart could not be read; never reported as a match
    return {"status": status, "score": round(sum(outcomes) / len(outcomes), 2) if outcomes else None,
            "reader": reader.name, "tolerance": tolerance, "checks": checks}
