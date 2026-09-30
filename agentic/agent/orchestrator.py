"""The agent loop: plan -> call tool -> observe -> ... -> final answer + evidence pack.

Every answer ships with its evidence (SQL, parameters, charts, verification report)
and a JSONL trace, so a human can check exactly how a number was produced.
"""
from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from agentic.agent.planners import LLMPlanner, Planner, RulePlanner
from agentic.agent.verifier import verify_chart
from agentic.audit import RunTrace, Timer
from agentic.config import Settings
from agentic.semantic.layer import QuerySpec, SemanticError, SemanticLayer
from agentic.tools.chart_reader import ChartReader, build_reader
from agentic.tools.charting import render_chart
from agentic.tools.warehouse import QueryResult, SQLGuardError, Warehouse


@dataclass
class AgentResponse:
    run_id: str
    question: str
    answer: str
    planner: str
    chart_reader: str
    steps: int
    evidence: dict[str, Any] = field(default_factory=dict)
    trace_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnalyticsAgent:
    def __init__(self, settings: Settings | None = None, reader: ChartReader | None = None,
                 planner: Planner | None = None):
        self.s = settings or Settings.from_env()
        if not Path(self.s.warehouse_path).exists():
            raise FileNotFoundError(f"No warehouse at {self.s.warehouse_path}. "
                                    "Run: python -m agentic.data.generate_synthetic")
        self.semantic = SemanticLayer(self.s.metrics_path)
        self.warehouse = Warehouse(self.s.warehouse_path, self.semantic.all_tables(), self.s.max_rows)
        self.reader = reader or build_reader(self.s.vlm_backend, self.s.vlm_base_model, self.s.vlm_adapter)
        self.as_of = self._as_of()
        if planner is not None:
            self.planner = planner
        elif self.s.llm_base_url:
            self.planner = LLMPlanner(self.s.llm_base_url, self.s.llm_model, self.s.llm_api_key,
                                      self.semantic, self.as_of)
        else:
            self.planner = RulePlanner(self.semantic, self._known_values(), self.as_of)

    # ------------------------------------------------------------------ setup
    def _as_of(self) -> date:
        con = duckdb.connect(self.s.warehouse_path, read_only=True)
        try:
            return con.execute("SELECT max(date) FROM fact_production_daily").fetchone()[0]
        finally:
            con.close()

    def _known_values(self) -> dict[str, list[str]]:
        """Distinct values of low-cardinality dimensions, so the planner can spot filters."""
        out: dict[str, list[str]] = {}
        for model in self.semantic.doc["models"].values():
            joins = " ".join(f"JOIN {j['table']} {j['alias']} ON {j['condition']}" for j in model.get("joins", []))
            for dim, d in model["dimensions"].items():
                if dim in out:
                    continue
                res = self.warehouse.execute(
                    f"SELECT DISTINCT {d['expr']} AS v FROM {model['table']} f {joins} LIMIT 50")
                out[dim] = [str(r["v"]) for r in res.rows]
        return out

    # ------------------------------------------------------------------ run
    def ask(self, question: str, image_path: str | None = None, user: str = "anonymous") -> AgentResponse:
        trace = RunTrace(self.s.runs_dir, question, user)
        state = _RunState(trace=trace)
        if image_path:
            upload = trace.dir / ("upload" + Path(image_path).suffix)
            shutil.copy(image_path, upload)
            state.images["upload"] = str(upload)

        history: list[dict[str, Any]] = []
        answer = "I could not finish within the step budget."
        for _ in range(self.s.max_steps):
            try:
                action = self.planner.next_action(question, history, has_upload=bool(image_path))
            except Exception as exc:  # malformed LLM output, network error...
                trace.log("planner_error", {"error": str(exc)})
                history.append({"action": {"tool": "planner_error", "args": {}},
                                "observation": {"error": f"{exc}. Reply with one valid JSON action."}})
                continue
            tool, args = action["tool"], action.get("args", {})
            if tool == "final_answer":
                answer = str(args.get("answer", ""))
                trace.log("final_answer", {"thought": action.get("thought"), "answer": answer})
                break
            with Timer() as t:
                observation = self._call(tool, args, state)
            trace.log("tool_call", {"thought": action.get("thought"), "tool": tool, "args": args,
                                    "observation": _short(observation)}, duration_ms=t.ms)
            history.append({"action": action, "observation": observation})

        return AgentResponse(
            run_id=trace.run_id, question=question, answer=answer, planner=self.planner.name,
            chart_reader=self.reader.name, steps=len(history),
            evidence={
                "queries": [{"sql": r.sql, "params": r.params, "governed": r.governed,
                             "rows": r.preview()} for r in state.results.values()],
                "charts": [p for k, p in state.images.items() if k != "upload"],
                "verification": state.verifications,
                "data_as_of": self.as_of.isoformat(),
            },
            trace_path=str(trace.file))

    # ------------------------------------------------------------------ tools
    def _call(self, tool: str, args: dict[str, Any], st: _RunState) -> dict[str, Any]:
        try:
            if tool == "list_metrics":
                return {"metrics": self.semantic.catalog()}
            if tool == "query_metric":
                spec = QuerySpec.from_dict(args)
                cq = self.semantic.compile(spec, self.s.max_rows)
                res = self.warehouse.execute(cq.sql, cq.params, governed=True)
                rid = st.add_result(res, cq.metric)
                return {"result_id": rid, "columns": res.columns, "rows": res.preview(40),
                        "row_count": len(res.rows), "unit": cq.metric["unit"],
                        "metric_label": cq.metric["label"], "time_grain": spec.time_grain,
                        "order": spec.order, "title": _title(cq.metric, spec), "sql": cq.sql}
            if tool == "run_sql":
                res = self.warehouse.execute(str(args["sql"]), governed=False)
                rid = st.add_result(res, {"label": "Custom query", "unit": ""})
                return {"result_id": rid, "columns": res.columns, "rows": res.preview(40),
                        "row_count": len(res.rows), "warning": "UNGOVERNED query: not from the semantic model"}
            if tool == "make_chart":
                rid = args["result_id"]
                res, metric = st.results[rid], st.metrics[rid]
                if "value" not in res.columns:
                    return {"error": "Result needs a 'value' column to be charted"}
                cid = f"c{len(st.images) + 1}"
                meta = render_chart(res.rows, res.columns, args.get("title") or metric["label"],
                                    metric.get("unit", ""), st.trace.dir / f"{cid}.png")
                st.images[cid] = meta["path"]
                return {"chart_id": cid, "path": meta["path"], "kind": meta["kind"]}
            if tool == "read_chart":
                path = st.image(args["image"])
                return {"answer": self.reader.ask(path, str(args["question"])), "reader": self.reader.name}
            if tool == "verify_chart":
                report = verify_chart(self.reader, st.image(args["image"]), st.results[args["result_id"]],
                                      self.s.verify_tolerance)
                report["image"] = args["image"]
                st.verifications.append(report)
                return report
            return {"error": f"Unknown tool '{tool}'"}
        except (SemanticError, SQLGuardError, KeyError, TypeError, ValueError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


@dataclass
class _RunState:
    trace: RunTrace
    results: dict[str, QueryResult] = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    images: dict[str, str] = field(default_factory=dict)
    verifications: list[dict[str, Any]] = field(default_factory=list)

    def add_result(self, res: QueryResult, metric: dict[str, Any]) -> str:
        rid = f"r{len(self.results) + 1}"
        self.results[rid], self.metrics[rid] = res, metric
        return rid

    def image(self, ref: str) -> str:
        # Only images created in this run or uploaded by the user: no arbitrary file paths.
        if ref not in self.images:
            raise KeyError(f"Unknown image '{ref}'. Use a chart_id or 'upload'.")
        return self.images[ref]


def _title(metric: dict[str, Any], spec: QuerySpec) -> str:
    parts = [metric["label"]]
    if spec.group_by:
        parts.append("by " + ", ".join(spec.group_by))
    if spec.time_grain:
        parts.append(f"per {spec.time_grain}")
    if spec.start or spec.end:
        parts.append(f"({spec.start or '...'} to {spec.end or '...'})")
    return " ".join(parts)


def _short(obs: dict[str, Any]) -> dict[str, Any]:
    out = dict(obs)
    if isinstance(out.get("rows"), list) and len(out["rows"]) > 5:
        out["rows"] = out["rows"][:5] + [f"... {len(obs['rows']) - 5} more"]
    return out
