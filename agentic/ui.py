"""Gradio web UI for the analytics agent.

  python -m agentic.ui            # http://localhost:7861

Type a question (optionally attach a dashboard screenshot to reconcile) and see the
answer, the chart, the verification checks, the governed SQL and the audit trace.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from agentic.agent.orchestrator import AnalyticsAgent

EXAMPLES = [
    "Which site had the highest energy intensity last quarter?",
    "Monthly pellet production by site in 2025",
    "Which equipment had the most downtime in 2025?",
    "Which site has the lowest CO2 intensity last quarter?",
    "Number of stops by cause in H1 2026",
]


@lru_cache(maxsize=1)
def get_agent() -> AnalyticsAgent:
    return AnalyticsAgent()


def run(question: str, image_path: str | None) -> tuple[str, str | None, str, str, str]:
    """Returns (answer_md, chart_path, verification_md, sql, trace_json)."""
    if not question or len(question.strip()) < 3:
        return "Please type a question.", None, "", "", ""
    resp = get_agent().ask(question.strip(), image_path=image_path or None, user="ui")
    ev = resp.evidence
    chart = ev["charts"][0] if ev["charts"] else None
    answer = f"### Answer\n{resp.answer}\n\n<sub>planner: `{resp.planner}` · chart reader: " \
             f"`{resp.chart_reader}` · data as of {ev['data_as_of']} · run `{resp.run_id}`</sub>"
    return answer, chart, _verification_md(ev["verification"]), _sql(ev["queries"]), _trace(resp.trace_path)


def _verification_md(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return "_No chart was verified for this question._"
    out = []
    for rep in reports:
        icon = {"passed": "✅", "failed": "❌"}.get(rep["status"], "⚠️")
        source = "uploaded chart" if rep.get("image") == "upload" else "generated chart"
        out.append(f"**{icon} {rep['status'].upper()}** ({source}, reader `{rep['reader']}`, "
                   f"tolerance {rep['tolerance']:.0%})\n")
        out.append("| Check | Expected | Model read | Result |\n|---|---|---|---|")
        for c in rep["checks"]:
            res = {True: "pass", False: "fail", None: "unreadable"}[c["passed"]]
            exp = f"{c['expected']:,.1f}" if isinstance(c["expected"], float) else c["expected"]
            out.append(f"| {c['check']} | {exp} | {c['vlm_answer']} | {res} |")
    return "\n".join(out)


def _sql(queries: list[dict[str, Any]]) -> str:
    parts = []
    for q in queries:
        tag = "governed (semantic layer)" if q["governed"] else "UNGOVERNED"
        parts.append(f"-- {tag}; parameters: {q['params']}\n{q['sql']}")
    return "\n\n".join(parts) or "-- no query was run"


def _trace(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        steps = [json.loads(line) for line in fh]
    slim = [{k: s.get(k) for k in ("event", "tool", "thought", "args", "duration_ms") if s.get(k) is not None}
            for s in steps]
    return json.dumps(slim, indent=2, ensure_ascii=False)


def build_ui():
    import gradio as gr

    with gr.Blocks(title="AskAnything Analytics Agent") as demo:
        gr.Markdown("# AskAnything Analytics Agent\nGoverned answers, verified charts. "
                    "Data is synthetic. Attach a chart image to reconcile it against the warehouse.")
        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(label="Question", lines=2, placeholder=EXAMPLES[0], elem_id="question")
                upload = gr.Image(label="Optional: dashboard screenshot to reconcile", type="filepath", elem_id="upload")
                ask = gr.Button("Ask", variant="primary", elem_id="ask")
                gr.Examples(EXAMPLES, inputs=question)
            with gr.Column(scale=3):
                answer = gr.Markdown(elem_id="answer")
                chart = gr.Image(label="Chart", type="filepath", elem_id="chart")
        with gr.Row():
            verification = gr.Markdown(elem_id="verification")
        with gr.Accordion("Governed SQL", open=False):
            sql = gr.Code(language="sql", elem_id="sql")
        with gr.Accordion("Audit trace", open=False):
            trace = gr.Code(language="json")
        outputs = [answer, chart, verification, sql, trace]
        ask.click(run, inputs=[question, upload], outputs=outputs)
        question.submit(run, inputs=[question, upload], outputs=outputs)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7861)
