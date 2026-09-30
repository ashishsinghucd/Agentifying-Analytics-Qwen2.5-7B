"""Fast, GPU-free tests for the agentic layer (run in CI with the mock chart reader)."""
from __future__ import annotations

import dataclasses

import pytest

from agentic.agent.orchestrator import AnalyticsAgent
from agentic.config import Settings
from agentic.data.generate_synthetic import build
from agentic.semantic.layer import QuerySpec, SemanticError, SemanticLayer
from agentic.tools.chart_reader import parse_number
from agentic.tools.charting import render_chart
from agentic.tools.warehouse import SQLGuardError, check_sql

TABLES = {"fact_production_daily", "fact_downtime", "dim_site"}


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    root = tmp_path_factory.mktemp("agentic")
    wh = build(root / "wh.duckdb", seed=7)
    return dataclasses.replace(Settings.from_env(), warehouse_path=str(wh), runs_dir=str(root / "runs"),
                               llm_base_url=None, vlm_backend="mock")


@pytest.fixture(scope="session")
def agent(settings) -> AnalyticsAgent:
    return AnalyticsAgent(settings)


# ---------------------------------------------------------------- SQL guard
@pytest.mark.parametrize("sql", [
    "DROP TABLE dim_site",
    "SELECT 1; DELETE FROM dim_site",
    "SELECT * FROM secret_table",
    "SELECT * FROM read_csv('/etc/passwd')",
    "COPY dim_site TO 'out.csv'",
    "ATTACH 'other.db'",
])
def test_guard_blocks_unsafe_sql(sql):
    with pytest.raises(SQLGuardError):
        check_sql(sql, TABLES)


def test_guard_allows_select_with_cte():
    check_sql("WITH x AS (SELECT * FROM dim_site) SELECT count(*) FROM x", TABLES)


# ---------------------------------------------------------------- semantic layer
def test_compile_binds_filters_as_parameters(settings):
    sem = SemanticLayer(settings.metrics_path)
    cq = sem.compile(QuerySpec(metric="energy_intensity_kwh_per_t", group_by=["site"],
                               filters={"site": ["Norra'; DROP TABLE x;--"]}))
    assert "DROP" not in cq.sql and "Norra'; DROP TABLE x;--" in cq.params
    assert "NULLIF" in cq.sql


@pytest.mark.parametrize("spec", [
    {"metric": "made_up_metric"},
    {"metric": "pellets_produced_t", "group_by": ["equipment"]},  # wrong model
    {"metric": "pellets_produced_t", "time_grain": "fortnight"},
    {"metric": "pellets_produced_t", "start": "last tuesday"},
])
def test_compile_rejects_invalid_specs(settings, spec):
    with pytest.raises(SemanticError):
        SemanticLayer(settings.metrics_path).compile(QuerySpec.from_dict(spec))


def test_dax_export_matches_semantic_model(settings):
    dax = SemanticLayer(settings.metrics_path).to_dax()
    assert "[Energy intensity (kWh/t)] = DIVIDE(" in dax
    assert "COUNTROWS('fact_downtime')" in dax


# ---------------------------------------------------------------- tools
@pytest.mark.parametrize("text,expected", [("1.61M", 1.61e6), ("45,300 t", 45300), ("about 3.5k", 3500), ("none", None)])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


# ---------------------------------------------------------------- end to end
def test_agent_answers_with_evidence_and_verification(agent):
    r = agent.ask("Which site had the highest energy intensity in 2025?")
    assert r.planner == "rules"
    q = r.evidence["queries"][0]
    assert q["governed"] is True
    top = q["rows"][0]
    assert r.answer.startswith(top["site"])
    assert r.evidence["charts"] and r.evidence["verification"][0]["status"] == "passed"


def test_agent_declines_when_no_metric_matches(agent):
    r = agent.ask("What will the weather be tomorrow?")
    assert "No governed metric" in r.answer and not r.evidence["queries"]


class ScriptedReader:
    """Simulates the VLM reading a chart whose numbers disagree with the warehouse."""
    name = "scripted"

    def __init__(self, answers):
        self.answers = answers

    def ask(self, image_path, question):
        return self.answers["label"] if question.startswith("Which") else self.answers["value"]


def test_reconciliation_flags_a_wrong_dashboard(settings, tmp_path):
    fake = render_chart([{"site": "Norra", "value": 9.9e6}, {"site": "Väst", "value": 1e5}],
                        ["site", "value"], "Stale dashboard", "t", tmp_path / "stale.png")
    agent = AnalyticsAgent(settings, reader=ScriptedReader({"label": "Norra", "value": "9.90M"}))
    r = agent.ask("Pellets produced by site in 2025", image_path=fake["path"])
    assert r.evidence["verification"][0]["status"] == "failed"
    assert "does NOT match" in r.answer


def test_unreadable_chart_is_inconclusive_not_a_match(settings, tmp_path):
    agent = AnalyticsAgent(settings, reader=ScriptedReader({"label": "unknown", "value": "unknown"}))
    fake = render_chart([{"site": "Norra", "value": 1.0}], ["site", "value"], "x", "t", tmp_path / "x.png")
    r = agent.ask("Pellets produced by site in 2025", image_path=fake["path"])
    assert r.evidence["verification"][0]["status"] == "inconclusive"
    assert "not verified" in r.answer


def test_llm_planner_loop_recovers_from_bad_output(settings, monkeypatch):
    """Simulates an OpenAI-compatible endpoint: one malformed reply, then a valid tool sequence."""
    import json

    from agentic.agent import planners

    replies = iter([
        "Sure! I think we should query the data.",  # not JSON -> must be recovered from
        {"thought": "query", "tool": "query_metric",
         "args": {"metric": "downtime_hours", "group_by": ["equipment"], "start": "2025-01-01", "end": "2025-12-31"}},
        {"thought": "chart", "tool": "make_chart", "args": {"result_id": "r1", "title": "Downtime 2025"}},
        {"thought": "verify", "tool": "verify_chart", "args": {"image": "c1", "result_id": "r1"}},
        {"thought": "done", "tool": "final_answer", "args": {"answer": "Conveyor had the most downtime."}},
    ])

    class FakeResponse:
        def __init__(self, content):
            self.content = content if isinstance(content, str) else json.dumps(content)

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": self.content}}]}

    monkeypatch.setattr(planners.httpx, "post", lambda *a, **k: FakeResponse(next(replies)))
    base = AnalyticsAgent(settings)
    llm = planners.LLMPlanner("http://fake/v1", "fake-model", "x", base.semantic, base.as_of)
    r = AnalyticsAgent(settings, planner=llm).ask("Which equipment had the most downtime in 2025?")
    assert r.planner == "llm" and r.answer == "Conveyor had the most downtime."
    assert r.evidence["verification"][0]["status"] == "passed"
    assert any('"planner_error"' in line for line in open(r.trace_path, encoding="utf-8"))


def test_ui_handler_returns_answer_chart_and_evidence(settings, monkeypatch):
    """The Gradio callback works without launching a server (gradio itself is optional)."""
    from agentic import ui

    monkeypatch.setattr(ui, "get_agent", lambda: AnalyticsAgent(settings))
    answer, chart, verification, sql, trace = ui.run("Which equipment had the most downtime in 2025?", None)
    assert "highest downtime" in answer and chart.endswith(".png")
    assert "PASSED" in verification and "governed" in sql and "query_metric" in trace
