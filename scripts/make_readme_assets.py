"""Regenerate the images used in README.md from real agent runs.

  python -m agentic.data.generate_synthetic
  python scripts/make_readme_assets.py          # writes docs/images/*.png

The UI screenshot is captured separately by scripts/capture_ui_screenshot.py.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("AGENT_RUNS_DIR", tempfile.mkdtemp(prefix="agent_runs_"))
os.environ.setdefault("VLM_BACKEND", "mock")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from agentic.agent.orchestrator import AnalyticsAgent  # noqa: E402
from agentic.agent.verifier import verify_chart  # noqa: E402
from agentic.semantic.layer import QuerySpec  # noqa: E402
from agentic.tools.chart_reader import MockChartReader  # noqa: E402
from agentic.tools.charting import compact, render_chart  # noqa: E402

OUT = REPO / "docs" / "images"
INK, MUTED, OK, BAD = "#1B2026", "#5A6470", "#2F6690", "#C0392B"


def trend(agent: AnalyticsAgent) -> None:
    resp = agent.ask("Monthly pellet production by site in 2025")
    Path(resp.evidence["charts"][0]).replace(OUT / "example_trend.png")
    print("example_trend.png:", resp.answer)


def reconciliation(agent: AnalyticsAgent) -> None:
    """A 'stale dashboard' (numbers from an old extract) is read and checked against the warehouse."""
    spec = QuerySpec(metric="pellets_produced_t", group_by=["site"], start="2026-01-01", end="2026-03-31")
    cq = agent.semantic.compile(spec)
    governed = agent.warehouse.execute(cq.sql, cq.params, governed=True)

    tmp = Path(tempfile.mkdtemp())
    stale_rows = [{"site": r["site"], "value": r["value"] * f}
                  for r, f in zip(governed.rows, [0.78, 1.0, 1.02], strict=True)]
    stale = render_chart(stale_rows, ["site", "value"], "Pellets by site, Q1 2026 (old dashboard)", "t",
                         tmp / "stale_dashboard.png")
    good = render_chart(governed.rows, governed.columns, "Pellets by site, Q1 2026 (warehouse)", "t",
                        tmp / "governed.png")
    report = verify_chart(MockChartReader(), stale["path"], governed, tolerance=0.05)

    fig = plt.figure(figsize=(14, 6.6), dpi=110)
    fig.patch.set_facecolor("#F4F2ED")
    for i, (path, label) in enumerate([(stale["path"], "Uploaded dashboard screenshot"),
                                       (good["path"], "Governed numbers from the warehouse")]):
        ax = fig.add_axes([0.02 + i * 0.49, 0.29, 0.47, 0.66])
        ax.imshow(mpimg.imread(path))
        ax.set_axis_off()
        ax.set_title(label, fontsize=14, color=INK, loc="left", fontweight="bold")
    verdict = report["status"].upper()
    color = BAD if report["status"] == "failed" else OK
    fig.text(0.02, 0.19, f"Verdict: {verdict}", fontsize=20, fontweight="bold", color=color)
    y = 0.135
    for c in report["checks"]:
        exp = compact(c["expected"]) if isinstance(c["expected"], float) else c["expected"]
        res = {True: "pass", False: "FAIL", None: "unreadable"}[c["passed"]]
        extra = f"  (error {c['relative_error']:.0%})" if c.get("relative_error") is not None else ""
        fig.text(0.02, y, f"{c['check']:<10}  warehouse: {exp:<8}  chart read: {c['vlm_answer']:<8}  -> {res}{extra}",
                 fontsize=13, family="monospace", color=INK)
        y -= 0.06
    fig.text(0.02, 0.012, "Chart read simulated with the CI mock reader. With VLM_BACKEND=qwen the fine-tuned "
             "ChartQA model reads the image.", fontsize=11, color=MUTED)
    fig.savefig(OUT / "reconciliation.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("reconciliation.png:", verdict)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    agent = AnalyticsAgent()
    trend(agent)
    reconciliation(agent)


if __name__ == "__main__":
    main()
