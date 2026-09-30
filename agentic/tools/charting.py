"""Render query results as PNG charts designed to be read by people AND by the chart VLM.

Values are printed on the chart (bars, or the peak of each line) so the ChartQA
model can read them back -- that is what makes the self-verification step work.
A JSON sidecar with the plotted data is saved next to each PNG for audit.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


def compact(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e4:
        return f"{v / 1e3:.1f}k"
    if a >= 100 or float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.1f}"


def render_chart(rows: list[dict[str, Any]], columns: list[str], title: str, unit: str,
                 out_path: str | Path) -> dict[str, Any]:
    """Pick a chart type from the result shape and save it. Returns chart metadata."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dims = [c for c in columns if c not in ("period", "value")]
    series_dim = dims[0] if dims else None
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

    if "period" in columns:
        kind = "line"
        series: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for r in rows:
            series[str(r[series_dim]) if series_dim else unit].append((str(r["period"])[:7], float(r["value"])))
        for name, pts in series.items():
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker="o", label=name)
            i = max(range(len(ys)), key=ys.__getitem__)
            ax.annotate(compact(ys[i]), (xs[i], ys[i]), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=10, fontweight="bold")
        if series_dim:
            ax.legend(title=series_dim, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        ax.tick_params(axis="x", rotation=45)
        plotted = {k: dict(v) for k, v in series.items()}
    else:
        kind = "bar"
        labels = [str(r[series_dim]) if series_dim else "total" for r in rows]
        values = [float(r["value"]) for r in rows]
        bars = ax.bar(labels, values, color="#2F6690")
        for b, v in zip(bars, values, strict=True):
            ax.annotate(compact(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=11, fontweight="bold")
        plotted = dict(zip(labels, values, strict=True))

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: compact(v)))
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(unit)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    meta = {"path": str(out_path), "kind": kind, "title": title, "unit": unit,
            "series_dim": series_dim, "data": plotted}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
