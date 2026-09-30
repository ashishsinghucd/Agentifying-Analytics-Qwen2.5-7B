"""Semantic layer: validates agent query specs and compiles them to SQL or DAX.

The agent (LLM or rule-based) only produces a *QuerySpec* -- a metric name,
dimensions, time grain, date range and filters. All aggregation logic comes
from metrics.yml, so every consumer (dashboard, API, AI agent) computes a
metric the same way. Filter values are always bound as SQL parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml


class SemanticError(ValueError):
    """Raised when a query spec does not match the semantic model."""


@dataclass
class QuerySpec:
    metric: str
    group_by: list[str] = field(default_factory=list)
    time_grain: str | None = None
    start: str | None = None  # ISO date, inclusive
    end: str | None = None  # ISO date, inclusive
    filters: dict[str, list[str]] = field(default_factory=dict)
    order: str | None = None  # "asc" | "desc" | None (auto)
    limit: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuerySpec:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(d) - allowed
        if unknown:
            raise SemanticError(f"Unknown query fields: {sorted(unknown)}")
        spec = cls(**d)
        spec.group_by = list(spec.group_by or [])
        spec.filters = {k: [str(x) for x in (v if isinstance(v, list) else [v])]
                        for k, v in (spec.filters or {}).items()}
        return spec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompiledQuery:
    sql: str
    params: list[Any]
    metric: dict[str, Any]
    model: str
    columns: list[str]


class SemanticLayer:
    def __init__(self, path: str | Path):
        with open(path, encoding="utf-8") as fh:
            self.doc = yaml.safe_load(fh)
        self.grains: list[str] = self.doc["time_grains"]
        self.metric_index: dict[str, str] = {}
        for model_name, model in self.doc["models"].items():
            for metric_name in model["metrics"]:
                if metric_name in self.metric_index:
                    raise SemanticError(f"Duplicate metric '{metric_name}'")
                self.metric_index[metric_name] = model_name

    # ---------- introspection ----------
    def model_for(self, metric: str) -> tuple[str, dict[str, Any]]:
        if metric not in self.metric_index:
            raise SemanticError(f"Unknown metric '{metric}'. Known: {sorted(self.metric_index)}")
        name = self.metric_index[metric]
        return name, self.doc["models"][name]

    def metric(self, metric: str) -> dict[str, Any]:
        return {"name": metric, **self.model_for(metric)[1]["metrics"][metric]}

    def all_tables(self) -> set[str]:
        tables = set()
        for model in self.doc["models"].values():
            tables.add(model["table"])
            tables.update(j["table"] for j in model.get("joins", []))
        return tables

    def catalog(self) -> list[dict[str, Any]]:
        """Compact, LLM-friendly description of what can be asked."""
        out = []
        for model in self.doc["models"].values():
            for name, m in model["metrics"].items():
                out.append({
                    "metric": name, "label": m["label"], "unit": m["unit"],
                    "description": m["description"], "synonyms": m.get("synonyms", []),
                    "dimensions": list(model["dimensions"]), "time_grains": self.grains,
                })
        return out

    def dimension_synonyms(self) -> dict[str, list[str]]:
        syn: dict[str, set[str]] = {}
        for model in self.doc["models"].values():
            for dim, d in model["dimensions"].items():
                syn.setdefault(dim, set()).update(d.get("synonyms", [dim]))
        return {k: sorted(v) for k, v in syn.items()}

    # ---------- compilation ----------
    def _metric_sql(self, model: dict[str, Any], m: dict[str, Any]) -> str:
        agg = m["agg"]
        if agg == "sum":
            return f"SUM(f.{m['column']})"
        if agg == "avg":
            return f"AVG(f.{m['column']})"
        if agg == "count":
            return "COUNT(*)"
        if agg == "ratio":
            num = model["metrics"][m["numerator"]]["column"]
            den = model["metrics"][m["denominator"]]["column"]
            scale = float(m.get("scale", 1))
            return f"{scale} * SUM(f.{num}) / NULLIF(SUM(f.{den}), 0)"
        raise SemanticError(f"Unsupported aggregation '{agg}'")

    def compile(self, spec: QuerySpec, max_rows: int = 5000) -> CompiledQuery:
        model_name, model = self.model_for(spec.metric)
        m = model["metrics"][spec.metric]
        dims = model["dimensions"]

        for d in list(spec.group_by) + list(spec.filters):
            if d not in dims:
                raise SemanticError(
                    f"Dimension '{d}' is not available for metric '{spec.metric}'. "
                    f"Available: {sorted(dims)}")
        if spec.time_grain and spec.time_grain not in self.grains:
            raise SemanticError(f"Unknown time grain '{spec.time_grain}'")
        if spec.order not in (None, "asc", "desc"):
            raise SemanticError("order must be 'asc', 'desc' or null")
        for label, value in (("start", spec.start), ("end", spec.end)):
            if value:
                try:
                    date.fromisoformat(value)
                except ValueError as exc:
                    raise SemanticError(f"{label} must be an ISO date, got '{value}'") from exc

        select, columns = [], []
        if spec.time_grain:
            select.append(f"CAST(date_trunc('{spec.time_grain}', f.{model['time_column']}) AS DATE) AS period")
            columns.append("period")
        for d in spec.group_by:
            select.append(f"{dims[d]['expr']} AS {d}")
            columns.append(d)
        select.append(f"{self._metric_sql(model, m)} AS value")
        columns.append("value")

        joins = " ".join(f"JOIN {j['table']} {j['alias']} ON {j['condition']}" for j in model.get("joins", []))
        where, params = [], []
        if spec.start:
            where.append(f"f.{model['time_column']} >= ?")
            params.append(spec.start)
        if spec.end:
            where.append(f"f.{model['time_column']} <= ?")
            params.append(spec.end)
        for d, values in spec.filters.items():
            where.append(f"{dims[d]['expr']} IN ({', '.join('?' for _ in values)})")
            params.extend(values)

        sql = f"SELECT {', '.join(select)} FROM {model['table']} f {joins}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        if len(columns) > 1:
            sql += " GROUP BY " + ", ".join(str(i) for i in range(1, len(columns)))
        if spec.time_grain:
            sql += " ORDER BY period" + ("".join(f", {d}" for d in spec.group_by))
        else:
            sql += f" ORDER BY value {(spec.order or 'desc').upper()}"
        sql += f" LIMIT {min(int(spec.limit or max_rows), max_rows)}"
        return CompiledQuery(sql=sql, params=params, metric=self.metric(spec.metric),
                             model=model_name, columns=columns)

    # ---------- Power BI export ----------
    def to_dax(self) -> str:
        """Export every metric as a Power BI DAX measure (same logic, human-facing)."""
        lines = []
        for model in self.doc["models"].values():
            t = f"'{model['table']}'"
            for name, m in model["metrics"].items():
                agg = m["agg"]
                if agg == "sum":
                    expr = f"SUM({t}[{m['column']}])"
                elif agg == "avg":
                    expr = f"AVERAGE({t}[{m['column']}])"
                elif agg == "count":
                    expr = f"COUNTROWS({t})"
                else:
                    num = model["metrics"][m["numerator"]]["column"]
                    den = model["metrics"][m["denominator"]]["column"]
                    expr = f"DIVIDE(SUM({t}[{num}]), SUM({t}[{den}])) * {m.get('scale', 1)}"
                lines.append(f"// {m['description']} Unit: {m['unit']}. Source metric: {name}")
                lines.append(f"[{m['label']} ({m['unit']})] = {expr}")
                lines.append("")
        return "\n".join(lines)
