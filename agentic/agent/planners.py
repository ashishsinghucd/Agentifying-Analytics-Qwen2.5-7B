"""Planners decide the agent's next tool call.

* LLMPlanner  - any OpenAI-compatible chat endpoint (Ollama, vLLM, Azure OpenAI...).
                Uses a plain JSON action protocol so small local models work too.
* RulePlanner - deterministic keyword planner over the semantic layer. Used in CI,
                in offline demos, and as a baseline to evaluate the LLM planner against.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any, Protocol

import httpx

from agentic.semantic.layer import SemanticLayer
from agentic.tools.charting import compact

TOOLS_DOC = """Tools (call exactly one per turn):
- list_metrics {} -> governed metrics, their dimensions and time grains.
- query_metric {"metric", "group_by": [..], "time_grain": null|"day"|"week"|"month"|"quarter"|"year",
                "start": "YYYY-MM-DD"|null, "end": "YYYY-MM-DD"|null, "filters": {dim: [values]},
                "order": "asc"|"desc"|null, "limit": int|null} -> result_id + rows. PREFERRED.
- run_sql {"sql"} -> read-only SELECT on allow-listed tables. Only if no metric fits; flagged ungoverned.
- make_chart {"result_id", "title"} -> chart_id. Values are printed on the chart.
- read_chart {"image": chart_id|"upload", "question"} -> answer from the fine-tuned ChartQA vision model.
- verify_chart {"image": chart_id|"upload", "result_id"} -> checks the chart against the data.
- final_answer {"answer"} -> ends the run. Cite numbers with units and the period."""

SYSTEM_PROMPT = """You are an analytics agent for industrial operations data.
Rules: answer only from tool results; prefer query_metric over run_sql; always make_chart
and verify_chart before final_answer when you queried data; if verification fails, say so.
Reply with ONE JSON object only: {{"thought": "...", "tool": "...", "args": {{...}}}}

{tools}

Data available up to {as_of}. Metric catalog:
{catalog}"""


class Planner(Protocol):
    name: str

    def next_action(self, question: str, history: list[dict[str, Any]], has_upload: bool) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- LLM
class LLMPlanner:
    name = "llm"

    def __init__(self, base_url: str, model: str, api_key: str, semantic: SemanticLayer, as_of: date):
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key = model, api_key
        self.system = SYSTEM_PROMPT.format(tools=TOOLS_DOC, as_of=as_of.isoformat(),
                                           catalog=json.dumps(semantic.catalog(), ensure_ascii=False))

    def next_action(self, question: str, history: list[dict[str, Any]], has_upload: bool) -> dict[str, Any]:
        user = question + ("\n(The user uploaded a chart image: use image='upload'.)" if has_upload else "")
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        for h in history:
            messages.append({"role": "assistant", "content": json.dumps(h["action"], ensure_ascii=False)})
            messages.append({"role": "user", "content": "Observation: " + json.dumps(h["observation"], default=str)[:6000]})
        resp = httpx.post(f"{self.base_url}/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "messages": messages, "temperature": 0,
                                "response_format": {"type": "json_object"}})
        resp.raise_for_status()
        return parse_action(resp.json()["choices"][0]["message"]["content"])


def parse_action(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON action in model output: {text[:200]}")
    action = json.loads(match.group(0))
    if "tool" not in action:
        raise ValueError("Action is missing 'tool'")
    action.setdefault("args", {})
    return action


# --------------------------------------------------------------------------- rules
GRAINS = {
    "day": ["daily", "per day", "by day"], "week": ["weekly", "per week", "by week"],
    "month": ["monthly", "per month", "by month", "trend", "over time", "each month"],
    "quarter": ["quarterly", "per quarter", "by quarter"], "year": ["yearly", "annual", "per year", "by year"],
}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


class RulePlanner:
    name = "rules"

    def __init__(self, semantic: SemanticLayer, known_values: dict[str, list[str]], as_of: date):
        self.sem, self.values, self.as_of = semantic, known_values, as_of

    # ---- parsing -------------------------------------------------------------
    def parse(self, question: str) -> dict[str, Any]:
        q = " " + question.lower() + " "
        best, best_len = None, 0
        for m in self.sem.catalog():
            for syn in [m["metric"].replace("_", " "), m["label"].lower(), *m["synonyms"]]:
                if re.search(rf"\b{re.escape(syn.lower())}\b", q) and len(syn) > best_len:
                    best, best_len = m, len(syn)
        if best is None:
            raise ValueError("No governed metric matches the question")
        dims = best["dimensions"]

        group_by, filters = [], {}
        for dim, syns in self.sem.dimension_synonyms().items():
            if dim not in dims:
                continue
            for s in syns:
                if re.search(rf"\b(by|per|each|which|across|every|compare)\s+(the\s+)?{s}s?\b", q):
                    group_by.append(dim)
                    break
            mentioned = [v for v in self.values.get(dim, []) if re.search(rf"\b{re.escape(v.lower())}\b", q)]
            if mentioned:
                filters[dim] = mentioned
                if len(mentioned) > 1 and dim not in group_by:
                    group_by.append(dim)

        grain = next((g for g, kws in GRAINS.items() if any(k in q for k in kws)), None)
        start, end, period_text = self._period(q)
        order = "asc" if re.search(r"\b(lowest|least|smallest|minimum|fewest)\b", q) else None
        top = re.search(r"\btop\s+(\d+)\b", q)
        return {"metric": best["metric"], "group_by": group_by, "time_grain": grain,
                "start": start, "end": end, "filters": filters, "order": order,
                "limit": int(top.group(1)) if top else None, "_period_text": period_text}

    def _period(self, q: str) -> tuple[str | None, str | None, str]:
        a = self.as_of
        if m := re.search(r"\bq([1-4])\s+(20\d\d)\b", q):
            qn, y = int(m.group(1)), int(m.group(2))
            s = date(y, 3 * qn - 2, 1)
            return s.isoformat(), _month_end(y, 3 * qn).isoformat(), f"Q{qn} {y}"
        if m := re.search(r"\bh([12])\s+(20\d\d)\b", q):
            h, y = int(m.group(1)), int(m.group(2))
            return date(y, 1 if h == 1 else 7, 1).isoformat(), _month_end(y, 6 if h == 1 else 12).isoformat(), f"H{h} {y}"
        for i, name in enumerate(MONTHS, start=1):
            if m := re.search(rf"\b{name}\s+(20\d\d)\b", q):
                y = int(m.group(1))
                return date(y, i, 1).isoformat(), _month_end(y, i).isoformat(), f"{name.title()} {y}"
        if m := re.search(r"\b(20\d\d)\b", q):
            y = int(m.group(1))
            return f"{y}-01-01", f"{y}-12-31", str(y)
        if "last quarter" in q:  # latest COMPLETE quarter in the data
            qs = date(a.year, 3 * ((a.month - 1) // 3) + 1, 1)
            if a != _month_end(qs.year, qs.month + 2):
                prev = qs - timedelta(days=1)
                qs = date(prev.year, prev.month - 2, 1)
            return qs.isoformat(), _month_end(qs.year, qs.month + 2).isoformat(), f"Q{(qs.month - 1) // 3 + 1} {qs.year}"
        if "last month" in q:  # latest COMPLETE month in the data
            ms = date(a.year, a.month, 1)
            if a != _month_end(a.year, a.month):
                ms = date((ms - timedelta(days=1)).year, (ms - timedelta(days=1)).month, 1)
            return ms.isoformat(), _month_end(ms.year, ms.month).isoformat(), f"{MONTHS[ms.month - 1].title()} {ms.year}"
        if "this year" in q or "ytd" in q or "year to date" in q:
            return date(a.year, 1, 1).isoformat(), a.isoformat(), f"{a.year} year to date"
        if "last 12 months" in q or "past year" in q:
            return (a - timedelta(days=365)).isoformat(), a.isoformat(), "the last 12 months"
        return None, None, "the full period available"

    # ---- plan ----------------------------------------------------------------
    def next_action(self, question: str, history: list[dict[str, Any]], has_upload: bool) -> dict[str, Any]:
        done = [h["action"]["tool"] for h in history]
        obs = {h["action"]["tool"]: h["observation"] for h in history}
        if has_upload and "read_chart" not in done:
            return _act("Read the uploaded chart with the ChartQA model.", "read_chart",
                        {"image": "upload", "question": question})
        if "query_metric" not in done:
            try:
                spec = self.parse(question)
            except ValueError:
                return _act("No governed metric fits.", "final_answer",
                            {"answer": _no_metric(obs.get("read_chart"))})
            spec.pop("_period_text")
            return _act("Query the governed metric.", "query_metric", spec)
        result = obs["query_metric"]
        if "error" in result:
            return _act("Query failed.", "final_answer", {"answer": f"I could not run the query: {result['error']}"})
        if "make_chart" not in done:
            return _act("Visualise the result.", "make_chart", {"result_id": result["result_id"], "title": result["title"]})
        if "verify_chart" not in done:
            image = "upload" if has_upload else obs["make_chart"]["chart_id"]
            return _act("Check the chart against the data.", "verify_chart",
                        {"image": image, "result_id": result["result_id"]})
        return _act("Summarise.", "final_answer",
                    {"answer": self.summarise(question, result, obs["verify_chart"], has_upload)})

    def summarise(self, question: str, result: dict[str, Any], verification: dict[str, Any], upload: bool) -> str:
        rows, unit, label = result["rows"], result["unit"], result["metric_label"]
        period = self.parse(question)["_period_text"]
        dims = [c for c in result["columns"] if c not in ("period", "value")]
        if not rows:
            return f"No data found for {label} in {period}."
        if "period" in result["columns"]:
            peak = max(rows, key=lambda r: r["value"])
            who = f" at {peak[dims[0]]}" if dims else ""
            last_period = rows[-1]["period"]
            latest = [r for r in rows if r["period"] == last_period]
            latest_txt = ", ".join((f"{r[dims[0]]} " if dims else "") + f"{compact(r['value'])} {unit}" for r in latest)
            text = (f"{label} per {result['time_grain']} over {period}: the peak was {compact(peak['value'])} {unit}"
                    f"{who} in {str(peak['period'])[:7]}; the latest period ({str(last_period)[:7]}) shows {latest_txt}.")
        elif dims:
            first = rows[0]
            word = "lowest" if result.get("order") == "asc" else "highest"
            ranking = ", ".join(f"{r[dims[0]]} {compact(r['value'])}" for r in rows)
            text = (f"{first[dims[0]]} has the {word} {_lower(label)} in {period}: {compact(first['value'])} {unit}. "
                    f"Ranking ({unit}): {ranking}.")
        else:
            text = f"{label} in {period} was {compact(rows[0]['value'])} {unit}."
        status, reader = verification.get("status"), verification.get("reader", "")
        how = "ChartQA model" if reader != "mock" else "mock reader; set VLM_BACKEND=qwen for the real model"
        verdicts = {
            ("passed", True): f" The uploaded chart MATCHES the governed warehouse figures ({how}).",
            ("failed", True): f" WARNING: the uploaded chart does NOT match the governed warehouse figures ({how}); see evidence.",
            ("passed", False): f" Chart read-back check passed ({how}).",
            ("failed", False): f" Note: the chart read-back did not match the data ({how}); treat the chart with care.",
        }
        text += verdicts.get((status, upload), f" The chart could not be read reliably, so it was not verified ({how}).")
        return text


def _lower(label: str) -> str:
    return label if label[:2].isupper() else label[0].lower() + label[1:]


def _act(thought: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"thought": thought, "tool": tool, "args": args}


def _no_metric(chart_obs: dict[str, Any] | None) -> str:
    if chart_obs and chart_obs.get("answer"):
        return f"From the chart: {chart_obs['answer']}. (No governed metric matched, so it was not cross-checked.)"
    return "No governed metric matches this question. Try one of the metrics listed at GET /metrics."


def _month_end(y: int, m: int) -> date:
    return (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
