"""HTTP API so dashboards, Teams bots or other agents can use the analytics agent.

Run:  uvicorn agentic.api:app --port 8000
"""
from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from agentic.agent.orchestrator import AnalyticsAgent

app = FastAPI(title="AskAnything Analytics Agent", version="0.1.0")
ALLOWED_IMAGE_TYPES = {"image/png": ".png", "image/jpeg": ".jpg"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


@lru_cache(maxsize=1)
def get_agent() -> AnalyticsAgent:
    return AnalyticsAgent()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    user: str = "anonymous"


@app.get("/health")
def health() -> dict:
    a = get_agent()
    return {"status": "ok", "planner": a.planner.name, "chart_reader": a.reader.name,
            "data_as_of": a.as_of.isoformat()}


@app.get("/metrics")
def metrics() -> list[dict]:
    return get_agent().semantic.catalog()


@app.get("/metrics/dax", response_class=PlainTextResponse)
def metrics_dax() -> str:
    """Same metric definitions, exported as Power BI DAX measures."""
    return get_agent().semantic.to_dax()


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return get_agent().ask(req.question, user=req.user).to_dict()


@app.post("/ask-with-chart")
async def ask_with_chart(question: str = Form(...), image: UploadFile = File(...)) -> dict:
    """Upload a dashboard screenshot or report figure: the agent reads it with the ChartQA
    model and reconciles it against the governed warehouse numbers."""
    suffix = ALLOWED_IMAGE_TYPES.get(image.content_type or "")
    if not suffix:
        raise HTTPException(415, "Only PNG or JPEG images are accepted")
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
    try:
        return get_agent().ask(question, image_path=tmp.name).to_dict()
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.get("/runs/{run_id}/{filename}")
def run_file(run_id: str, filename: str) -> FileResponse:
    base = Path(get_agent().s.runs_dir).resolve()
    path = (base / run_id / filename).resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path)
