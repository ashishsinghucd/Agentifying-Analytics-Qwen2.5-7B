"""Runtime settings, read from environment variables (12-factor style).

Every value can be overridden per environment (dev / test / prod) without
code changes, which keeps CI and deployment configuration out of the code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class Settings:
    warehouse_path: str
    metrics_path: str
    runs_dir: str
    # Orchestrator LLM: any OpenAI-compatible endpoint (Ollama, vLLM, Azure OpenAI...).
    # If unset, the deterministic rule-based planner is used (CI, offline demos).
    llm_base_url: str | None
    llm_model: str
    llm_api_key: str
    # Chart reader: "qwen" loads the fine-tuned Qwen2.5-VL ChartQA LoRA, "mock" is for CI.
    vlm_backend: str
    vlm_base_model: str
    vlm_adapter: str
    max_steps: int
    verify_tolerance: float
    max_rows: int

    @classmethod
    def from_env(cls) -> Settings:
        env = os.getenv
        return cls(
            warehouse_path=env("AGENT_WAREHOUSE", str(PACKAGE_ROOT / "data" / "warehouse.duckdb")),
            metrics_path=env("AGENT_METRICS", str(PACKAGE_ROOT / "semantic" / "metrics.yml")),
            runs_dir=env("AGENT_RUNS_DIR", str(REPO_ROOT / "outputs" / "agent_runs")),
            llm_base_url=env("LLM_BASE_URL") or None,
            llm_model=env("LLM_MODEL", "qwen2.5:7b-instruct"),
            llm_api_key=env("LLM_API_KEY", "not-needed"),
            vlm_backend=env("VLM_BACKEND", "mock"),
            vlm_base_model=env("VLM_BASE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            vlm_adapter=env("VLM_ADAPTER", "prakashchhipa/Qwen2.5-VL-7B-ChartQA-LoRA"),
            max_steps=int(env("AGENT_MAX_STEPS", "8")),
            verify_tolerance=float(env("AGENT_VERIFY_TOL", "0.05")),
            max_rows=int(env("AGENT_MAX_ROWS", "5000")),
        )
