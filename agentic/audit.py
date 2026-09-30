"""Append-only JSONL trace of every agent step: who asked what, which tool ran with
which arguments, how long it took and what it returned. One folder per run also
holds the charts, so any answer can be reproduced and reviewed."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunTrace:
    def __init__(self, runs_dir: str | Path, question: str, user: str = "anonymous"):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.dir = Path(runs_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "trace.jsonl"
        self.steps: list[dict[str, Any]] = []
        self.log("run_started", {"question": question, "user": user})

    def log(self, event: str, payload: dict[str, Any], duration_ms: float | None = None) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
                  "event": event, "duration_ms": duration_ms, **payload}
        self.steps.append(record)
        with self.file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


class Timer:
    def __enter__(self) -> Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = round((time.perf_counter() - self.t0) * 1000, 1)
