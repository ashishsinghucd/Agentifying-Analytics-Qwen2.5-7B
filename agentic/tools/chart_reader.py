"""Chart reading tool backed by the fine-tuned Qwen2.5-VL-7B ChartQA LoRA from this repo.

The agent uses it for two jobs:
  * reading charts it did not create (dashboard screenshots, PDF report figures);
  * reading back charts it DID create, to verify the picture matches the data.

`MockChartReader` answers from the JSON sidecar written by charting.py so the whole
pipeline runs in CI without a GPU. It is never used when VLM_BACKEND=qwen.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from agentic.tools.charting import compact


class ChartReader(Protocol):
    name: str

    def ask(self, image_path: str, question: str) -> str: ...


class QwenChartReader:
    """Loads base Qwen2.5-VL-7B + the ChartQA LoRA adapter (merged for faster inference)."""

    name = "qwen2.5-vl-7b-chartqa-lora"

    def __init__(self, base_model: str, adapter: str, device_map: str = "auto"):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._torch = torch
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map=device_map)
        self.model = PeftModel.from_pretrained(base, adapter).merge_and_unload().eval()
        try:
            self.processor = AutoProcessor.from_pretrained(adapter)
        except OSError:
            self.processor = AutoProcessor.from_pretrained(base_model)

    def ask(self, image_path: str, question: str, max_new_tokens: int = 48) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
            {"type": "text", "text": question},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True,
                                return_tensors="pt").to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


class MockChartReader:
    """Deterministic stand-in for tests. Understands the two verification questions."""

    name = "mock"

    def ask(self, image_path: str, question: str) -> str:
        sidecar = Path(image_path).with_suffix(".json")
        if not sidecar.exists():
            return "unknown"
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        flat: dict[str, float] = {}
        for k, v in meta["data"].items():
            if isinstance(v, dict):  # line chart: series -> {x: y}
                flat[k] = max(v.values())
            else:
                flat[k] = v
        best = max(flat, key=flat.get)
        q = question.lower()
        if q.startswith("which") or "label" in q:
            return best
        return compact(flat[best])


def build_reader(backend: str, base_model: str, adapter: str) -> ChartReader:
    if backend == "qwen":
        return QwenChartReader(base_model, adapter)
    if backend == "mock":
        return MockChartReader()
    raise ValueError(f"Unknown VLM backend '{backend}'")


_NUM = re.compile(r"(-?\d[\d,\s]*\.?\d*)\s*([kKmM]?)")


def parse_number(text: str) -> float | None:
    """'1.23M' -> 1230000.0, '45,300 t' -> 45300.0, 'about 3.5k' -> 3500.0."""
    m = _NUM.search(text or "")
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", "").replace(" ", ""))
    except ValueError:
        return None
    return value * {"k": 1e3, "m": 1e6}.get(m.group(2).lower(), 1.0)
