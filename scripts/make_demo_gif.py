"""Record docs/images/demo.gif: a captioned walkthrough of the real web UI.

  pip install gradio playwright pillow && playwright install chromium
  python -m agentic.data.generate_synthetic
  python scripts/make_readme_assets.py        # reconciliation.png is reused as one scene
  python scripts/make_demo_gif.py

The script starts the Gradio UI itself, drives it with a headless browser, captures
frames at each step, then adds captions and highlight boxes.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("AGENT_RUNS_DIR", tempfile.mkdtemp(prefix="agent_runs_"))
os.environ.setdefault("VLM_BACKEND", "mock")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from playwright.async_api import Page, async_playwright  # noqa: E402

OUT = REPO / "docs" / "images" / "demo.gif"
RECON = REPO / "docs" / "images" / "reconciliation.png"
PORT = 7862
VIEW_W, VIEW_H, BAND = 1280, 1060, 84
OUT_W = 960
DARK, LIGHT, ACCENT, MUTED = (21, 25, 30), (244, 242, 237), (249, 115, 22), (167, 176, 186)
Q1 = "Which site had the highest energy intensity last quarter?"
Q2 = "Monthly pellet production by site in 2025"


@dataclass
class Frame:
    image: Image.Image
    ms: int
    step: int | None = None
    caption: str = ""
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"] if bold
             else ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"])
    dirs = ["/usr/share/fonts/truetype/dejavu", "/Library/Fonts", "/System/Library/Fonts/Supplemental",
            "C:/Windows/Fonts", ""]
    for d in dirs:
        for n in names:
            try:
                return ImageFont.truetype(os.path.join(d, n) if d else n, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


# ------------------------------------------------------------------ capture
async def shot(page: Page) -> Image.Image:
    return Image.open(io.BytesIO(await page.screenshot())).convert("RGB")


async def box(page: Page, selector: str) -> tuple[float, float, float, float]:
    b = await page.locator(selector).first.bounding_box()
    return (b["x"], b["y"], b["width"], b["height"]) if b else (0, 0, 0, 0)


async def type_question(page: Page, text: str, frames: list[Frame], step: int, caption: str, n: int) -> None:
    field_ = page.locator("#question textarea")
    for i in range(1, n + 1):
        await field_.fill(text[: round(len(text) * i / n)])
        frames.append(Frame(await shot(page), 110 if i < n else 900, step, caption, [await box(page, "#question")]))


async def record() -> list[Frame]:
    from agentic.ui import build_ui

    demo = build_ui()
    demo.launch(server_port=PORT, prevent_thread_lock=True, quiet=True)
    frames: list[Frame] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={"width": VIEW_W, "height": VIEW_H})
            await page.goto(f"http://localhost:{PORT}/", wait_until="networkidle")
            await page.wait_for_timeout(800)

            cap1 = "Ask a business question in plain language"
            frames.append(Frame(await shot(page), 1200, 1, cap1))
            await type_question(page, Q1, frames, 1, cap1, n=12)
            frames.append(Frame(await shot(page), 700, 1, cap1, [await box(page, "#ask")]))
            await page.locator("#ask").click()
            await page.wait_for_selector("#answer >> text=Chart read-back", timeout=60_000)
            await page.wait_for_timeout(1500)

            img = await shot(page)
            frames.append(Frame(img, 3800, 2, "Get a governed answer and the chart the agent drew",
                                [await box(page, "#answer"), await box(page, "#chart")]))
            frames.append(Frame(img, 3800, 3, "The chart is read back and checked against the data",
                                [await box(page, "#verification")]))
            await page.get_by_text("Governed SQL", exact=True).click()
            await page.wait_for_timeout(900)
            await page.evaluate("document.querySelector('#sql').scrollIntoView({block: 'center'})")
            await page.wait_for_timeout(500)
            frames.append(Frame(await shot(page), 3800, 4, "See the exact SQL, compiled from the semantic model",
                                [await box(page, "#sql")]))

            cap5 = "Trends work the same way"
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(400)
            await type_question(page, Q2, frames, 5, cap5, n=6)
            await page.locator("#ask").click()
            await page.wait_for_selector("#answer >> text=per month", timeout=60_000)
            await page.wait_for_timeout(1500)
            frames.append(Frame(await shot(page), 3500, 5, cap5,
                                [await box(page, "#answer"), await box(page, "#chart")]))
            await browser.close()
    finally:
        demo.close()
    return frames


# ------------------------------------------------------------------ compose
def card(title: str, lines: list[str], note: str) -> Image.Image:
    im = Image.new("RGB", (VIEW_W, VIEW_H + BAND), DARK)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, VIEW_W, 10], fill=ACCENT)
    y = 360
    d.text((120, y), title, font=font(64, True), fill=LIGHT)
    y += 120
    for line in lines:
        d.text((120, y), line, font=font(36), fill=MUTED if line.startswith("$") else LIGHT)
        y += 62
    d.text((120, VIEW_H + BAND - 110), note, font=font(24), fill=MUTED)
    return im


def compose(f: Frame) -> Image.Image:
    base = f.image
    if f.boxes:  # dim everything except the highlighted regions
        shade = Image.new("L", base.size, 110)
        sd = ImageDraw.Draw(shade)
        for x, y, w, h in f.boxes:
            sd.rounded_rectangle([x - 8, y - 8, x + w + 8, y + h + 8], radius=12, fill=0)
        base = Image.composite(Image.new("RGB", base.size, (0, 0, 0)), base, shade)
        d = ImageDraw.Draw(base)
        for x, y, w, h in f.boxes:
            d.rounded_rectangle([x - 8, y - 8, x + w + 8, y + h + 8], radius=12, outline=ACCENT, width=5)
    im = Image.new("RGB", (VIEW_W, VIEW_H + BAND), DARK)
    im.paste(base, (0, BAND))
    d = ImageDraw.Draw(im)
    if f.step is not None:
        d.ellipse([28, 18, 76, 66], fill=ACCENT)
        n = str(f.step)
        tw = d.textlength(n, font=font(30, True))
        d.text((52 - tw / 2, 24), n, font=font(30, True), fill=DARK)
    d.text((96, 22), f.caption, font=font(34, True), fill=LIGHT)
    return im


def reconciliation_frame() -> Image.Image:
    shot_ = Image.open(RECON).convert("RGB")
    scale = min(VIEW_W / shot_.width, VIEW_H / shot_.height)
    shot_ = shot_.resize((int(shot_.width * scale), int(shot_.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (VIEW_W, VIEW_H), (244, 242, 237))
    left, top = (VIEW_W - shot_.width) // 2, (VIEW_H - shot_.height) // 2
    canvas.paste(shot_, (left, top))
    # highlight the verdict block (bottom quarter of reconciliation.png, see make_readme_assets.py)
    verdict = (left + 0.01 * shot_.width, top + 0.75 * shot_.height, 0.66 * shot_.width, 0.20 * shot_.height)
    return compose(Frame(canvas, 0, 6, "Upload an old dashboard: mismatches are flagged", [verdict]))


def main() -> None:
    frames = asyncio.run(record())
    images = [card("AskAnything Analytics Agent",
                   ["Governed answers and verified charts,",
                    "built on a fine-tuned ChartQA model"],
                   "Real UI run on synthetic data. Chart read-back uses the CI mock reader.")]
    durations = [2600]
    for f in frames:
        images.append(compose(f))
        durations.append(f.ms)
    if RECON.exists():
        images.append(reconciliation_frame())
        durations.append(5000)
    images.append(card("Try it", ["$ pip install -r requirements-agentic.txt gradio",
                                  "$ python -m agentic.data.generate_synthetic",
                                  "$ python -m agentic.ui"],
                       "Set VLM_BACKEND=qwen to read charts with the fine-tuned Qwen2.5-VL-7B LoRA."))
    durations.append(3500)

    h = round(images[0].height * OUT_W / images[0].width)
    small = [im.resize((OUT_W, h), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=128)
             for im in images]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    small[0].save(OUT, save_all=True, append_images=small[1:], duration=durations, loop=0, optimize=True)
    print(f"Saved {OUT} ({len(small)} frames, {OUT.stat().st_size / 1e6:.1f} MB, {sum(durations) / 1000:.0f}s)")


if __name__ == "__main__":
    main()
