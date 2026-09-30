"""Capture docs/images/ui_ask.png from the running Gradio UI (needs `pip install playwright`
and `playwright install chromium`).

  python -m agentic.ui &                         # serves http://localhost:7861
  python scripts/capture_ui_screenshot.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "ui_ask.png"
QUESTION = "Which site had the highest energy intensity last quarter?"


async def main(url: str = "http://localhost:7861/") -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1100})
        await page.goto(url, wait_until="networkidle")
        await page.get_by_label("Question").fill(QUESTION)
        await page.get_by_role("button", name="Ask").click()
        await page.wait_for_selector("text=Chart read-back", timeout=60_000)
        await page.wait_for_timeout(2000)
        await page.get_by_text("Governed SQL", exact=True).click()
        await page.wait_for_timeout(1000)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(OUT), full_page=True)
        await browser.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
