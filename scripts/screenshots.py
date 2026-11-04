"""Capture README screenshots from a running LitScout instance.

    LITSCOUT_DB_PATH=/tmp/shots.db litscout seed-demo
    LITSCOUT_DB_PATH=/tmp/shots.db LITSCOUT_SOURCES_ENABLED=openalex,arxiv,crossref,demo litscout serve --port 8765 &
    python scripts/screenshots.py http://127.0.0.1:8765

Requires `pip install playwright && playwright install chromium` (or set CHROME_PATH).
"""

import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path=os.environ.get("CHROME_PATH") or None)
    page = browser.new_page(viewport={"width": 1400, "height": 900})

    def shot(name: str, full: bool = False) -> None:
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
        print("saved", name)

    page.goto(f"{BASE}/")
    shot("01-projects")
    page.goto(f"{BASE}/projects/1/review")
    shot("02-review", full=True)
    page.goto(f"{BASE}/projects/1")
    page.fill("input[name=query]", "adversarial robustness of injection detectors")
    shot("03-search")

    # A fresh project to show screening in progress
    page.goto(f"{BASE}/")
    page.fill("input[name=title]", "Computer-based testing in schools (demo)")
    page.fill("textarea[name=question]", "What factors affect the adoption and integrity of computer-based testing in secondary schools?")
    page.click("text=Create project")
    page.fill("input[name=query]", "computer-based testing examinations schools")
    page.uncheck("input[value=openalex]")
    page.uncheck("input[value=arxiv]")
    page.uncheck("input[value=crossref]")
    page.check("input[value=demo]")
    page.click("button:has-text('Search')")
    page.wait_for_selector(".paper")
    shot("04-screening")
    page.goto(f"{BASE}/projects/1/screen?status=all")
    shot("05-screening-decisions")

    mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True)
    mobile.goto(f"{BASE}/projects/1/review")
    mobile.screenshot(path=str(OUT / "06-mobile-review.png"))
    print("saved 06-mobile-review")
    browser.close()
