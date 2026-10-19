"""Accessibility and layout gate for the built documentation site.

Usage:
    mkdocs build -d /tmp/site
    python -m http.server 8799 -d /tmp/site &
    python scripts/devtools/site_qa.py http://127.0.0.1:8799 [page.html ...]

Needs Playwright with an installed Chrome (``uv run --with playwright ...``).
For every page, in light and dark colour schemes, it reports page-level
horizontal overflow at 320 and 390 px, console errors, and axe-core
WCAG 2.2 A/AA violations. Exits non-zero if anything is found. axe is a
gate, not a certification: keyboard use and charts still need manual checks.
"""

import sys
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

AXE = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"
TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]
IGNORED_CONSOLE = ("api.github.com",)  # Material's repo widget is rate-limited offline


def discover(base):
    import re
    from urllib.request import urlopen

    xml = urlopen(urljoin(base + "/", "sitemap.xml")).read().decode()
    locs = re.findall(r"<loc>([^<]+)</loc>", xml)
    return sorted({l.split("trustcv/", 1)[-1] or "index.html" for l in locs})


def main(base, names):
    problems = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        if not names:
            names = discover(base)
        for name in names:
            url = urljoin(base + "/", name)
            for width in (320, 390):
                pg = browser.new_page(viewport={"width": width, "height": 800})
                errors = []
                pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and not any(s in (m.location or {}).get("url", "") for s in IGNORED_CONSOLE) else None)
                pg.on("pageerror", lambda e: errors.append(str(e)))
                pg.goto(url, wait_until="networkidle")
                over = pg.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
                pg.close()
                if over > 0 or errors:
                    problems += 1
                    print(f"FAIL {name} @{width}px overflow={over} console={errors[:2]}")
            for scheme in ("default", "slate"):
                pg = browser.new_page(viewport={"width": 1280, "height": 900})
                pg.goto(url, wait_until="networkidle")
                pg.evaluate(f"document.body.setAttribute('data-md-color-scheme', '{scheme}')")
                pg.wait_for_timeout(300)
                pg.add_script_tag(url=AXE)
                found = pg.evaluate(
                    "async (tags) => (await axe.run(document, {runOnly: tags})).violations"
                    ".map(v => ({id: v.id, n: v.nodes.length, ex: v.nodes[0].failureSummary.split('\\n')[1] || '', t: v.nodes[0].target.join(' ')}))",
                    TAGS,
                )
                pg.close()
                for v in found:
                    problems += 1
                    print(f"FAIL {name} [{scheme}] {v['id']} x{v['n']} at {v['t'][:60]} | {v['ex'][:110]}")
        browser.close()
    print("OK: no problems found" if not problems else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1].rstrip("/"), sys.argv[2:]))
