"""Open the static experiment in Chromium using its exported local files.

Requests are fulfilled directly from the repository, so the browser checks and the README picture
need neither a separate HTTP server nor network access. Install requirements-page.txt
and run `playwright install chromium` before using these scripts.
"""

from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "http://flowgnn.test"


def open_page(browser, width=1366, height=900, missing=None):
    """Load the page with its recorded data; optionally omit one resource to test the error message."""
    page = browser.new_page(viewport={"width": width, "height": height})

    page.errors = []
    page.on("pageerror", lambda error: page.errors.append(str(error)))
    page.on("console", lambda message: page.errors.append(message.text) if message.type == "error" else None)

    def serve(route):
        """Serve only repository files, preserving missing-file errors."""
        relative = unquote(urlsplit(route.request.url).path).lstrip("/")
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or relative == missing or not path.is_file():
            route.fulfill(status=404, body="Not found")
        else:
            route.fulfill(path=path)

    page.set_default_timeout(15000)
    page.route(f"{ORIGIN}/**", serve)
    page.goto(f"{ORIGIN}/docs/index.html")
    if missing is None:
        page.wait_for_function("window.flowgnnReady === true")
        page.evaluate("document.fonts.ready")
    return page


def set_slider(page, selector, value):
    """Change a real page control and dispatch its input event."""
    page.locator(selector).evaluate(
        "(element, value) => {element.value = value; element.dispatchEvent(new Event('input', {bubbles:true}));}",
        str(value),
    )
