"""Render the opening comparison (solver | network) used as the README picture.

Run python scripts/make_cover.py after installing requirements-page.txt and Chromium.
The output uses the same interpolation, bounds and representative model as the page.
"""
from playwright.sync_api import sync_playwright
from page_browser import ROOT, open_page


def main():
    """Capture the loaded solver and prediction comparison without training or changing data."""
    with sync_playwright() as engine:
        browser = engine.chromium.launch()
        try:
            page = open_page(browser, 1920, 1080)
            page.locator('#divider .hint').evaluate('e => (e.hidden = true)')  # an instruction, not part of the picture
            page.locator('#compare').screenshot(path=ROOT / 'media/prediction.png')
            assert not page.errors, page.errors
        finally:
            browser.close()


if __name__ == '__main__':
    main()
