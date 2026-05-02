"""
Browser automation tools.

Uses Playwright for arbitrary web page interaction — separate from
the LLM browser automation (chatgpt/gemini pages).

Lazy-init: browser starts only on first tool call.
"""

import os
import tempfile

from browser_llm_agent.tools.registry import tool

# Lazy-initialized browser state
_BROWSER = None
_PAGE = None
_PW = None


def _get_page():
    """Get or create the automation browser page."""
    global _BROWSER, _PAGE, _PW
    if _PAGE is not None and not _PAGE.is_closed():
        return _PAGE

    from playwright.sync_api import sync_playwright
    if _PW is None:
        _PW = sync_playwright().start()
    if _BROWSER is None:
        _BROWSER = _PW.chromium.launch(headless=True)
    _PAGE = _BROWSER.new_page()
    return _PAGE


def _close_browser():
    """Cleanup browser resources."""
    global _BROWSER, _PAGE, _PW
    if _PAGE:
        try:
            _PAGE.close()
        except Exception:
            pass
        _PAGE = None
    if _BROWSER:
        try:
            _BROWSER.close()
        except Exception:
            pass
        _BROWSER = None
    if _PW:
        try:
            _PW.stop()
        except Exception:
            pass
        _PW = None


@tool("browser_navigate", "Navigate to a URL and return the page title + text content.", {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "URL to navigate to"},
    },
    "required": ["url"],
})
def browser_navigate(url: str) -> str:
    try:
        page = _get_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        title = page.title()
        text = page.inner_text("body")
        # Truncate long pages
        if len(text) > 5000:
            text = text[:5000] + f"\n\n... (truncated, {len(text)} total chars)"
        return f"Title: {title}\n\n{text}"
    except Exception as e:
        return f"Error navigating to {url}: {e}"


@tool("browser_screenshot", "Take a screenshot of the current page. Returns the file path.", {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "URL to navigate to first (optional — uses current page if omitted)"},
        "full_page": {"type": "boolean", "description": "Capture full scrollable page (default: false)"},
    },
    "required": [],
})
def browser_screenshot(url: str = None, full_page: bool = False) -> str:
    try:
        page = _get_page()
        if url:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)

        path = os.path.join(tempfile.gettempdir(), "browser_screenshot.png")
        page.screenshot(path=path, full_page=full_page)
        return f"Screenshot saved: {path}"
    except Exception as e:
        return f"Error taking screenshot: {e}"


@tool("browser_click", "Click an element on the current page by CSS selector.", {
    "type": "object",
    "properties": {
        "selector": {"type": "string", "description": "CSS selector of the element to click"},
    },
    "required": ["selector"],
})
def browser_click(selector: str) -> str:
    try:
        page = _get_page()
        page.click(selector, timeout=10000)
        return f"Clicked: {selector}"
    except Exception as e:
        return f"Error clicking '{selector}': {e}"


@tool("browser_fill", "Fill a form field on the current page.", {
    "type": "object",
    "properties": {
        "selector": {"type": "string", "description": "CSS selector of the input field"},
        "value": {"type": "string", "description": "Value to fill"},
    },
    "required": ["selector", "value"],
})
def browser_fill(selector: str, value: str) -> str:
    try:
        page = _get_page()
        page.fill(selector, value, timeout=10000)
        return f"Filled '{selector}' with value"
    except Exception as e:
        return f"Error filling '{selector}': {e}"


@tool("browser_get_text", "Get text content of an element on the current page.", {
    "type": "object",
    "properties": {
        "selector": {"type": "string", "description": "CSS selector (default: body)"},
    },
    "required": [],
})
def browser_get_text(selector: str = "body") -> str:
    try:
        page = _get_page()
        text = page.inner_text(selector, timeout=10000)
        if len(text) > 5000:
            text = text[:5000] + f"\n\n... (truncated, {len(text)} total chars)"
        return text
    except Exception as e:
        return f"Error getting text from '{selector}': {e}"


@tool("browser_eval", "Execute JavaScript in the current page and return the result.", {
    "type": "object",
    "properties": {
        "js": {"type": "string", "description": "JavaScript code to evaluate"},
    },
    "required": ["js"],
})
def browser_eval(js: str) -> str:
    try:
        page = _get_page()
        result = page.evaluate(js)
        return str(result)
    except Exception as e:
        return f"Error evaluating JS: {e}"
