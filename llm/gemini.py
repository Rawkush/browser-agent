import time
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout


GEMINI_URL = "https://gemini.google.com"

SELECTORS = {
    "input": "rich-textarea .ql-editor",
    "send_btn": "button.send-button",
    "response": "model-response .markdown",
    "loading": ".loading-indicator",
}


def open_gemini(browser):
    page = browser.new_page()
    page.goto(GEMINI_URL, wait_until="domcontentloaded")
    print("[Gemini] Opened. Please log in if prompted, then press Enter...")
    input()
    return page


def send_message(page: Page, message: str) -> str:
    # type message — retry up to 3× in case the editor is temporarily non-editable
    # (Gemini locks the input while a response is streaming)
    editor = page.locator(SELECTORS["input"])
    for attempt in range(3):
        try:
            editor.wait_for(state="visible", timeout=15000)
            editor.click()
            editor.fill(message)
            break
        except PlaywrightTimeout:
            if attempt == 2:
                raise
            time.sleep(3)

    # send
    page.keyboard.press("Enter")

    time.sleep(2)

    # wait for response to settle — poll last response until text stops changing
    last_text = ""
    stable_count = 0
    for _ in range(120):
        time.sleep(1)
        responses = page.locator(SELECTORS["response"]).all()
        if not responses:
            continue
        current_text = responses[-1].inner_text().strip()
        if current_text == last_text and current_text:
            stable_count += 1
            if stable_count >= 2:
                break
        else:
            stable_count = 0
        last_text = current_text

    responses = page.locator(SELECTORS["response"]).all()
    if not responses:
        return ""
    return responses[-1].inner_text().strip()


def new_conversation(page: Page):
    page.goto(GEMINI_URL, wait_until="domcontentloaded")
    time.sleep(1)
