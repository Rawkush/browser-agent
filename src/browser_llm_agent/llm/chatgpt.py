import time
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout


CHATGPT_URL = "https://chat.openai.com"

SELECTORS = {
    "input": "#prompt-textarea",
    "send_btn": '[data-testid="send-button"]',
    "response": '[data-message-author-role="assistant"]',
    "stop_btn": '[data-testid="stop-button"]',
}


def open_chatgpt(browser):
    page = browser.new_page()
    page.goto(CHATGPT_URL, wait_until="domcontentloaded")
    print("[ChatGPT] Opened. Please log in if prompted, then press Enter...")
    input()
    return page


def send_message(page: Page, message: str) -> str:
    # type message
    textarea = page.locator(SELECTORS["input"])
    textarea.wait_for(state="visible", timeout=15000)
    textarea.click()
    textarea.fill(message)

    # send
    page.keyboard.press("Enter")

    # wait for generation to start
    time.sleep(1.5)

    # wait for stop button to disappear (generation complete)
    try:
        page.locator(SELECTORS["stop_btn"]).wait_for(state="visible", timeout=8000)
        page.locator(SELECTORS["stop_btn"]).wait_for(state="hidden", timeout=120000)
    except PlaywrightTimeout:
        pass  # already done or didn't appear

    time.sleep(0.5)

    # extract last assistant message
    messages = page.locator(SELECTORS["response"]).all()
    if not messages:
        return ""
    return messages[-1].inner_text().strip()


def new_conversation(page: Page):
    page.goto(CHATGPT_URL, wait_until="domcontentloaded")
    time.sleep(1)
