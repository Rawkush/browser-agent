import time
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout

DEEPSEEK_URL = "https://chat.deepseek.com"

SELECTORS = {
    "input": "#chat-input",
    "send_btn": ".ds-icon--send",
    "response": ".ds-markdown.ds-markdown--block",
    "stop_btn": ".ds-icon--stop",
}

def open_deepseek(browser):
    page = browser.new_page()
    page.goto(DEEPSEEK_URL, wait_until="domcontentloaded")
    print("[DeepSeek] Opened. Please log in if prompted, then press Enter...")
    input()
    return page

def send_message(page: Page, message: str) -> str:
    textarea = page.locator(SELECTORS["input"])
    for attempt in range(3):
        try:
            textarea.wait_for(state="visible", timeout=15000)
            textarea.click()
            textarea.fill(message)
            break
        except PlaywrightTimeout:
            if attempt == 2: raise
            time.sleep(3)

    page.keyboard.press("Enter")
    time.sleep(1.5)

    try:
        # Wait for generation to start and then finish
        page.locator(SELECTORS["stop_btn"]).wait_for(state="visible", timeout=8000)
        page.locator(SELECTORS["stop_btn"]).wait_for(state="hidden", timeout=180000)
    except PlaywrightTimeout:
        pass

    time.sleep(0.5)
    messages = page.locator(SELECTORS["response"]).all()
    return messages[-1].inner_text().strip() if messages else ""

def new_conversation(page: Page):
    page.goto(DEEPSEEK_URL, wait_until="domcontentloaded")
    time.sleep(1)