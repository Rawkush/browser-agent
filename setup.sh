#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installing Playwright browsers..."
playwright install chromium

echo ""
echo "Setup complete. Run with:"
echo "  python agent.py --llm chatgpt"
echo "  python agent.py --llm gemini"
