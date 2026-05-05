import ollama

def send_message(model_name: str = "qwen2.5-coder:14b", message: str = "") -> str:
    """Simple wrapper for Ollama local API."""
    try:
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': message},
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error connecting to Ollama: {e}. Make sure Ollama is running (`ollama serve`)."