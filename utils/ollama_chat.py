from __future__ import annotations

import requests

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:1b"


def get_model_name() -> str:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        response.raise_for_status()
        models = [item["name"] for item in response.json().get("models", [])]
        for preferred in (DEFAULT_MODEL, "llama3.2", "llama3.1:8b", "llama3"):
            if preferred in models:
                return preferred
        return models[0] if models else DEFAULT_MODEL
    except requests.RequestException:
        return DEFAULT_MODEL


def ask_ollama(prompt: str, context: str = "", history: list[dict] | None = None) -> str:
    model = get_model_name()
    system = (
        "You are MediMind AI, a careful medical education assistant. "
        "Give clear, non-diagnostic guidance, mention red flags, and advise professional care when appropriate."
    )
    messages = [{"role": "system", "content": system}]
    if context:
        messages.append({"role": "system", "content": f"Use this retrieved medical context when useful:\n{context}"})
    for item in history or []:
        if item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": item["message"]})
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=45,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()
    except requests.RequestException:
        return (
            "I could not reach Ollama right now. Demo guidance: monitor symptoms, hydrate, rest, "
            "and seek urgent care for chest pain, trouble breathing, confusion, severe weakness, or symptoms that worsen quickly."
        )
