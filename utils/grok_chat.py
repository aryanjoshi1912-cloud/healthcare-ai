from __future__ import annotations

import os
import requests
import streamlit as st

XAI_API_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-beta"


def get_api_key() -> str | None:
    # 1. Try to get it from Streamlit Secrets (for Streamlit Cloud deployment)
    try:
        if "XAI_API_KEY" in st.secrets:
            return st.secrets["XAI_API_KEY"]
    except Exception:
        pass

    # 2. Try to get it from environment variables (local or other deployments)
    return os.getenv("XAI_API_KEY")


def get_model_name() -> str:
    # We can default to the stable grok-beta model, or let users customize it
    try:
        if "XAI_MODEL" in st.secrets:
            return st.secrets["XAI_MODEL"]
    except Exception:
        pass
    return os.getenv("XAI_MODEL", DEFAULT_MODEL)


def ask_grok(prompt: str, context: str = "", history: list[dict] | None = None) -> str:
    api_key = get_api_key()
    if not api_key:
        return (
            "Grok API Key (XAI_API_KEY) is missing. "
            "Please configure the XAI_API_KEY in your Streamlit secrets or environment variables."
        )

    model = get_model_name()
    system_prompt = (
        "You are MediMind AI, a careful medical education assistant. "
        "Give clear, non-diagnostic guidance, mention red flags, and advise professional care when appropriate."
    )
    if context:
        system_prompt += f"\n\nUse this retrieved medical context when useful:\n{context}"

    messages = [{"role": "system", "content": system_prompt}]
    
    for item in history or []:
        if item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": item["message"]})
            
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "stream": False,
    }

    try:
        response = requests.post(
            f"{XAI_API_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        detail = ""
        available_models_info = ""
        if e.response is not None:
            detail = f" | Details: {e.response.text}"
            if "Model not found" in e.response.text or "invalid-argument" in e.response.text:
                try:
                    models_response = requests.get(
                        f"{XAI_API_URL}/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10
                    )
                    if models_response.status_code == 200:
                        model_names = [m["id"] for m in models_response.json().get("data", [])]
                        available_models_info = f" | Available models for your key: {', '.join(model_names)}"
                    else:
                        available_models_info = f" | Failed listing models (HTTP {models_response.status_code}): {models_response.text}"
                except Exception as ex:
                    available_models_info = f" | Failed listing models error: {ex}"
        return (
            f"Error communicating with Grok API: {e}{detail}{available_models_info}. "
            "Demo guidance: monitor symptoms, hydrate, rest, "
            "and seek urgent care for chest pain, trouble breathing, confusion, severe weakness, or symptoms that worsen quickly."
        )



