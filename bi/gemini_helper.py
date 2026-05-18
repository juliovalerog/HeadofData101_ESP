from __future__ import annotations

import os

try:
    from google import genai
except ImportError:
    genai = None


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def resolved_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def gemini_available() -> bool:
    return genai is not None and bool(resolved_gemini_api_key())


def gemini_unavailable_message() -> str | None:
    if genai is None:
        return "Gemini is unavailable because the `google-genai` package is not installed."
    if not resolved_gemini_api_key():
        return (
            "Gemini is optional and no API key is configured. Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` "
            "to enable AI output."
        )
    return None


def generate_gemini_content(prompt: str) -> tuple[str | None, str | None]:
    unavailable_message = gemini_unavailable_message()
    if unavailable_message:
        return None, unavailable_message

    try:
        client = genai.Client(api_key=resolved_gemini_api_key())
        response = client.models.generate_content(model=DEFAULT_GEMINI_MODEL, contents=prompt)
    except Exception as exc:
        message = str(exc).lower()
        if "api key" in message or "unauthenticated" in message or "invalid" in message:
            return None, "Gemini rejected the configured API key. Check `GEMINI_API_KEY` or `GOOGLE_API_KEY`."
        if "permission" in message or "denied" in message or "forbidden" in message:
            return None, "Gemini permissions were denied for the configured API key or project."
        return None, "Gemini request failed. Check network access, API availability, and the configured key."

    text = getattr(response, "text", None)
    if not text:
        return None, "Gemini returned an empty response."
    return str(text), None
