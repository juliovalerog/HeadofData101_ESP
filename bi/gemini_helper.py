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
        return "Gemini no está disponible porque el paquete `google-genai` no está instalado."
    if not resolved_gemini_api_key():
        return (
            "Gemini es opcional y no hay ninguna clave API configurada. Establecer `GEMINI_API_KEY` o `GOOGLE_API_KEY` "
            "para habilitar la salida de IA."
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
        if "clave API" in message or "unauthenticated" in message or "invalid" in message:
            return None, "Gemini rechazó la clave API configurada. Marque `GEMINI_API_KEY` o `GOOGLE_API_KEY`."
        if "permission" in message or "denied" in message or "forbidden" in message:
            return None, "Se denegaron los permisos Gemini para la clave o proyecto API configurado."
        return None, "Gemini falló la solicitud. Verifique el acceso a la red, la disponibilidad de API y la clave configurada."

    text = getattr(response, "text", None)
    if not text:
        return None, "Gemini devolvió una respuesta vacía."
    return str(text), None
