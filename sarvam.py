"""
Sarvam AI integration for FLASHPOINT
- translate_to_hindi(): English → Hindi via Mayura v1
- text_to_speech(): Hindi TTS via Bulbul v3 → returns MP3 bytes
"""
import base64
import os

import httpx

SARVAM_BASE = "https://api.sarvam.ai"


def _headers() -> dict:
    """Read key at call time so Railway env vars are always picked up."""
    return {
        "api-subscription-key": os.getenv("SARVAM_API_KEY", ""),
        "Content-Type": "application/json",
    }


def translate_to_hindi(text: str) -> str:
    """Translate English text to formal Hindi using Mayura v1."""
    key = os.getenv("SARVAM_API_KEY", "")
    if not key or not text.strip():
        return text
    try:
        r = httpx.post(
            f"{SARVAM_BASE}/translate",
            headers=_headers(),
            json={
                "input":                text[:3000],   # Mayura limit
                "source_language_code": "en-IN",
                "target_language_code": "hi-IN",
                "model":                "mayura:v1",
                "mode":                 "formal",
                "output_script":        "fully-native",
                "numerals_format":      "international",
                "enable_preprocessing": False,
            },
            timeout=20.0,
        )
        r.raise_for_status()
        return r.json().get("translated_text", text)
    except Exception as exc:
        print(f"[Sarvam] Translation error: {exc}")
        return text


def text_to_speech(text: str, lang: str = "hi-IN", speaker: str = "shubh") -> bytes:
    """Convert text to speech using Bulbul v3. Returns MP3 bytes."""
    key = os.getenv("SARVAM_API_KEY", "")
    if not key or not text.strip():
        return b""
    try:
        r = httpx.post(
            f"{SARVAM_BASE}/text-to-speech",
            headers=_headers(),
            json={
                "text":                text[:2500],   # Bulbul v3 limit
                "target_language_code": lang,
                "model":               "bulbul:v3",
                "speaker":             speaker,
                "pace":                1.0,
                "temperature":         1.0,
                "speech_sample_rate":  24000,
                "output_audio_codec":  "mp3",
            },
            timeout=30.0,
        )
        r.raise_for_status()
        return base64.b64decode(r.json()["audios"][0])
    except Exception as exc:
        print(f"[Sarvam] TTS error: {exc}")
        return b""
