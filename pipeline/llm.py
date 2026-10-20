import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pipeline.paths import ENV_PATH

ENV_KEY = "GEMINI_API_KEY"
ENV_MODEL = "GEMINI_MODEL"

MISSING_KEY_HELP = (
    "GEMINI_API_KEY is not set. Add it to .env as "
    "GEMINI_API_KEY=... (create a key in Google AI Studio)."
)
MISSING_MODEL_HELP = (
    "GEMINI_MODEL is not set. Add a model name to .env as GEMINI_MODEL=... "
    "(copy the exact name from the model dropdown in Google AI Studio)."
)


class LLMError(RuntimeError):
    """Something went wrong talking to the LLM or with its configuration."""


class LLMParseError(LLMError):
    """The model returned a response we could not parse into the expected JSON."""


def _api_key() -> str:
    load_dotenv(ENV_PATH)
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        raise LLMError(MISSING_KEY_HELP)
    return key


def _model_name() -> str:
    load_dotenv(ENV_PATH)
    model = os.environ.get(ENV_MODEL, "").strip()
    if not model:
        raise LLMError(MISSING_MODEL_HELP)
    return model


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    return genai.Client(api_key=_api_key())


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        text = text[newline + 1 :] if newline != -1 else ""
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def generate_json(prompt: str) -> dict:
    """Send one prompt to Gemini and return the parsed JSON object."""
    client = _client()
    model = _model_name()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
    except Exception as err:
        raise LLMError(f"Gemini request failed: {err}") from err

    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise LLMParseError("Gemini returned an empty response")

    try:
        parsed = json.loads(_strip_fences(text))
    except json.JSONDecodeError as err:
        raise LLMParseError(f"Gemini did not return valid JSON: {err}") from err

    if not isinstance(parsed, dict):
        raise LLMParseError(
            f"expected a JSON object, got {type(parsed).__name__}"
        )
    return parsed
