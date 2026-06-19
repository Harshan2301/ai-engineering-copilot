"""
Gemini API client — wraps google-generativeai with retry logic,
content generation, and text embedding.
"""
import time
import logging
from functools import lru_cache
from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from app.config import get_settings

logger = logging.getLogger("app.core.gemini")
settings = get_settings()


class GeminiClient:
    """Thread-safe Gemini API wrapper with exponential-backoff retries."""

    _MAX_RETRIES = 4
    _BASE_DELAY = 1.0  # seconds

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 8192,
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )
        logger.info("GeminiClient initialised with model: %s", settings.GEMINI_MODEL)

    # ------------------------------------------------------------------
    # Internal retry helper
    # ------------------------------------------------------------------
    def _with_retry(self, fn, *args, **kwargs):
        delay = self._BASE_DELAY
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except (ResourceExhausted, ServiceUnavailable) as exc:
                if attempt == self._MAX_RETRIES:
                    raise
                logger.warning(
                    "Gemini rate-limit hit (attempt %d/%d). Retrying in %.1fs …",
                    attempt, self._MAX_RETRIES, delay,
                )
                time.sleep(delay)
                delay *= 2
            except Exception as exc:
                logger.exception("Gemini API error: %s", exc)
                raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate text from a prompt, with optional system instruction."""
        full_prompt = (
            f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
        )
        logger.debug("Generating content (prompt length=%d chars)", len(full_prompt))
        response = self._with_retry(self._model.generate_content, full_prompt)
        return response.text.strip()

    def embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Return an embedding vector for the given text."""
        result = self._with_retry(
            genai.embed_content,
            model=settings.GEMINI_EMBED_MODEL,
            content=text,
            task_type=task_type,
        )
        return result["embedding"]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (different task type for better retrieval)."""
        return self.embed(text, task_type="RETRIEVAL_QUERY")


@lru_cache(maxsize=1)
def get_gemini_client() -> GeminiClient:
    """Singleton accessor — safe to call from FastAPI depends."""
    return GeminiClient()
