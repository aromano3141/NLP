import os
import asyncio
import logging
from dotenv import load_dotenv

from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

if not OLLAMA_API_KEY:
    logger.error("OLLAMA_API_KEY is missing from environment variables!")

client = AsyncOpenAI(
    api_key=OLLAMA_API_KEY,
    base_url="https://ollama.com/v1",
)

DEFAULT_MODEL = "gpt-oss:20b"


async def async_generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 5000,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful assistant.",
    json_mode: bool = False,
) -> str:
    """
    Generate text using Ollama Cloud models.
    """

    try:
        logger.debug(f"Calling Ollama model: {model}")

        kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)

        return response.choices[0].message.content or ""

    except Exception as e:
        logger.error(f"Ollama API Error: {e}", exc_info=True)
        return ""