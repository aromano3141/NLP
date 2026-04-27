import os
import asyncio
import logging
import aiohttp
import requests
from dotenv import load_dotenv

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

load_dotenv()

logger = logging.getLogger(__name__)

ENDPOINT = "https://models.github.ai/inference"
DEFAULT_MODEL = "openai/gpt-4o"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
FALLBACK_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    retry_total=0,
)

async def fallback_openrouter(
    prompt: str, 
    system_prompt: str, 
    max_tokens: int, 
    temperature: float
) -> str:
    """Async helper function to call OpenRouter API."""
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set. Cannot use OpenRouter fallback.")
        return ""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": FALLBACK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        logger.warning(f"Initiating OpenRouter fallback using {FALLBACK_MODEL} (async)...")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await resp.text()
                    logger.error(f"OpenRouter Fallback failed with status {resp.status}: {error_text}")
                    return ""
    except Exception as e:
        logger.error(f"Error during OpenRouter fallback: {e}", exc_info=True)
        return ""

async def async_generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful assistant."
) -> str:
    """Asynchronous generation using GitHub Models with Fallback."""
    try:
        logger.debug(f"Calling {model} (async)")
        response = await asyncio.to_thread(
            client.complete,
            model=model,
            messages=[
                SystemMessage(system_prompt),
                UserMessage(prompt),
            ],
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    except HttpResponseError as e:
        if e.status_code == 429:
            logger.warning(f"Rate limit (429) calling {model} on GitHub. Triggering fallback.")
            return await fallback_openrouter(prompt, system_prompt, max_tokens, temperature)
        logger.error(f"HTTP error calling {model} on GitHub: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling {model} on GitHub: {e}")
        raise