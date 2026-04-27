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
FALLBACK_MODEL = "openrouter/free"
#FALLBACK_MODEL = "tencent/hy3-preview:free"

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    retry_total=0,
)

async def fallback_openrouter(
    prompt: str, 
    system_prompt: str, 
    max_tokens: int, 
    temperature: float,
    json_mode: bool = False) -> str:
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
    if json_mode:
        data["response_format"] = {"type": "json_object"}

    try:
        logger.warning(f"Initiating OpenRouter fallback using {FALLBACK_MODEL} (async)...")
        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                async with session.post(url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0].get("message", {}).get("content", "")
                            if not content:
                                logger.warning(f"OpenRouter returned empty content. Full response: {result}")
                            return content
                        elif "error" in result:
                            logger.error(f"OpenRouter API error on attempt {attempt+1}: {result['error']}")
                            if attempt < 2:
                                await asyncio.sleep(2)
                                continue
                            return ""
                        else:
                            logger.error(f"OpenRouter Fallback returned unexpected JSON: {result}")
                            return ""
                    else:
                        error_text = await resp.text()
                        logger.error(f"OpenRouter Fallback failed with status {resp.status}: {error_text}")
                        if attempt < 2:
                            await asyncio.sleep(2)
                            continue
                        return ""
            return ""
    except Exception as e:
        logger.error(f"Error during OpenRouter fallback: {e}", exc_info=True)
        return ""

async def async_generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 5000,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful assistant.",
    json_mode: bool = False
) -> str:
    """Asynchronous generation using GitHub Models with Fallback."""
    try:
        logger.debug(f"Calling {model} (async)")
        kwargs = {
            "model": model,
            "messages": [
                SystemMessage(system_prompt),
                UserMessage(prompt),
            ],
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = "json_object"
            
        response = await asyncio.to_thread(
            client.complete,
            **kwargs
        )
        return response.choices[0].message.content

    except HttpResponseError as e:
        status_code = getattr(e, 'status_code', None)
        if status_code == 429 or "Too many requests" in str(e):
            logger.warning(f"Rate limit (429) calling {model} on GitHub. Triggering fallback.")
            return await fallback_openrouter(prompt, system_prompt, max_tokens, temperature, json_mode)
        logger.error(f"HTTP error calling {model} on GitHub: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling {model} on GitHub: {e}")
        raise