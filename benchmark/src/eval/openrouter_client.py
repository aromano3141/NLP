import asyncio
import aiohttp
import logging
import os

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"


async def openrouter_chat(
    messages: list[dict],
    model: str,
    api_key: str | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    semaphore: asyncio.Semaphore | None = None,
    retries: int = 3,
) -> str:
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async def _call(session: aiohttp.ClientSession) -> str:
        for attempt in range(retries):
            try:
                async with session.post(
                    OPENROUTER_BASE,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"].get("content", "") or ""
                    elif resp.status == 429:
                        wait = 30 * (attempt + 1)
                        logger.warning(f"[{model}] Rate limited. Waiting {wait}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait)
                    else:
                        err = await resp.text()
                        logger.error(f"[{model}] HTTP {resp.status}: {err[:300]}")
                        await asyncio.sleep(5 * (attempt + 1))
            except asyncio.TimeoutError:
                logger.warning(f"[{model}] Timeout on attempt {attempt + 1}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"[{model}] Error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(5)
        logger.error(f"[{model}] All {retries} attempts failed")
        return ""

    if semaphore:
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                return await _call(session)
    else:
        async with aiohttp.ClientSession() as session:
            return await _call(session)
