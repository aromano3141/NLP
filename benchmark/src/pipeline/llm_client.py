import os
import asyncio
import logging
from dotenv import load_dotenv

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

logger = logging.getLogger(__name__)

ENDPOINT = "https://models.github.ai/inference"
DEFAULT_MODEL = "openai/gpt-4o"

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
)

def generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful assistant."
) -> str:
    try:
        logger.debug(f"Calling {model} (sync)")
        response = client.complete(
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

    except Exception as e:
        logger.error(f"Error calling {model}: {e}", exc_info=True)
        return ""


async def async_generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful assistant."
) -> str:
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

    except Exception as e:
        logger.error(f"Error calling {model}: {e}", exc_info=True)
        return ""


async def async_generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    system_prompt: str = "You are a helpful assistant."
) -> str:
    try:
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

    except Exception as e:
        print(f"Error calling {model}: {e}")
        return ""