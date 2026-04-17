import os
import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv

# We use litellm for unified access and simplified retries
import litellm
from litellm import acompletion

# Load environment variables
load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Asynchronous client for handling model inference.
    Supports standard OpenAI formats and robust error handling.
    """

    def __init__(self, default_model: str = "gpt-4o-mini", max_retries: int = 3):
        self.default_model = default_model

        # litellm config
        litellm.num_retries = max_retries
        litellm.backoff_factor = 1.5

    async def generate_response(self, prompt: str, model: Optional[str] = None, system_prompt: Optional[str] = None, json_mode: bool = False) -> Optional[str]:
        """
        Generates a response from the LLM.
        """
        target_model = model or self.default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": target_model,
            "messages": messages,
        }

        if json_mode:
            kwargs["response_format"] = { "type": "json_object" }

        try:
            response = await acompletion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during LLM generation with model {target_model}: {str(e)}")
            return None

# Simple test block
if __name__ == "__main__":
    async def test():
        client = LLMClient()
        print("Testing basic generation (requires API key in .env)...")
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY not found. Skipping live test.")
            return

        resp = await client.generate_response("Say hello in Spanish.")
        print(f"Response: {resp}")

    asyncio.run(test())
