import os
from dotenv import load_dotenv
import litellm
from litellm import completion

load_dotenv()

# Optionally configure litellm settings globally
litellm.set_verbose = False

def generate_text(prompt: str, model: str = "gpt-4o", max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """
    Generate text using litellm.
    Supports models like 'gpt-4o', 'deepseek/deepseek-chat', etc.
    """
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling {model}: {e}")
        return ""

async def async_generate_text(prompt: str, model: str = "gpt-4o", max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """
    Async version for generating text.
    """
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling {model}: {e}")
        return ""
