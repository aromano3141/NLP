import asyncio
import json
from typing import Dict, Any, Tuple
from NLP.benchmark.src.schemas.models import Prompt
from NLP.benchmark.src.pipeline.llm_client import async_generate_text

class InstructionValidator:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    async def extract_and_verify(self, prompt: Prompt) -> Tuple[bool, str]:
        """
        Phase 3: Instruction-Level Validation
        Extracts requirements and checks for logical consistency and feasibility.
        """
        prompt_text = f"""
        You are an automated Requirement Extraction and Feasibility checker.
        Analyze the following Prompt and its constraints based on four pillars:
        1. Explicitness: Are they clearly stated?
        2. Completeness: Are all constraints accounted for?
        3. Atomicity: Does each address a single measurable aspect?
        4. Categorization: Are they correctly labeled?

        Additionally, check for Logical Conflicts (e.g., demanding a 50-word limit while requiring a comprehensive 5-column Markdown table).

        Prompt Instruction: {prompt.instruction}
        Sub-tasks and Constraints:
        """
        for t in prompt.sub_tasks:
            prompt_text += f"\n- {t.instruction}\n"
            for c in t.constraints:
                prompt_text += f"  * [{c.type.value}] {c.description}\n"

        prompt_text += """
        Return a JSON object:
        {
            "is_valid": true/false,
            "reason": "If false, explain the logical conflict or failure in the four pillars. If true, write 'PASS'."
        }
        """

        response = await async_generate_text(prompt_text, model=self.model_name)
        
        try:
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(response)
            return parsed.get("is_valid", False), parsed.get("reason", "Parse failed")
        except:
            return False, "LLM Output was not valid JSON."
