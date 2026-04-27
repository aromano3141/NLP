import asyncio
import json
import logging
from typing import Dict, Any, Tuple
from src.schemas.models import Prompt, ValidationResult
from src.pipeline.llm_client import async_generate_text
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class InstructionValidator:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    async def extract_and_verify(self, prompt: Prompt) -> Tuple[bool, str]:
        """
        Phase 3: Instruction-Level Validation
        Extracts requirements and checks for logical consistency and feasibility.
        """
        logger.debug(f"Extracting and verifying instructions for prompt {prompt.id}")
        system_prompt = "You are an automated Requirement Extraction and Feasibility checker."
        prompt_text = f"""
        You are an automated Requirement Extraction and Feasibility checker.

        Evaluate the prompt using a *tolerant, human-like grading approach*.

        You must NOT be overly strict. Minor ambiguity or natural language flexibility is acceptable.

        ---

        ## Evaluation Criteria (graded, not binary)

        1. Explicitness
        - Check if instructions are generally understandable and actionable.
        - Minor ambiguity is acceptable.
        - Only flag if interpretation is genuinely unclear or conflicting.

        2. Completeness
        - Check if required information is mostly present.
        - Do NOT require exhaustive coverage.
        - Accept reasonable abstraction or missing non-critical details.

        3. Atomicity
        - Prefer single-measure constraints.
        - HOWEVER, allow small compound constraints if they describe closely related requirements.

        4. Categorization
        - Check if categories are reasonable and consistent.
        - Do NOT require perfect taxonomy alignment if intent is clear.

        ---

        ## Logical Conflict Check (STRICT ONLY HERE)
        Flag ONLY if there is a true contradiction such as:
        - Impossible constraints (e.g., "max 10 words + fully detailed 5-page explanation")
        - Mutually exclusive requirements
        - Unmeasurable constraints that cannot be evaluated at all

        ---

        ## Decision Rules

        - VALID if:
        - No logical conflicts AND
        - At most 1 moderate issue across all pillars

        - INVALID if:
        - Any severe logical conflict OR
        - Multiple major pillar failures (2+ pillars severely broken)

        ---

        Prompt Instruction:
        {prompt.instruction}
        
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

        response = await async_generate_text(prompt_text, model=self.model_name, system_prompt=system_prompt)
        
        try:
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            raw_json = json.loads(response)
            parsed = ValidationResult(**raw_json)
            if not parsed.is_valid:
                logger.debug(f"Prompt {prompt.id} validation failed: {parsed.reason}")
            return parsed.is_valid, parsed.reason
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"LLM Output was not valid JSON or failed schema validation: {e}")
            return False, f"LLM Output was not valid JSON or failed schema validation: {e}"
