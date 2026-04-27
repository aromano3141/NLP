import asyncio
import json
import logging
from typing import List, Optional
from src.schemas.models import Prompt, LocalizationSchema
from pydantic import ValidationError
from .llm_client import async_generate_text

logger = logging.getLogger(__name__)

class LocalizationPipeline:
    def __init__(self, target_languages: List[str] = ["Chinese", "Arabic", "Hindi"], model_name: str = "gpt-4o"):
        self.target_languages = target_languages
        self.model_name = model_name

    async def translate_and_localize(self, english_prompt: Prompt, target_lang: str) -> Optional[Prompt]:
        """Step 1 & 2: Translate and swap cultural anchors."""
        logger.debug(f"Translating and localizing prompt {english_prompt.id} to {target_lang}")
        system_prompt = "You are an expert translator and cultural localization specialist."
        prompt_text = f"""
        Translate the following benchmark prompt into {target_lang}.
        
        CRITICAL RULES:
        1. Accurately translate all instructions, tasks, and constraints.
        2. Cultural Localization: Replace English-centric cultural anchors (e.g., "YouTube", "New York", "Thanksgiving") with localized equivalents that resonate deeply with the {target_lang} speaking culture.
        3. Do NOT change the logical difficulty or the rigid constraints (e.g., word count limits, format requirements).
        4. If it's a Programming task, explicitly instruct the model to write the functional code/syntax in English, but output all docstrings/comments in {target_lang}.
        
        Original English Instruction:
        {english_prompt.instruction}
        
        Original Base Text:
        {english_prompt.reading_materials or "N/A"}
        
        Sub-tasks to translate:
        """
        for t in english_prompt.sub_tasks:
            prompt_text += f"\nTask: {t.instruction}\nConstraints:\n"
            for c in t.constraints:
                prompt_text += f"- {c.description} ({c.type.value})\n"
                
        prompt_text += f"""
        Return the localized content in JSON format matching the structure:
        {{
            "localized_base_text": "...",
            "localized_instruction": "...",
            "cultural_accessibility_labels": ["list", "of", "replaced", "anchors", "e.g., YouTube -> Hotstar"],
            "sub_tasks": [
                {{
                    "instruction": "...",
                    "constraints": [
                        {{"description": "..."}}
                    ]
                }}
            ]
        }}
        """
        
        response = await async_generate_text(prompt_text, model=self.model_name, system_prompt=system_prompt)
        
        try:
            # Clean JSON markdown
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            raw_json = json.loads(response)
            parsed = LocalizationSchema(**raw_json)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse localization JSON for {target_lang}: {e}")
            return None

        # Create a deep copy of the original prompt but replace fields
        localized_prompt = english_prompt.model_copy(deep=True)
        localized_prompt.language = target_lang
        localized_prompt.id = f"{english_prompt.id}_{target_lang[:2].lower()}"
        localized_prompt.reading_materials = parsed.localized_base_text
        localized_prompt.instruction = parsed.localized_instruction
        localized_prompt.cultural_accessibility_labels = parsed.cultural_accessibility_labels
        
        for i, st in enumerate(localized_prompt.sub_tasks):
            if i < len(parsed.sub_tasks):
                st.instruction = parsed.sub_tasks[i].instruction
                for j, c in enumerate(st.constraints):
                    if j < len(parsed.sub_tasks[i].constraints):
                        c.description = parsed.sub_tasks[i].constraints[j].description
        
        logger.debug(f"Successfully localized prompt {localized_prompt.id}")
        return localized_prompt

    async def verify_back_translation(self, localized_prompt: Prompt) -> bool:
        """Step 3: Back-Translation verification."""
        logger.debug(f"Verifying back-translation for {localized_prompt.id}")
        system_prompt = "You are a QA verification system."
        prompt_text = f"""
        Back-translate the following {localized_prompt.language} text into English, and check if the core semantic intent and atomic constraints are preserved compared to strict benchmark standards.
        
        Instruction: {localized_prompt.instruction}
        
        Reply ONLY with "PASS" if the logic and constraints are perfectly intact, or "FAIL: <reason>" if they are broken.
        """
        response = await async_generate_text(prompt_text, model=self.model_name, system_prompt=system_prompt)
        passed = "PASS" in response.upper()
        if not passed:
            logger.warning(f"Back-translation failed for {localized_prompt.id}: {response}")
        return passed

    async def run_pipeline(self, english_prompt: Prompt) -> List[Prompt]:
        """Run the full localization pipeline for all target languages."""
        results = []
        for lang in self.target_languages:
            loc_prompt = await self.translate_and_localize(english_prompt, lang)
            if loc_prompt:
                is_valid = await self.verify_back_translation(loc_prompt)
                if is_valid:
                    results.append(loc_prompt)
                else:
                    logger.warning(f"Prompt {english_prompt.id} failed back-translation verification for {lang}.")
        return results
