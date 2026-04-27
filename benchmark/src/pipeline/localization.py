import asyncio
from typing import List
from NLP.benchmark.src.schemas.models import Prompt
from .llm_client import async_generate_text

class LocalizationPipeline:
    def __init__(self, target_languages: List[str] = ["Chinese", "Arabic", "Hindi"], model_name: str = "gpt-4o"):
        self.target_languages = target_languages
        self.model_name = model_name

    async def translate_and_localize(self, english_prompt: Prompt, target_lang: str) -> Prompt:
        """Step 1 & 2: Translate and swap cultural anchors."""
        prompt_text = f"""
        You are an expert translator and cultural localization specialist.
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
        
        response = await async_generate_text(prompt_text, model=self.model_name)
        
        import json
        try:
            # Clean JSON markdown
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse localization JSON for {target_lang}")
            return None

        # Create a deep copy of the original prompt but replace fields
        localized_prompt = english_prompt.model_copy(deep=True)
        localized_prompt.language = target_lang
        localized_prompt.id = f"{english_prompt.id}_{target_lang[:2].lower()}"
        localized_prompt.reading_materials = parsed.get("localized_base_text")
        localized_prompt.instruction = parsed.get("localized_instruction")
        localized_prompt.cultural_accessibility_labels = parsed.get("cultural_accessibility_labels", [])
        
        for i, st in enumerate(localized_prompt.sub_tasks):
            if i < len(parsed.get("sub_tasks", [])):
                st.instruction = parsed["sub_tasks"][i].get("instruction", st.instruction)
                for j, c in enumerate(st.constraints):
                    if j < len(parsed["sub_tasks"][i].get("constraints", [])):
                        c.description = parsed["sub_tasks"][i]["constraints"][j].get("description", c.description)
        
        return localized_prompt

    async def verify_back_translation(self, localized_prompt: Prompt) -> bool:
        """Step 3: Back-Translation verification."""
        prompt_text = f"""
        You are a QA verification system.
        Back-translate the following {localized_prompt.language} text into English, and check if the core semantic intent and atomic constraints are preserved compared to strict benchmark standards.
        
        Instruction: {localized_prompt.instruction}
        
        Reply ONLY with "PASS" if the logic and constraints are perfectly intact, or "FAIL: <reason>" if they are broken.
        """
        response = await async_generate_text(prompt_text, model=self.model_name)
        return "PASS" in response.upper()

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
                    print(f"Prompt {english_prompt.id} failed back-translation verification for {lang}.")
        return results
