import asyncio
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from .llm_client import async_generate_text
from src.schemas.models import Prompt, SubTask, Constraint, ConstraintType, ParsedPromptData
from src.schemas.constants import CORE_TASK_CATEGORIES, CONSTRAINT_DIMENSIONS
from pydantic import ValidationError
import random
logger = logging.getLogger(__name__)

class SeedGenerator:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    async def generate_base_text(self, category: str) -> str:
        """Step 1: Synthesize base text/dialogue for a specific category."""
        logger.debug(f"Generating base text for category: {category}")
        system_prompt = "You are an expert dataset creator."
        prompt = f"""
        Generate a base reading material or dialogue for a complex instruction following task.
        The core task category is: {category}
        
        If the category is 'Dialogue System', generate a realistic multi-turn conversation.
        If it's 'Programming', generate a problem description or a code snippet.
        Otherwise, generate an interesting article, story, or informational text.
        
        Output ONLY the text or dialogue, nothing else.
        """
        return await async_generate_text(prompt, model=self.model_name, system_prompt=system_prompt)

    async def expand_tasks(self, base_text: str, category: str, num_tasks: int = 3) -> str:
        """Step 2: Task Expansion based on the base text."""
        logger.debug(f"Expanding {num_tasks} tasks for category: {category}")
        system_prompt = "You are an assistant to help generate comprehensive multi-task instructions from basic texts."
        prompt = f"""
        Based on the given basic text, design {num_tasks} different types of extended tasks.
        The generated tasks should be placed after "-output-:".
        
        --- PRE-GENERATION VALIDATION CHECK ---
        Before writing any tasks, ensure each task satisfies:

        1. Explicitness: no ambiguity; instruction must be directly actionable
        2. Completeness: includes all required steps implied by base text
        3. Atomicity: exactly one measurable action per task
        4. Categorization: correctly aligned with category {category}

        If any task fails a check, revise internally before outputting.

        --- RULES ---
        1. Task design must be based on the input text content.
        2. Task instructions should be clear and specific.
        3. Aim to increase task difficulty, selecting tasks that require multi-step reasoning.
        
        --input--:
        {base_text}
        
        -output-:
        """
        return await async_generate_text(prompt, model=self.model_name, system_prompt=system_prompt)

    async def expand_constraints(self, tasks_text: str, density: str = "Medium") -> str:
        """Step 3: Constraint Expansion."""
        logger.debug(f"Expanding constraints with density: {density}")
        system_prompt = "You are an expert at generating constraints."
        constraint_options = "\n".join(
            [f"- {key}: {', '.join(map(str, values))}" 
            for key, values in CONSTRAINT_DIMENSIONS.items()]
        )
        num_constraints = {"Low": "1-4", "Medium": "4-5", "High": "6-8"}[density]
        
        prompt = f"""
        Modify the original constraint information for each instruction.
        For every SUB_INSTRUCTION, generate at most {num_constraints} high-quality constraints.

        --- PRE-GENERATION VALIDATION CHECK ---
        Before generating constraints, ensure each constraint satisfies:

        1. Explicitness: clearly measurable and not vague (avoid "reasonable", "appropriate", etc.)
        2. Completeness: fully covers necessary conditions for correctness of task execution
        3. Atomicity: one constraint = one measurable rule
        4. Categorization: must match one allowed constraint type exactly

        Additionally:
        - Do not create overlapping constraints
        - Do not split one requirement into redundant constraints
        - Ensure constraints are independently verifiable

        --- CONSTRAINT RULES ---
        - Each constraint must address key requirements of the task.
        - Constraints must be measurable and analyzable.
        - Select constraint types only from the allowed types and values below.

        Constraint types and allowed values:
        {constraint_options}
        --input--:
        {tasks_text}
        
        -output-:
        """
        return await async_generate_text(prompt, model=self.model_name, system_prompt=system_prompt)

    async def generate_seed_prompt(self, category: str, density: str = "Medium") -> Optional[Prompt]:
        """Orchestrate the full pipeline to generate one seed prompt in English."""
        logger.info(f"Generating seed prompt for category '{category}' with density '{density}'")
        base_text = await self.generate_base_text(category)
        random_num_tasks = random.randint(3, 10)
        tasks_response = await self.expand_tasks(base_text, category,random_num_tasks)
        
        constraints_response = await self.expand_constraints(tasks_response, density)
        
        constraint_types = ", ".join(c.value for c in ConstraintType)

        system_prompt = "You are an expert parsing assistant."
        parsing_prompt = f"""
        You are given outputs from a multi-stage prompt generation pipeline.

        STAGE 1 — BASE PROMPT (main instruction):
        {base_text}

        STAGE 2 — EXPANDED SUBTASKS:
        {tasks_response}

        STAGE 3 — SUBTASK CONSTRAINTS:
        {constraints_response}

        Using ALL THREE stages, construct a structured JSON object.

        Rules:
        - "instruction" must come from STAGE 1 (base prompt).
        - "sub_tasks" must come from STAGE 2.
        - Constraints for each subtask must come from STAGE 3.
        - Preserve original wording.
        - Match each constraint to the correct subtask.
        - Do not invent tasks or constraints.
        - Return ONLY raw valid JSON.

        Schema:
        {{
        "instruction": "<base prompt>",
        "sub_tasks": [
            {{
            "instruction": "<subtask>",
            "constraints": [
                {{
                "type": "<one of {constraint_types}>",
                "description": "<constraint text>"
                }}
            ]
            }}
        ]
        }}
        """
                
        parsed_data = None
        for attempt in range(3):
            logger.debug(f"Parsing structured output, attempt {attempt+1}")
            structured_output_str = await async_generate_text(parsing_prompt, model=self.model_name, system_prompt=system_prompt, json_mode=True)
            
            if not structured_output_str:
                logger.warning(f"Received empty response from model on attempt {attempt+1}")
                if attempt == 2:
                    logger.error("All parsing attempts failed due to empty response. Returning None.")
                    return None
                continue
            
            # Clean JSON markdown blocks if present
            if structured_output_str.startswith("```json"):
                structured_output_str = structured_output_str.split("```json")[1].split("```")[0].strip()
            elif structured_output_str.startswith("```"):
                structured_output_str = structured_output_str.split("```")[1].split("```")[0].strip()
                
            try:
                raw_json = json.loads(structured_output_str)
                parsed_data = ParsedPromptData(**raw_json)
                break
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Failed to parse or validate JSON on attempt {attempt+1}: {e}")
                if attempt == 2:
                    logger.error("All parsing attempts failed. Returning None.")
                    return None
            
        if not parsed_data:
            return None

        prompt_id = str(uuid.uuid4())
        
        sub_tasks = []
        for i, st in enumerate(parsed_data.sub_tasks):
            constraints = []
            for j, c in enumerate(st.constraints):
                ctype_str = c.type
                try:
                    ctype = ConstraintType(ctype_str)
                except ValueError:
                    ctype = ConstraintType.CONTENT
                    
                constraints.append(Constraint(
                    id=f"{prompt_id}_task_{i}_c_{j}",
                    type=ctype,
                    description=c.description
                ))
                
            sub_tasks.append(SubTask(
                id=f"{prompt_id}_task_{i}",
                instruction=st.instruction,
                constraints=constraints
            ))

        prompt = Prompt(
            id=prompt_id,
            language="English",
            core_task_category=category,
            instruction=parsed_data.instruction,
            sub_tasks=sub_tasks,
            reading_materials=base_text,
            cultural_accessibility_labels=[],
            density_level=density
        )
        logger.info(f"Successfully generated seed prompt {prompt_id}")
        return prompt
