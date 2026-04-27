import asyncio
import json
import uuid
from typing import List, Dict, Any
from .llm_client import async_generate_text
from NLP.benchmark.src.schemas.models import Prompt, SubTask, Constraint, ConstraintType
from NLP.benchmark.src.schemas.constants import CORE_TASK_CATEGORIES, CONSTRAINT_DIMENSIONS

class SeedGenerator:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    async def generate_base_text(self, category: str) -> str:
        """Step 1: Synthesize base text/dialogue for a specific category."""
        prompt = f"""
        You are an expert dataset creator. Generate a base reading material or dialogue for a complex instruction following task.
        The core task category is: {category}
        
        If the category is 'Dialogue System', generate a realistic multi-turn conversation.
        If it's 'Programming', generate a problem description or a code snippet.
        Otherwise, generate an interesting article, story, or informational text.
        
        Output ONLY the text or dialogue, nothing else.
        """
        return await async_generate_text(prompt, model=self.model_name)

    async def expand_tasks(self, base_text: str, category: str, num_tasks: int = 3) -> str:
        """Step 2: Task Expansion based on the base text."""
        # Adapted from E.1 Prompt for task expansion
        prompt = f"""
        You are an assistant to help generate comprehensive multi-task instructions from basic texts.
        Based on the given basic text, design {num_tasks} different types of extended tasks.
        The generated tasks should be placed after "-output-:".
        
        Rules:
        1. Task design must be based on the input text content.
        2. Task instructions should be clear and specific.
        3. Aim to increase task difficulty, selecting tasks that require multi-step reasoning.
        4. Write the thought process after "-explanation-:".
        
        --input--:
        {base_text}
        
        -output-:
        """
        response = await async_generate_text(prompt, model=self.model_name)
        return response

    async def expand_constraints(self, tasks_text: str, density: str = "Medium") -> str:
        """Step 3: Constraint Expansion."""
        num_constraints = {"Low": "1-2", "Medium": "3-4", "High": "5"}[density]
        
        # Adapted from E.4 Prompt for constraint expansion
        prompt = f"""
        You are an expert at generating constraints.
        Modify the original constraint information for each instruction.
        For every SUB_INSTRUCTION, generate {num_constraints} high-quality constraints.
        Each constraint must address key requirements of the task with measurable analysis.
        
        Constraint types should be selected from: Theme, Exclusion, Inclusion, Value, Privacy, Numerical, Role-Playing, Target Audience, Prior Condition, Text Background, Tone, Emotion, Multilingual, Output Format, Text Pattern, Grammar Structure, Length.
        
        --input--:
        {tasks_text}
        
        -output-:
        """
        response = await async_generate_text(prompt, model=self.model_name)
        return response

    async def generate_seed_prompt(self, category: str, density: str = "Medium") -> Prompt:
        """Orchestrate the full pipeline to generate one seed prompt in English."""
        # 1. Base Text
        base_text = await self.generate_base_text(category)
        
        # 2. Tasks
        tasks_response = await self.expand_tasks(base_text, category)
        
        # 3. Constraints
        constraints_response = await self.expand_constraints(tasks_response, density)
        
        # 4. Parse into Pydantic schema using another LLM call for structure
        parsing_prompt = f"""
        Extract the structured tasks and constraints from the following text and format it as JSON.
        
        Text:
        {constraints_response}
        
        Return ONLY valid JSON matching this schema:
        {{
            "instruction": "The main instruction",
            "sub_tasks": [
                {{
                    "instruction": "Subtask instruction",
                    "constraints": [
                        {{
                            "type": "Format Constraint", // Must be one of: Content Constraint, Situation Constraint, Style Constraint, Format Constraint, Length Constraint
                            "description": "Output as a Markdown table"
                        }}
                    ]
                }}
            ]
        }}
        """
        
        structured_output_str = await async_generate_text(parsing_prompt, model=self.model_name)
        
        # Clean JSON markdown blocks if present
        if structured_output_str.startswith("```json"):
            structured_output_str = structured_output_str.split("```json")[1].split("```")[0].strip()
        elif structured_output_str.startswith("```"):
            structured_output_str = structured_output_str.split("```")[1].split("```")[0].strip()
            
        try:
            parsed_data = json.loads(structured_output_str)
        except json.JSONDecodeError:
            print("Failed to parse JSON, returning fallback structure.")
            parsed_data = {
                "instruction": "Please complete the tasks based on the provided text.",
                "sub_tasks": []
            }
            
        prompt_id = str(uuid.uuid4())
        
        sub_tasks = []
        for i, st in enumerate(parsed_data.get("sub_tasks", [])):
            constraints = []
            for j, c in enumerate(st.get("constraints", [])):
                ctype_str = c.get("type", "Content Constraint")
                try:
                    ctype = ConstraintType(ctype_str)
                except ValueError:
                    ctype = ConstraintType.CONTENT
                    
                constraints.append(Constraint(
                    id=f"{prompt_id}_task_{i}_c_{j}",
                    type=ctype,
                    description=c.get("description", "")
                ))
                
            sub_tasks.append(SubTask(
                id=f"{prompt_id}_task_{i}",
                instruction=st.get("instruction", ""),
                constraints=constraints
            ))

        prompt = Prompt(
            id=prompt_id,
            language="English",
            core_task_category=category,
            instruction=parsed_data.get("instruction", "Please complete the following tasks based on the provided text."),
            sub_tasks=sub_tasks,
            reading_materials=base_text,
            cultural_accessibility_labels=[],
            density_level=density
        )
        return prompt
