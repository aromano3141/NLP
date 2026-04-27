import re
import json
import logging
from typing import Dict, Any, List, Optional
from src.schemas.models import Prompt, GeneratedOutput, EvaluationResult, ConstraintType, SubjectiveEvalResult
from src.pipeline.llm_client import async_generate_text
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, judge_model_name: str = "gpt-4o"):
        self.judge_model = judge_model_name

    def evaluate_format_constraint(self, output_text: str, constraint_desc: str) -> Optional[bool]:
        """Deterministic Evaluation: Check for specific formats like Markdown tables, JSON."""
        logger.debug(f"Evaluating format constraint: {constraint_desc}")
        desc_lower = constraint_desc.lower()
        if "markdown table" in desc_lower or "table format" in desc_lower:
            # Check for markdown table structure
            return bool(re.search(r"\|.*\|", output_text)) and bool(re.search(r"\|[-\s:]+\|", output_text))
        if "json" in desc_lower:
            return "{" in output_text and "}" in output_text
        if "numbered list" in desc_lower or "numbering" in desc_lower:
            return bool(re.search(r"^\s*\d+\.", output_text, re.MULTILINE))
        if "bulleted list" in desc_lower or "bullet points" in desc_lower:
            return bool(re.search(r"^\s*[-*]\s+", output_text, re.MULTILINE))
        
        # If deterministic regex doesn't match a known format, return None to fallback to LLM
        return None 

    def evaluate_length_constraint(self, output_text: str, constraint_desc: str) -> Optional[bool]:
        """Deterministic Evaluation: Calculate word or character count adherence."""
        logger.debug(f"Evaluating length constraint: {constraint_desc}")
        desc_lower = constraint_desc.lower()
        
        is_char = "character" in desc_lower
        
        if is_char:
            count = len(output_text)
        else:
            count = len(output_text.split())
        
        # Look for numbers
        numbers = [int(n) for n in re.findall(r'\d+', constraint_desc)]
        
        if len(numbers) == 2 and ("between" in desc_lower or "range" in desc_lower or "-" in desc_lower):
            limit_low, limit_high = sorted(numbers)
            return limit_low <= count <= limit_high
            
        elif len(numbers) >= 1:
            limit = numbers[0]
            if "limit" in desc_lower or "maximum" in desc_lower or "at most" in desc_lower or "no more than" in desc_lower:
                return count <= limit
            if "minimum" in desc_lower or "at least" in desc_lower or "no less than" in desc_lower:
                return count >= limit
            if "exactly" in desc_lower or "exact" in desc_lower:
                # Allow 5% deviation for words, exact for chars
                if is_char:
                    return count == limit
                return abs(count - limit) / limit <= 0.05
        
        # If we can't reliably parse the constraint description, fallback to LLM
        return None

    async def evaluate_subjective_constraint(self, output_text: str, constraint_desc: str, c_type: str) -> bool:
        """LLM-as-a-Judge Evaluation for Tone, Emotion, Style, Content inclusions."""
        logger.debug(f"Evaluating subjective constraint (LLM fallback): {constraint_desc}")
        system_prompt = "You are an impartial judge evaluating an AI model's response."
        prompt = f"""
        Did the model successfully follow this specific constraint?
        
        Constraint Type: {c_type}
        Constraint Description: {constraint_desc}
        
        Model Output:
        {output_text}
        
        Reply strictly with a JSON object: {{"success": true}} or {{"success": false}}
        """
        response = await async_generate_text(prompt, model=self.judge_model, system_prompt=system_prompt)
        
        try:
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()
            
            raw_json = json.loads(response)
            parsed = SubjectiveEvalResult(**raw_json)
            return parsed.success
        except (json.JSONDecodeError, ValidationError) as e:
            # Fallback to simple string matching if json parsing fails
            logger.warning(f"Failed to parse JSON in subjective evaluation. Falling back to string matching. Error: {e}")
            return "true" in response.lower()

    async def evaluate(self, prompt: Prompt, output: GeneratedOutput) -> EvaluationResult:
        logger.info(f"Starting evaluation for prompt {prompt.id}")
        constraint_results = {}
        total_constraints = 0
        satisfied_constraints = 0

        for task in prompt.sub_tasks:
            for constraint in task.constraints:
                total_constraints += 1
                c_id = constraint.id
                
                success = None
                
                # Route based on deterministic vs subjective
                if constraint.type == ConstraintType.FORMAT:
                    success = self.evaluate_format_constraint(output.output_text, constraint.description)
                elif constraint.type == ConstraintType.LENGTH:
                    success = self.evaluate_length_constraint(output.output_text, constraint.description)
                
                # If deterministic evaluator returns None, it means fallback is needed
                if success is None:
                    success = await self.evaluate_subjective_constraint(output.output_text, constraint.description, constraint.type.value)
                
                constraint_results[c_id] = success
                if success:
                    satisfied_constraints += 1

        rfr = satisfied_constraints / total_constraints if total_constraints > 0 else 1.0
        ifr = 1.0 if rfr == 1.0 else 0.0
        
        logger.info(f"Completed evaluation for prompt {prompt.id}: RFR={rfr:.2f}, IFR={ifr:.2f}")

        return EvaluationResult(
            prompt_id=prompt.id,
            model_name=output.model_name,
            rfr_score=rfr,
            ifr_score=ifr,
            constraint_results=constraint_results
        )
