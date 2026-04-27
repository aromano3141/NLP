import re
from typing import Dict, Any, List
from NLP.benchmark.src.schemas.models import Prompt, GeneratedOutput, EvaluationResult, ConstraintType
from NLP.benchmark.src.pipeline.llm_client import async_generate_text

class Evaluator:
    def __init__(self, judge_model_name: str = "gpt-4o"):
        self.judge_model = judge_model_name

    def evaluate_format_constraint(self, output_text: str, constraint_desc: str) -> bool:
        """Deterministic Evaluation: Check for specific formats like Markdown tables, JSON."""
        desc_lower = constraint_desc.lower()
        if "markdown table" in desc_lower:
            # Simple check for markdown table structure
            return bool(re.search(r"\|.*\|", output_text)) and bool(re.search(r"\|[-\s:]+\|", output_text))
        if "json" in desc_lower:
            return "{" in output_text and "}" in output_text
        if "numbered list" in desc_lower or "numbering" in desc_lower:
            return bool(re.search(r"^\s*\d+\.", output_text, re.MULTILINE))
        # Fallback to true if we don't have a specific deterministic check, or we could return False.
        # Ideally, we would have regex rules mapped to constraint metadata.
        return True 

    def evaluate_length_constraint(self, output_text: str, constraint_desc: str) -> bool:
        """Deterministic Evaluation: Calculate word count adherence."""
        words = output_text.split()
        count = len(words)
        
        # Extremely basic extraction of target length from description for demonstration
        # E.g., "50-word limit"
        numbers = re.findall(r'\d+', constraint_desc)
        if numbers:
            limit = int(numbers[0])
            if "limit" in constraint_desc.lower() or "maximum" in constraint_desc.lower():
                return count <= limit
            if "minimum" in constraint_desc.lower() or "at least" in constraint_desc.lower():
                return count >= limit
            if "exact" in constraint_desc.lower():
                # Allow 10% deviation
                return abs(count - limit) / limit <= 0.1
        return True # If no numbers found, pass

    async def evaluate_subjective_constraint(self, output_text: str, constraint_desc: str, c_type: str) -> bool:
        """LLM-as-a-Judge Evaluation for Tone, Emotion, Style, Content inclusions."""
        prompt = f"""
        You are an impartial judge evaluating an AI model's response.
        Did the model successfully follow this specific constraint?
        
        Constraint Type: {c_type}
        Constraint Description: {constraint_desc}
        
        Model Output:
        {output_text}
        
        Reply strictly with a JSON object: {{"success": true}} or {{"success": false}}
        """
        response = await async_generate_text(prompt, model=self.judge_model)
        return "true" in response.lower()

    async def evaluate(self, prompt: Prompt, output: GeneratedOutput) -> EvaluationResult:
        constraint_results = {}
        total_constraints = 0
        satisfied_constraints = 0

        for task in prompt.sub_tasks:
            for constraint in task.constraints:
                total_constraints += 1
                c_id = constraint.id
                
                # Route based on deterministic vs subjective
                if constraint.type == ConstraintType.FORMAT:
                    success = self.evaluate_format_constraint(output.output_text, constraint.description)
                elif constraint.type == ConstraintType.LENGTH:
                    success = self.evaluate_length_constraint(output.output_text, constraint.description)
                else:
                    success = await self.evaluate_subjective_constraint(output.output_text, constraint.description, constraint.type.value)
                
                constraint_results[c_id] = success
                if success:
                    satisfied_constraints += 1

        rfr = satisfied_constraints / total_constraints if total_constraints > 0 else 1.0
        ifr = 1.0 if rfr == 1.0 else 0.0

        return EvaluationResult(
            prompt_id=prompt.id,
            model_name=output.model_name,
            rfr_score=rfr,
            ifr_score=ifr,
            constraint_results=constraint_results
        )
