import json
import logging
from typing import Dict, List, Any
from data_models import Instruction, EvaluationResult, ConstraintType, Constraint
from llm_client import LLMClient
from metrics_engine import calculate_bls

logger = logging.getLogger(__name__)

class Evaluator:
    """
    LLM-as-a-Judge evaluation engine for COMP-EVAL.
    """

    def __init__(self, judge_model: str = "gpt-4o"):
        self.llm_client = LLMClient(default_model=judge_model)

    def _count_length(self, text: str, unit: str) -> int:
        """Helper to compute deterministic length."""
        if unit == "words":
            return len(text.split())
        elif unit == "characters":
            return len(text)
        else:
            logger.warning(f"Unknown unit {unit}, defaulting to words")
            return len(text.split())

    async def _evaluate_llm_constraints(self, instruction: Instruction, output_text: str, constraints_to_eval: List[Constraint]) -> Dict[str, bool]:
        """
        Uses an LLM judge to evaluate non-length constraints.
        Asks questions based on the English semantic anchors (rubrics).
        """
        if not constraints_to_eval:
            return {}

        rubrics = instruction.evaluation_criteria.rubrics

        # Build the evaluation prompt
        eval_prompt = "You are an expert AI evaluator. Analyze the provided response against the following criteria.\n\n"
        eval_prompt += f"--- TARGET RESPONSE ---\n{output_text}\n-----------------------\n\n"

        eval_prompt += "For each criterion ID, determine if the response satisfies the condition.\n"
        eval_prompt += "Output a JSON object mapping the ID to a boolean (true if satisfied, false otherwise).\n\n"

        eval_prompt += "Criteria:\n"
        expected_keys = []
        for c in constraints_to_eval:
            q = rubrics.get(c.id)
            if not q:
                logger.error(f"No rubric found for constraint {c.id}")
                continue

            expected_keys.append(c.id)
            if c.activation_condition:
                eval_prompt += f"- {c.id}: First check if condition met: '{c.activation_condition}'. If NOT met, output true. If MET, answer: {q}\n"
            else:
                eval_prompt += f"- {c.id}: {q}\n"

        # Call the judge
        system_prompt = "You must respond strictly with valid JSON mapping string IDs to boolean values."
        raw_response = await self.llm_client.generate_response(
            prompt=eval_prompt,
            system_prompt=system_prompt,
            json_mode=True
        )

        if not raw_response:
            logger.error("Failed to get a response from the Judge LLM.")
            return {k: False for k in expected_keys}

        try:
            results = json.loads(raw_response)
            # Ensure all keys are present
            return {k: bool(results.get(k, False)) for k in expected_keys}
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Judge JSON output: {raw_response}")
            return {k: False for k in expected_keys}

    async def evaluate_output(self, instruction: Instruction, output_text: str) -> EvaluationResult:
        """
        Core evaluation function. Computes deterministic length and uses LLM for semantic checks.
        """
        results_dict: Dict[str, bool] = {}
        length_score = None

        llm_eval_constraints = []

        # Flatten all constraints to evaluate
        def flatten_constraints(c_list: List[Constraint]) -> List[Constraint]:
            flat = []
            for c in c_list:
                flat.append(c)
                if c.sub_constraints:
                    flat.extend(flatten_constraints(c.sub_constraints))
            return flat

        all_constraints = flatten_constraints(instruction.constraints)

        for constraint in all_constraints:
            # Deterministic LENGTH evaluation
            if constraint.type == ConstraintType.LENGTH:
                if constraint.target_min is not None and constraint.target_max is not None and constraint.unit:
                    actual_length = self._count_length(output_text, constraint.unit)
                    l_score = calculate_bls(actual_length, constraint.target_min, constraint.target_max)
                    # For holistic pass/fail, LENGTH constraints need to be strictly met
                    results_dict[constraint.id] = (l_score == 1.0)
                    length_score = l_score
                else:
                    logger.warning(f"Length constraint {constraint.id} missing target bounds or unit.")
                    results_dict[constraint.id] = False
            else:
                # Add to queue for LLM Evaluation
                llm_eval_constraints.append(constraint)

        # Run LLM judge
        if llm_eval_constraints:
            llm_results = await self._evaluate_llm_constraints(instruction, output_text, llm_eval_constraints)
            results_dict.update(llm_results)

        # Holistic pass
        holistic = all(results_dict.values()) if results_dict else False

        return EvaluationResult(
            instruction_id=instruction.id,
            constraint_results=results_dict,
            length_score=length_score,
            holistic_pass=holistic
        )
