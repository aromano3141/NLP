import re
import json
import logging
import asyncio
from typing import Optional, Tuple
from src.schemas.models import Prompt, GeneratedOutput, EvaluationResult, ConstraintType
from src.eval.openrouter_client import openrouter_chat

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        judge_model_name: str = "google/gemini-2.5-flash",
        api_key: str | None = None,
        concurrency: int = 10,
    ):
        self.judge_model = judge_model_name
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(concurrency)

    def evaluate_format_constraint(self, output_text: str, constraint_desc: str) -> Optional[bool]:
        desc_lower = constraint_desc.lower()
        if "markdown table" in desc_lower or "table format" in desc_lower:
            return bool(re.search(r"\|.*\|", output_text)) and bool(re.search(r"\|[-\s:]+\|", output_text))
        if "json" in desc_lower:
            return "{" in output_text and "}" in output_text
        if "numbered list" in desc_lower or "numbering" in desc_lower:
            return bool(re.search(r"^\s*\d+\.", output_text, re.MULTILINE))
        if "bulleted list" in desc_lower or "bullet points" in desc_lower:
            return bool(re.search(r"^\s*[-*]\s+", output_text, re.MULTILINE))
        return None

    @staticmethod
    def _count_sentences(text: str) -> int:
        return len([s for s in re.split(r'[.!?]+', text) if s.strip()])

    @staticmethod
    def _extract_numbers(constraint_desc: str) -> list[int]:
        word_map = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
        }
        nums = [int(n) for n in re.findall(r'\d+', constraint_desc)]
        for word, val in word_map.items():
            if re.search(rf'\b{word}\b', constraint_desc.lower()):
                nums.append(val)
        return sorted(set(nums))

    def evaluate_length_constraint(self, output_text: str, constraint_desc: str) -> Tuple[Optional[bool], Optional[float]]:
        """Returns (passed, deviation_ratio). deviation_ratio is 0.0 when passed, >0 when failed."""
        desc_lower = constraint_desc.lower()
        is_char = "character" in desc_lower
        is_sentence = "sentence" in desc_lower
        is_paragraph = "paragraph" in desc_lower

        if is_sentence:
            count = self._count_sentences(output_text)
        elif is_char:
            count = len(output_text)
        elif is_paragraph:
            count = len([p for p in output_text.split("\n\n") if p.strip()])
        else:
            count = len(output_text.split())

        numbers = self._extract_numbers(constraint_desc)

        if len(numbers) == 2 and ("between" in desc_lower or "range" in desc_lower or "-" in desc_lower):
            lo, hi = sorted(numbers)
            passed = lo <= count <= hi
            target = (lo + hi) / 2
            deviation = 0.0 if passed else (abs(count - target) / target if target > 0 else 0.0)
            return passed, deviation

        if len(numbers) >= 1:
            limit = numbers[0]
            if any(k in desc_lower for k in ("limit", "maximum", "at most", "no more than")):
                passed = count <= limit
                deviation = 0.0 if passed else (max(0.0, (count - limit) / limit) if limit > 0 else 0.0)
                return passed, deviation
            if any(k in desc_lower for k in ("minimum", "at least", "no less than")):
                passed = count >= limit
                deviation = 0.0 if passed else (max(0.0, (limit - count) / limit) if limit > 0 else 0.0)
                return passed, deviation
            if any(k in desc_lower for k in ("exactly", "exact")):
                if is_char:
                    passed = count == limit
                else:
                    passed = (abs(count - limit) / limit <= 0.05) if limit > 0 else (count == 0)
                deviation = (abs(count - limit) / limit) if limit > 0 else 0.0
                return passed, (0.0 if passed else deviation)

        return None, None

    async def batch_evaluate_subjective(
        self,
        output_text: str,
        constraints: list[tuple[str, str, str]],  # (c_id, c_type, c_description)
        prompt_instruction: str,
    ) -> dict[str, bool]:
        if not constraints:
            return {}

        constraint_lines = "\n".join(
            f"{i + 1}. [{ctype}] {cdesc}"
            for i, (_, ctype, cdesc) in enumerate(constraints)
        )

        prompt = f"""You are evaluating whether an AI model's response satisfies a set of constraints.

Original instruction given to the model:
{prompt_instruction[:500]}

Model response:
{output_text[:3000]}

For each constraint below, determine if the model's response satisfies it.

Constraints:
{constraint_lines}

Reply ONLY with a JSON object in this exact format:
{{"results": [true, false, ...]}}

The array must have exactly {len(constraints)} boolean values, one per constraint in order."""

        response = await openrouter_chat(
            messages=[
                {"role": "system", "content": "You are an impartial evaluation judge. Reply only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            model=self.judge_model,
            api_key=self.api_key,
            max_tokens=500,
            temperature=0.0,
            semaphore=self.semaphore,
        )

        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1].strip()
                if clean.startswith("json"):
                    clean = clean[4:].strip()

            data = json.loads(clean)
            results_list = data.get("results", [])

            if len(results_list) != len(constraints):
                logger.warning(f"Judge returned {len(results_list)} results for {len(constraints)} constraints. Defaulting to False.")
                return {c_id: False for c_id, _, _ in constraints}

            return {c_id: bool(r) for (c_id, _, _), r in zip(constraints, results_list)}

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse judge response: {e}. Raw: {response[:200]}")
            return {c_id: False for c_id, _, _ in constraints}

    async def evaluate(self, prompt: Prompt, output: GeneratedOutput) -> EvaluationResult:
        logger.debug(f"Evaluating {prompt.id} for {output.model_name}")
        constraint_results: dict[str, bool] = {}
        length_deviations: dict[str, float] = {}
        subjective_batch: list[tuple[str, str, str]] = []

        for task in prompt.sub_tasks:
            for constraint in task.constraints:
                c_id = constraint.id

                if constraint.type == ConstraintType.FORMAT:
                    result = self.evaluate_format_constraint(output.output_text, constraint.description)
                    if result is not None:
                        constraint_results[c_id] = result
                        continue

                elif constraint.type == ConstraintType.LENGTH:
                    passed, deviation = self.evaluate_length_constraint(output.output_text, constraint.description)
                    if passed is not None:
                        constraint_results[c_id] = passed
                        if deviation is not None:
                            length_deviations[c_id] = deviation
                        continue

                subjective_batch.append((c_id, constraint.type.value, constraint.description))

        if subjective_batch:
            batch_results = await self.batch_evaluate_subjective(
                output.output_text, subjective_batch, prompt.full_prompt()
            )
            constraint_results.update(batch_results)

        total = len(constraint_results)
        satisfied = sum(1 for v in constraint_results.values() if v)
        rfr = satisfied / total if total > 0 else 1.0
        ifr = 1.0 if rfr == 1.0 else 0.0

        logger.debug(f"{prompt.id} [{output.model_name}]: RFR={rfr:.2f}, IFR={ifr:.2f}")

        return EvaluationResult(
            prompt_id=prompt.id,
            model_name=output.model_name,
            rfr_score=rfr,
            ifr_score=ifr,
            constraint_results=constraint_results,
            length_deviations=length_deviations,
        )
