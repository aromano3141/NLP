import asyncio
import os
import json
from src.pipeline.seed_generation import SeedGenerator
from src.pipeline.localization import LocalizationPipeline
from src.eval.validator import InstructionValidator
from src.eval.evaluator import Evaluator
from src.schemas.models import GeneratedOutput

async def test_pipeline():
    print("Starting pipeline test...")
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("DEEPSEEK_API_KEY"):
        print("Please set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env")
        # For testing purposes, we can use a mock or require the user to set it
        print("Note: The pipeline will fail if no valid API key is configured.")

    # We'll use gpt-4o as default if available
    generator = SeedGenerator(model_name="gpt-4o")
    
    print("\n[1] Generating Seed Prompt (Category: Classification, Density: Low)...")
    seed_prompt = await generator.generate_seed_prompt("Classification", density="Low")
    print(f"Generated Seed Prompt ID: {seed_prompt.id}")
    print(f"Instruction: {seed_prompt.instruction}")
    for task in seed_prompt.sub_tasks:
        print(f"  Task: {task.instruction}")
        for c in task.constraints:
            print(f"    - [{c.type.value}] {c.description}")

    print("\n[2] Validating Seed Prompt...")
    validator = InstructionValidator(model_name="gpt-4o")
    is_valid, reason = await validator.extract_and_verify(seed_prompt)
    print(f"Is Valid: {is_valid}, Reason: {reason}")
    
    if not is_valid:
        print("Skipping translation and evaluation due to invalid prompt.")
        return

    print("\n[3] Localizing Prompt to Hindi...")
    localizer = LocalizationPipeline(target_languages=["Hindi"], model_name="gpt-4o")
    localized_prompts = await localizer.run_pipeline(seed_prompt)
    if not localized_prompts:
        print("Localization failed.")
        return
        
    hindi_prompt = localized_prompts[0]
    print(f"Hindi Prompt ID: {hindi_prompt.id}")
    print(f"Hindi Instruction: {hindi_prompt.instruction}")

    print("\n[4] Simulating Model Generation (Mock)...")
    mock_output_text = "Here is the response. | Entity | Type |\n|---|---|\n| Apple | Fruit |" 
    mock_output = GeneratedOutput(
        prompt_id=seed_prompt.id,
        model_name="test-model-1",
        output_text=mock_output_text
    )

    print("\n[5] Evaluating Output...")
    evaluator = Evaluator(judge_model_name="gpt-4o")
    eval_result = await evaluator.evaluate(seed_prompt, mock_output)
    print(f"Evaluation Results for test-model-1:")
    print(f"RFR Score: {eval_result.rfr_score}")
    print(f"IFR Score: {eval_result.ifr_score}")
    print(f"Constraints Passed: {eval_result.constraint_results}")

    print("\nPipeline test complete.")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
