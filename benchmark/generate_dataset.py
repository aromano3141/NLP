import asyncio
import json
import os
from NLP.benchmark.src.schemas.constants import CORE_TASK_CATEGORIES, LANGUAGE_DISTRIBUTION
from NLP.benchmark.src.pipeline.seed_generation import SeedGenerator
from NLP.benchmark.src.pipeline.localization import LocalizationPipeline
from NLP.benchmark.src.eval.validator import InstructionValidator

async def generate_full_dataset():
    """
    Orchestrates the generation of the 300 prompts dataset.
    This script will take a significant amount of time and API calls to run.
    """
    target_english_prompts = LANGUAGE_DISTRIBUTION["English"] # 101
    
    # We will distribute the 101 English prompts across the 7 categories
    prompts_per_category = target_english_prompts // len(CORE_TASK_CATEGORIES)
    
    generator = SeedGenerator(model_name="gpt-4o")
    validator = InstructionValidator(model_name="gpt-4o")
    localizer = LocalizationPipeline(target_languages=["Chinese", "Arabic", "Hindi"], model_name="gpt-4o")
    
    dataset = []
    
    for category in CORE_TASK_CATEGORIES:
        print(f"Generating prompts for category: {category}")
        for i in range(prompts_per_category):
            print(f"  Generating seed {i+1}/{prompts_per_category}...")
            # Vary density: simplistic round-robin
            density = ["Low", "Medium", "High"][i % 3]
            
            seed_prompt = await generator.generate_seed_prompt(category, density)
            
            # Validate
            is_valid, reason = await validator.extract_and_verify(seed_prompt)
            if not is_valid:
                print(f"  [Invalid] Seed prompt failed validation: {reason}. Skipping.")
                continue
                
            dataset.append(seed_prompt.model_dump())
            
            # Localize
            # Note: The distribution target is 75 Chinese, 62 Arabic, 62 Hindi.
            # Here we naively generate all 3 for each valid English prompt for demonstration,
            # but you would add logic to cap the localized prompts to hit the exact matrix.
            localized_prompts = await localizer.run_pipeline(seed_prompt)
            for lp in localized_prompts:
                dataset.append(lp.model_dump())
                
    # Save dataset
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/benchmark_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print(f"Generation complete. Total prompts generated: {len(dataset)}")

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set in .env before running
    print("Warning: Running this script will incur LLM API costs and take time.")
    # asyncio.run(generate_full_dataset()) # Uncomment to actually run
