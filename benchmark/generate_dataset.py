import asyncio
import json
import os
import logging
from src.schemas.constants import CORE_TASK_CATEGORIES, LANGUAGE_DISTRIBUTION
from src.pipeline.seed_generation import SeedGenerator
from src.pipeline.localization import LocalizationPipeline
from src.eval.validator import InstructionValidator
from src.schemas.models import Prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_dataset(dataset: list):
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/benchmark_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    logger.debug(f"Saved dataset with {len(dataset)} items to data/processed/benchmark_dataset.json")

async def generate_full_dataset():
    logger.info("Starting dataset generation process")
    generator = SeedGenerator(model_name="gemma4:31b-cloud")
    validator = InstructionValidator(model_name="deepseek-v4-flash:cloud")
    
    dataset = []
    generated_counts = {
        "English": 0,
        "Chinese": 0,
        "Arabic": 0,
        "Hindi": 0
    }
    
    for target_lang, target_count in LANGUAGE_DISTRIBUTION.items():
        if target_count <= 0:
            continue
            
        logger.info(f"--- Generating {target_count} unique prompts for {target_lang} ---")
        
        # Distribute targets evenly across core task categories
        category_targets = {cat: target_count // len(CORE_TASK_CATEGORIES) for cat in CORE_TASK_CATEGORIES}
        for i in range(target_count % len(CORE_TASK_CATEGORIES)):
            category_targets[CORE_TASK_CATEGORIES[i]] += 1
            
        # If the target language isn't English, prepare the localizer
        localizer = None
        if target_lang != "English":
            localizer = LocalizationPipeline(target_languages=[target_lang], model_name="gpt-oss:120b-cloud")
            
        for category, cat_target in category_targets.items():
            logger.info(f"Generating {cat_target} {target_lang} prompts for category: {category}")
            cat_count = 0
            attempts = 0
            
            while cat_count < cat_target:
                attempts += 1
                density = ["Low", "Medium", "High", "High","High"][cat_count % 5]
                logger.info(f"  Attempt {attempts} (Current valid: {cat_count}/{cat_target}, Density: {density})...")
                
                # Step 1: Always generate a brand new unique English seed
                seed_prompt = await generator.generate_seed_prompt(category, density)
                
                if not seed_prompt:
                    logger.warning(f"  [Invalid] Seed prompt generation failed schema parsing. Retrying.")
                    continue
                
                # Step 2: Validate the English seed
                is_valid, reason = await validator.extract_and_verify(seed_prompt)
                if not is_valid:
                    logger.warning(f"  [Invalid] Seed prompt failed validation: {reason}. Retrying.")
                    continue
                
                # Step 3: Handle target language
                if target_lang == "English":
                    dataset.append(seed_prompt.model_dump())
                    generated_counts["English"] += 1
                    cat_count += 1
                    logger.info(f"  [Success] Added unique English prompt.")
                    save_dataset(dataset)
                else:
                    # Localize the valid English seed to the target language
                    localized_prompts = await localizer.run_pipeline(seed_prompt)
                    
                    if not localized_prompts:
                        logger.warning(f"  [Invalid] Localization or back-translation failed for {target_lang}. Retrying with new seed.")
                        continue
                        
                    # We only passed [target_lang] so there should be exactly 1 result if successful
                    loc_prompt = localized_prompts[0]
                    dataset.append(loc_prompt.model_dump())
                    generated_counts[target_lang] += 1
                    cat_count += 1
                    logger.info(f"  [Success] Added unique {target_lang} prompt (Original English seed discarded).")
                    save_dataset(dataset)

    logger.info(f"Generation complete. Final distribution: {generated_counts}")

if __name__ == "__main__":
    # logger.info("Warning: Running this script will incur LLM API costs and take time.")
    asyncio.run(generate_full_dataset())