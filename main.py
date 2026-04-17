import argparse
import asyncio
import json

import logging
from typing import List, Dict
from dotenv import load_dotenv

import pandas as pd

from data_models import EvaluationResult
from dataset_builder import generate_mock_dataset
from llm_client import LLMClient
from evaluator import Evaluator
from metrics_engine import calculate_cla, calculate_ila, calculate_cdi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def run_pipeline(model_to_eval: str, judge_model: str, use_mock: bool = True):
    """
    Orchestrates the entire evaluation pipeline.
    """
    logger.info("Starting COMP-EVAL Pipeline")
    logger.info(f"Target Model: {model_to_eval}")
    logger.info(f"Judge Model: {judge_model}")

    if use_mock:
        logger.info("Loading mock dataset...")
        instructions = generate_mock_dataset()
    else:
        # Placeholder for loading actual YAML/JSON dataset
        logger.error("Non-mock datasets are not yet implemented.")
        return

    client = LLMClient(default_model=model_to_eval)
    evaluator = Evaluator(judge_model=judge_model)

    # 1. Generate Responses
    logger.info("Generating responses...")
    generations: Dict[str, str] = {}
    for inst in instructions:
        # In a real pipeline, we'd gather tasks and run asyncio.gather for concurrency
        # Running sequentially here to respect rate limits if OPENAI_API_KEY is not set
        logger.info(f"Generating for instruction: {inst.id}")
        resp = await client.generate_response(prompt=inst.base_prompt)
        generations[inst.id] = resp or "ERROR: No response generated."

    # 2. Evaluate
    logger.info("Evaluating responses...")
    results: List[EvaluationResult] = []
    for inst in instructions:
        output_text = generations[inst.id]
        logger.info(f"Evaluating output for: {inst.id}")
        result = await evaluator.evaluate_output(instruction=inst, output_text=output_text)
        results.append(result)

    # 3. Compute Metrics
    logger.info("Computing metrics...")
    cla_list = []
    ila_list = []
    total_length_score = 0.0
    length_score_count = 0

    export_data = []

    for inst, res in zip(instructions, results):
        cla = calculate_cla(res.constraint_results)
        ila = calculate_ila(cla)

        cla_list.append(cla)
        ila_list.append(ila)

        if res.length_score is not None:
            total_length_score += res.length_score
            length_score_count += 1

        export_data.append({
            "instruction_id": inst.id,
            "target_language": inst.target_language,
            "execution_mode": inst.execution_mode.value,
            "prompt": inst.base_prompt,
            "response": generations[inst.id],
            "holistic_pass": res.holistic_pass,
            "cla": cla,
            "ila": ila,
            "length_score": res.length_score,
            "constraint_details": json.dumps(res.constraint_results)
        })

    cdi = calculate_cdi(cla_list, ila_list)
    avg_cla = sum(cla_list) / len(cla_list) if cla_list else 0.0
    avg_ila = sum(ila_list) / len(ila_list) if ila_list else 0.0
    avg_bls = total_length_score / length_score_count if length_score_count > 0 else 0.0

    metrics = {
        "Target_Model": model_to_eval,
        "Total_Instructions": len(instructions),
        "Average_CLA": avg_cla,
        "Average_ILA": avg_ila,
        "CDI": cdi,
        "Average_BLS": avg_bls
    }

    logger.info("--- AGGREGATE METRICS ---")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    # 4. Export
    logger.info("Exporting results...")

    df = pd.DataFrame(export_data)
    df.to_csv("report.csv", index=False)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Pipeline completed. See report.csv and metrics.json.")

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="COMP-EVAL CLI")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Target model to evaluate")
    parser.add_argument("--judge", type=str, default="gpt-4o", help="Judge model to use for evaluation")
    parser.add_argument("--mock", action="store_true", default=True, help="Use the mock dataset")

    args = parser.parse_args()

    asyncio.run(run_pipeline(args.model, args.judge, args.mock))

if __name__ == "__main__":
    main()
