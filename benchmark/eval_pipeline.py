import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.schemas.models import Prompt
from src.eval.inference_runner import InferenceRunner
from src.eval.evaluator import Evaluator
from src.eval.aggregator import aggregate

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODELS = [
    "google/gemini-2.5-flash-lite",
    "openai/gpt-4o-mini",
    "qwen/qwen3.5-flash-02-23",
    "meta-llama/llama-4-scout",
]

JUDGE_MODEL = "google/gemini-2.5-flash"

DATASET_DIR = Path(__file__).parent.parent / "dataset"
DATASET_FILES = {
    "English": DATASET_DIR / "english.json",
    "Chinese": DATASET_DIR / "chinese.json",
    "Arabic":  DATASET_DIR / "arabic.json",
    "Hindi":   DATASET_DIR / "hindi.json",
}

BENCHMARK_FILE = DATASET_DIR / "benchmark_dataset.json"


def load_dataset() -> list[Prompt]:
    # Use combined benchmark file if it exists, otherwise fall back to per-language files
    if BENCHMARK_FILE.exists():
        with open(BENCHMARK_FILE, encoding="utf-8") as f:
            data = json.load(f)
        prompts = [Prompt(**item) for item in data]
        logger.info(f"Loaded {len(prompts)} prompts from benchmark_dataset.json")
        return prompts

    prompts = []
    for lang, path in DATASET_FILES.items():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            prompts.append(Prompt(**item))
        logger.info(f"Loaded {len(data)} {lang} prompts")
    return prompts


def save_results(report: dict, raw_results: dict, label: str = ""):
    from datetime import datetime
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}_{ts}" if label else f"_{ts}"
    with open(out_dir / f"eval_report{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    raw_out = {
        model: [r.model_dump() for r in results]
        for model, results in raw_results.items()
    }
    with open(out_dir / f"raw_results{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(raw_out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results/eval_report{suffix}.json and raw_results{suffix}.json")


def print_summary(report: dict):
    SEP = "=" * 80
    model_keys = [k for k in report if k != "cultural_gap_analysis"]

    print(f"\n{SEP}\nEVALUATION SUMMARY\n{SEP}")

    # Overall
    print(f"\n{'Model':<40} {'RFR':>8} {'IFR':>8} {'N':>6}")
    print("-" * 65)
    for m in model_keys:
        o = report[m]["overall"]
        print(f"{m:<40} {o['rfr']:>8.3f} {o['ifr']:>8.3f} {o['n']:>6}")

    # By language
    langs = ["English", "Chinese", "Arabic", "Hindi"]
    print(f"\nRFR by Language")
    print(f"{'Model':<40} " + " ".join(f"{l:>10}" for l in langs))
    print("-" * (40 + 11 * len(langs)))
    for m in model_keys:
        row = f"{m:<40}"
        for lang in langs:
            rfr = report[m]["by_language"].get(lang, {}).get("rfr")
            row += f" {rfr:>10.3f}" if rfr is not None else f" {'N/A':>10}"
        print(row)

    # By constraint type
    ctypes = ["Content Constraint", "Format Constraint", "Style Constraint", "Situation Constraint", "Length Constraint"]
    ctype_short = ["Content", "Format", "Style", "Situation", "Length"]
    print(f"\nConstraint Type Accuracy")
    print(f"{'Model':<40} " + " ".join(f"{s:>10}" for s in ctype_short))
    print("-" * (40 + 11 * len(ctypes)))
    for m in model_keys:
        row = f"{m:<40}"
        for ct in ctypes:
            acc = report[m]["by_constraint_type"].get(ct, {}).get("accuracy")
            row += f" {acc:>10.3f}" if acc is not None else f" {'N/A':>10}"
        print(row)

    # Cultural gap
    print(f"\nCultural Gap Analysis (RFR per language)")
    print(f"{'Language':<12} " + " ".join(f"{m.split('/')[-1]:>22}" for m in model_keys))
    print("-" * (12 + 23 * len(model_keys)))
    for lang in langs:
        row = f"{lang:<12}"
        for m in model_keys:
            rfr = report["cultural_gap_analysis"][lang].get(m)
            row += f" {rfr:>22.3f}" if rfr is not None else f" {'N/A':>22}"
        print(row)

    print(f"\n{SEP}\n")


async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env")

    prompts = load_dataset()
    logger.info(f"Total prompts: {len(prompts)}")

    # Phase 1: Inference — send prompts to all 4 models
    runner = InferenceRunner(
        models=MODELS,
        api_key=api_key,
        concurrency=5,
        checkpoint_dir="results/responses",
    )
    model_outputs = await runner.run(prompts)

    # Phase 2: Evaluation — score each response against its constraints
    evaluator = Evaluator(
        judge_model_name=JUDGE_MODEL,
        api_key=api_key,
        concurrency=10,
    )
    prompt_map = {p.id: p for p in prompts}
    eval_results: dict[str, list] = {}

    for model, outputs in model_outputs.items():
        logger.info(f"Evaluating responses from {model}...")
        tasks = [
            evaluator.evaluate(prompt_map[o.prompt_id], o)
            for o in outputs
            if o.output_text and o.prompt_id in prompt_map
        ]
        results = await asyncio.gather(*tasks)
        eval_results[model] = list(results)
        logger.info(f"[{model}] Evaluated {len(results)} prompts")

    # Phase 3: Aggregate and report
    report = aggregate(eval_results, prompts)
    n = sum(len(v) for v in eval_results.values())
    save_results(report, eval_results, label=f"{n}prompts")
    print_summary(report)


if __name__ == "__main__":
    asyncio.run(main())
