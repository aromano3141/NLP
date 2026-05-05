import math
from collections import defaultdict
from src.schemas.models import EvaluationResult, Prompt


def _scores(rs: list[EvaluationResult]) -> dict:
    if not rs:
        return {"rfr": 0.0, "ifr": 0.0, "gap": 0.0, "n": 0}
    rfr = round(sum(r.rfr_score for r in rs) / len(rs), 4)
    ifr = round(sum(r.ifr_score for r in rs) / len(rs), 4)
    return {"rfr": rfr, "ifr": ifr, "gap": round(rfr - ifr, 4), "n": len(rs)}


def _length_score(ld: float) -> float:
    """Non-linear length score from LIFEBENCH. Maps deviation to [0, 100].
    LD=0 → LS=100, larger deviation → lower score."""
    k = 2.0
    return round(100 * math.exp(-k * abs(ld)), 4)


def aggregate(
    eval_results: dict[str, list[EvaluationResult]],
    prompts: list[Prompt],
) -> dict:
    prompt_map = {p.id: p for p in prompts}
    report = {}

    for model, results in eval_results.items():
        by_language: dict[str, list[EvaluationResult]] = defaultdict(list)
        by_category: dict[str, list[EvaluationResult]] = defaultdict(list)
        by_ctype: dict[str, dict] = defaultdict(lambda: {"passed": 0, "total": 0})
        length_devs: list[float] = []
        length_scores: list[float] = []

        for result in results:
            prompt = prompt_map.get(result.prompt_id)
            if not prompt:
                continue

            by_language[prompt.language].append(result)
            by_category[prompt.core_task_category].append(result)

            for task in prompt.sub_tasks:
                for c in task.constraints:
                    ct = c.type.value
                    passed = result.constraint_results.get(c.id, False)
                    by_ctype[ct]["total"] += 1
                    if passed:
                        by_ctype[ct]["passed"] += 1

            for dev in result.length_deviations.values():
                length_devs.append(dev)
                length_scores.append(_length_score(dev))

        report[model] = {
            "overall": _scores(results),
            "by_language": {lang: _scores(rs) for lang, rs in by_language.items()},
            "by_category": {cat: _scores(rs) for cat, rs in by_category.items()},
            "by_constraint_type": {
                ct: {
                    "accuracy": round(v["passed"] / v["total"], 4) if v["total"] else 0.0,
                    "passed": v["passed"],
                    "total": v["total"],
                }
                for ct, v in by_ctype.items()
            },
            # LIFEBENCH metrics for length constraints
            "avg_length_deviation": round(sum(length_devs) / len(length_devs), 4) if length_devs else 0.0,
            "avg_length_score": round(sum(length_scores) / len(length_scores), 4) if length_scores else 0.0,
        }

    model_keys = list(eval_results.keys())
    langs = ["English", "Chinese", "Arabic", "Hindi"]

    # RFR/IFR per language per model (cultural gap analysis from XIFBench)
    report["cultural_gap_analysis"] = {
        lang: {
            model: {
                "rfr": report[model]["by_language"].get(lang, {}).get("rfr"),
                "ifr": report[model]["by_language"].get(lang, {}).get("ifr"),
                "gap": report[model]["by_language"].get(lang, {}).get("gap"),
            }
            for model in model_keys
        }
        for lang in langs
    }

    return report
