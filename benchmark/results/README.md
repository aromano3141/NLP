# Evaluation Results

**Dataset:** 200 prompts — 80 English, 40 Chinese, 40 Arabic, 40 Hindi  
**Models evaluated:** gemini-2.5-flash, gpt-4o-mini, qwen3.5-flash-02-23, llama-4-scout  
**Judge model:** google/gemini-2.5-flash  
**Run date:** 2026-05-03

---

## Metrics

- **RFR (Requirement Following Ratio):** % of individual constraints satisfied across all prompts
- **IFR (Instruction Following Ratio):** % of prompts where every single constraint was satisfied simultaneously — the primary measure of compositional instruction following

---

## Evaluation Methodology

Constraint scoring uses a two-track approach:

**Deterministic (no LLM):**
- Format constraints — regex checks for markdown tables, numbered lists, bullet points, JSON structure
- Length constraints — word count, sentence count, character count, paragraph count with deviation scoring

**LLM-as-Judge:**
- Content constraints — e.g. "must mention X", "must reference rule Y"
- Style constraints — tone, emotion, formality
- Situation constraints — role-play, audience, context-setting
- Format/Length fallback — if the regex/counter can't parse the constraint description, falls back to judge

Roughly 75-80% of constraint checks go through the judge, 20-25% are deterministic. The judge receives only the constraint description and the model response — it is blind to which model generated the response, avoiding self-evaluation bias.

All subjective constraints for a single prompt are batched into one judge call (rather than one call per constraint) for efficiency.

---

## Overall

| Model | RFR | IFR | N |
|---|---|---|---|
| meta-llama/llama-4-scout | 0.600 | 0.105 | 200 |
| openai/gpt-4o-mini | 0.593 | 0.110 | 200 |
| qwen/qwen3.5-flash-02-23 | 0.592 | 0.111 | 199 |
| google/gemini-2.5-flash | 0.475 | 0.070 | 200 |

## RFR by Language

| Language | Gemini-2.5 | GPT-4o-Mini | Qwen3.5 | Llama-4-Scout |
|---|---|---|---|---|
| English | 0.418 | 0.504 | 0.494 | 0.578 |
| Chinese | 0.644 | 0.717 | 0.785 | 0.645 |
| Arabic | 0.423 | 0.611 | 0.620 | 0.579 |
| Hindi | 0.472 | 0.628 | 0.563 | 0.620 |

## Constraint Type Accuracy

| Type | Gemini-2.5 | GPT-4o-Mini | Qwen3.5 | Llama-4-Scout |
|---|---|---|---|---|
| Content | 0.526 | 0.622 | 0.629 | 0.642 |
| Format | 0.470 | 0.575 | 0.572 | 0.552 |
| Style | 0.468 | 0.579 | 0.554 | 0.600 |
| Situation | 0.285 | 0.461 | 0.436 | 0.461 |
| Length | 0.217 | 0.337 | 0.301 | 0.349 |

---

## Files

- `eval_report_799prompts_20260503_183714.json` — final aggregated results (RFR/IFR by model, language, category, constraint type)
- `raw_results_799prompts_20260503_183714.json` — per-prompt, per-constraint pass/fail results for every model
- `eval_report_300prompts_original_20260502_182251.json` — original run, see note below

> **Note on file naming:** The `799` in filenames refers to total model evaluations across all 4 models (200+200+199+200), not the number of prompts.

---

## Notes

### Why Qwen has 199 prompts
`qwen/qwen3.5-flash-02-23` dropped 1 prompt in both runs due to repeated API timeouts (90s timeout, 3 retries exhausted). All other models completed 200/200. This model showed reliability issues across both runs and may be worth replacing in future evaluations.

### Why the 300-prompt original run is unreliable for non-English
The original dataset (`english.json`, `chinese.json`, `arabic.json`, `hindi.json`) stored background context in a `reading_materials` field separately from the `instruction` field. For English prompts, the generation pipeline embedded `reading_materials` directly into `instruction`, so the model received the full context. For Chinese, Arabic, and Hindi, `reading_materials` was stored separately and was never sent to the model — roughly 90% of non-English prompts were evaluated without their background context. Models correctly identified something was missing and responded asking for more input, resulting in near-zero scores. English results from that run are valid. The new 200-prompt dataset uses a unified `base_information` field that is always sent to the model.

### Recommended judge models for future runs
If re-running with a different judge, good options that are not in the evaluated set:
- `anthropic/claude-haiku-4-5` — strong instruction-following evaluation, not in evaluated set
- `openai/gpt-4o` — strong and reliable, higher cost
- `meta-llama/llama-3.3-70b-instruct` — capable, cheap, not in evaluated set
