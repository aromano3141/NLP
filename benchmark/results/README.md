# Evaluation Results

**Dataset:** 200 prompts — 80 English, 40 Chinese, 40 Arabic, 40 Hindi  
**Models evaluated:** gemini-2.5-flash, gpt-4o-mini, qwen3.5-flash-02-23, llama-4-scout  
**Judge model:** google/gemini-2.5-flash  
**Run date:** 2026-05-03

---

## Metrics

- **RFR (Requirement Following Ratio):** % of individual constraints satisfied across all prompts
- **IFR (Instruction Following Ratio):** % of prompts where every constraint was satisfied simultaneously

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

## Notes

- Qwen evaluated on 199/200 prompts due to 1 timeout failure
- `eval_report_799prompts_20260503_183714.json` — final results (gemini-2.5-flash)
- `eval_report_799prompts_20260503_174941.json` — intermediate run with gemini-2.5-flash-lite
- `eval_report_300prompts_original_20260502_182251.json` — original 300-prompt dataset run; English results valid, non-English results unreliable due to missing context in prompts
