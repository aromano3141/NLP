# Evaluation Results

**Dataset:** 199 prompts — ~80 English, ~40 Chinese, ~40 Arabic, ~40 Hindi  
**Models evaluated:** gemini-2.5-flash, gpt-4o-mini, qwen3.5-flash-02-23, llama-4-scout  
**Judge model:** anthropic/claude-haiku-4-5  
**Run date:** 2026-05-04

---

## Metrics

Metrics drawn from related works (XIFBench, EIFBENCH, LIFEBENCH):

- **RFR (Requirement Following Ratio):** % of individual constraints satisfied across all prompts — from XIFBench
- **IFR (Instruction Following Ratio):** % of prompts where every single constraint was satisfied simultaneously — primary measure of compositional following, from XIFBench
- **GAP (RFR-IFR Gap):** RFR minus IFR per language — reveals compositional collapse: how much performance drops when all constraints must be satisfied simultaneously vs. individually, from XIFBench
- **LD (Length Deviation):** `(output_length - target_length) / target_length` — signed ratio showing over/under-generation magnitude, from LIFEBENCH
- **LS (Length Score):** `100 × e^(-2 × |LD|)` — non-linear score mapping deviation to [0,100] so outliers don't skew averages, from LIFEBENCH

---

## Evaluation Methodology

Constraint scoring uses a two-track approach:

**Deterministic (no LLM):**
- Format constraints — regex checks for markdown tables, numbered lists, bullet points, JSON structure
- Length constraints — word count, sentence count, character count, paragraph count with deviation scoring

**LLM-as-Judge (`anthropic/claude-haiku-4-5`):**
- Content constraints — e.g. "must mention X", "must reference rule Y"
- Style constraints — tone, emotion, formality
- Situation constraints — role-play, audience, context-setting
- Format/Length fallback — if regex/counter cannot parse the constraint description

Roughly 75-80% of constraint checks go through the judge, 20-25% are deterministic. The judge receives only the constraint description and the model response — it is blind to which model generated the response, avoiding self-evaluation bias. All subjective constraints for a single prompt are batched into one judge call for efficiency.

---

## Overall

| Model | RFR | IFR | N |
|---|---|---|---|
| meta-llama/llama-4-scout | 0.649 | 0.070 | 199 |
| openai/gpt-4o-mini | 0.643 | 0.101 | 199 |
| qwen/qwen3.5-flash-02-23 | 0.606 | 0.080 | 199 |
| google/gemini-2.5-flash | 0.490 | 0.040 | 199 |

## RFR by Language

| Language | Gemini-2.5 | GPT-4o-Mini | Qwen3.5 | Llama-4-Scout |
|---|---|---|---|---|
| English | 0.447 | 0.572 | 0.563 | 0.635 |
| Chinese | 0.653 | 0.750 | 0.767 | 0.668 |
| Arabic | 0.410 | 0.678 | 0.573 | 0.668 |
| Hindi | 0.492 | 0.644 | 0.566 | 0.639 |

## Constraint Type Accuracy

| Type | Gemini-2.5 | GPT-4o-Mini | Qwen3.5 | Llama-4-Scout |
|---|---|---|---|---|
| Content | 0.504 | 0.655 | 0.608 | 0.658 |
| Format | 0.522 | 0.676 | 0.665 | 0.647 |
| Style | 0.511 | 0.652 | 0.644 | 0.682 |
| Situation | 0.309 | 0.551 | 0.467 | 0.624 |
| Length | 0.145 | 0.301 | 0.229 | 0.337 |

---

## RFR-IFR Gap by Language

High gap = model follows individual constraints but collapses when all must be satisfied simultaneously.

| Language | Gemini-2.5 | GPT-4o-Mini | Qwen3.5 | Llama-4-Scout |
|---|---|---|---|---|
| English | 0.409 | 0.509 | 0.525 | 0.597 |
| Chinese | 0.603 | 0.575 | 0.567 | 0.568 |
| Arabic | 0.385 | 0.553 | 0.498 | 0.594 |
| Hindi | 0.442 | 0.569 | 0.516 | 0.539 |

---

## Files

- `eval_report_796prompts_20260504_223340.json` — final aggregated results with all metrics (RFR, IFR, GAP, LD, LS)
- `raw_results_796prompts_20260504_215022.json` — per-prompt, per-constraint pass/fail for every model
- `eval_report_300prompts_original_20260502_182251.json` — original run, see notes below

> **Note on file naming:** The number in filenames refers to total model evaluations (199 × 4 = 796), not the number of prompts.

---

## Notes

### Why 199 prompts instead of 200
One prompt (`a01c497a-795f-4a24-9203-8a078f7758df`) consistently caused API timeouts for `qwen/qwen3.5-flash-02-23` across multiple runs with no successful response after 3 retries. To keep all 4 models evaluated on identical prompts for a fair comparison, this prompt was excluded from all models.

### Why the 300-prompt original run is unreliable for non-English
The original dataset stored background context in a `reading_materials` field separately from the `instruction` field. For English, the pipeline embedded this context directly into `instruction`. For Chinese, Arabic, and Hindi (~90% of prompts), context was stored separately and never sent to the model — models responded asking for more input, resulting in near-zero scores. English results from that run are valid. The new 200-prompt dataset uses a unified `base_information` field that is always included.

