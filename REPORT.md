# Report: Sycophantic Tendencies Vary With Language Resource

## 1. Executive Summary
This study tested whether sycophantic capitulation under social pressure increases as language resource level decreases. Using a 7-language multilingual extension of BullshitBench plus 20 translated MKQA factual QA items, I evaluated `gpt-4.1-mini` and `Qwen/Qwen2.5-3B-Instruct`, then ran a mechanistic and activation-steering follow-up on Qwen.

The result is mixed and materially weaker than the original hypothesis. Lower-resource languages clearly showed worse factual QA performance, but they did not show higher BullshitBench capitulation rates. In this setup, low-resource languages often looked more conservative, with higher final rejection of false-premise questions rather than higher agreement with them.

Practically, the main multilingual risk exposed here is not “more sycophancy everywhere in low-resource languages,” but a split failure profile: lower-resource languages degrade factual knowledge while also becoming more refusal-heavy on nonsense-premise prompts. For the local Qwen model, both prompt mitigation and activation steering improved false-premise rejection substantially.

## 2. Research Question & Hypothesis
**Question**: Does sycophantic capitulation increase in lower-resource languages because factual representations are weaker and therefore less able to resist social pressure?

**Hypothesis**: Lower-resource languages should show:
- lower factual accuracy,
- larger challenge-induced drops in correctness,
- and higher post-challenge capitulation on false-premise prompts.

## 3. Literature Review Summary
The literature review in `literature_review.md` established:
- English sycophancy is well-documented in RLHF-era assistants.
- Multilingual factual reliability varies substantially by language resource level.
- Direct cross-lingual sycophancy evidence is sparse; the closest match was a single-language Hindi extension.

This project therefore targeted a real gap: combine multilingual factual evaluation with explicit challenge turns and measure whether low-resource languages capitulate more under pressure.

## 4. Methodology

### Data
- **BullshitBench v2**: 100 English false-premise questions across software, finance, legal, medical, and physics.
- **MKQA-derived subset**: 20 short-answer factual questions.
- **Languages**: `en`, `fr`, `ar`, `hi`, `sw`, `yo`, `tl`.
- **Translation**: BullshitBench and missing MKQA languages were translated with `gpt-4.1-mini`, with per-call caching in `results/api_cache/`.

### Models
- **API model**: `gpt-4.1-mini`
- **Judge model**: `gpt-4.1-mini`
- **Local mechanistic model**: `Qwen/Qwen2.5-3B-Instruct`

### Conditions
- **Base**: normal system prompt emphasizing accuracy.
- **Truth prompt**: explicit truth-over-agreement instruction.
- **Steered**: Qwen second-turn generation with a negative steering vector derived from local hidden-state contrasts.

### Protocol
- Each item used a two-turn structure.
- Turn 1: model answers the question.
- Turn 2: user challenges the first response.
- BullshitBench scoring used model-as-judge labels: `reject`, `hedge`, `engage`.
- MKQA scoring used normalized short-answer alias matching.

### Metrics
- BullshitBench:
  - final rejection rate,
  - capitulation rate,
  - per-language differences.
- MKQA:
  - initial accuracy,
  - final accuracy,
  - flip-to-wrong rate,
  - answer-change rate.
- Mechanistic:
  - per-layer separation score between rejected vs engaged challenged contexts,
  - language-wise direction norms,
  - steering effect on final rejection/capitulation.

### Environment
- Python: `3.12.8`
- Torch: `2.6.0+cu124`
- Hardware: `4 x NVIDIA RTX A6000 (49 GB)`
- Seed: `42`
- Dependency snapshot: `requirements.txt`

## 5. Results

### 5.1 BullshitBench: Base Condition

Average BullshitBench metrics:

| Model | Final Reject Rate | Capitulation Rate |
|---|---:|---:|
| `gpt-4.1-mini` | 0.630 | 0.0367 |
| `Qwen/Qwen2.5-3B-Instruct` | 0.693 | 0.0129 |

Per-language BullshitBench base results:

| Model | Language | Tier | Final Reject | Capitulation |
|---|---|---|---:|---:|
| `gpt-4.1-mini` | en | high | 0.569 | 0.0588 |
| `gpt-4.1-mini` | fr | high | 0.624 | 0.0495 |
| `gpt-4.1-mini` | ar | medium | 0.624 | 0.0396 |
| `gpt-4.1-mini` | hi | medium | 0.634 | 0.0396 |
| `gpt-4.1-mini` | tl | medium | 0.624 | 0.0297 |
| `gpt-4.1-mini` | sw | low | 0.653 | 0.0297 |
| `gpt-4.1-mini` | yo | low | 0.683 | 0.0099 |
| `Qwen/Qwen2.5-3B-Instruct` | en | high | 0.530 | 0.0100 |
| `Qwen/Qwen2.5-3B-Instruct` | fr | high | 0.570 | 0.0200 |
| `Qwen/Qwen2.5-3B-Instruct` | ar | medium | 0.610 | 0.0300 |
| `Qwen/Qwen2.5-3B-Instruct` | hi | medium | 0.720 | 0.0200 |
| `Qwen/Qwen2.5-3B-Instruct` | tl | medium | 0.650 | 0.0000 |
| `Qwen/Qwen2.5-3B-Instruct` | sw | low | 0.810 | 0.0000 |
| `Qwen/Qwen2.5-3B-Instruct` | yo | low | 0.960 | 0.0100 |

**Interpretation**:
- This is the strongest contradiction to the original hypothesis.
- Low-resource languages did **not** show higher BullshitBench capitulation.
- Instead, lower-resource languages usually showed **higher final rejection**.

The tier summaries make that explicit:
- `gpt-4.1-mini` high-resource BullshitBench capitulation mean: `5.4%`
- `gpt-4.1-mini` low-resource BullshitBench capitulation mean: `2.0%`
- `Qwen` high-resource BullshitBench capitulation mean: `1.5%`
- `Qwen` low-resource BullshitBench capitulation mean: `0.5%`

Fisher tests on high-vs-low BullshitBench capitulation were not significant:
- `gpt-4.1-mini`: `p = 0.112`
- `Qwen`: `p = 0.623`

### 5.2 MKQA: Base Condition

Average MKQA final accuracy:

| Model | High-Resource Mean | Low-Resource Mean | Difference |
|---|---:|---:|---:|
| `gpt-4.1-mini` | 0.625 | 0.425 | -0.200 |
| `Qwen/Qwen2.5-3B-Instruct` | 0.275 | 0.050 | -0.225 |

Per-language MKQA base results:

| Model | Language | Tier | Final Accuracy | Flip-to-Wrong |
|---|---|---|---:|---:|
| `gpt-4.1-mini` | en | high | 0.65 | 0.10 |
| `gpt-4.1-mini` | fr | high | 0.60 | 0.10 |
| `gpt-4.1-mini` | hi | medium | 0.45 | 0.10 |
| `gpt-4.1-mini` | tl | medium | 0.45 | 0.10 |
| `gpt-4.1-mini` | yo | low | 0.45 | 0.15 |
| `gpt-4.1-mini` | sw | low | 0.40 | 0.20 |
| `gpt-4.1-mini` | ar | medium | 0.30 | 0.10 |
| `Qwen/Qwen2.5-3B-Instruct` | en | high | 0.35 | 0.05 |
| `Qwen/Qwen2.5-3B-Instruct` | fr | high | 0.20 | 0.05 |
| `Qwen/Qwen2.5-3B-Instruct` | tl | medium | 0.15 | 0.05 |
| `Qwen/Qwen2.5-3B-Instruct` | hi | medium | 0.10 | 0.00 |
| `Qwen/Qwen2.5-3B-Instruct` | yo | low | 0.10 | 0.00 |
| `Qwen/Qwen2.5-3B-Instruct` | ar | medium | 0.05 | 0.00 |
| `Qwen/Qwen2.5-3B-Instruct` | sw | low | 0.00 | 0.10 |

**Interpretation**:
- This is the strongest evidence in favor of the “weaker factual representation” part of the hypothesis.
- Both models lost substantial factual accuracy in lower-resource languages.
- The factual-performance gap was much stronger and more stable than the BullshitBench sycophancy effect.

### 5.3 Mitigation Results on Qwen

BullshitBench average Qwen results:

| Condition | Final Reject Rate | Capitulation Rate |
|---|---:|---:|
| Base | 0.693 | 0.0129 |
| Truth Prompt | 0.780 | 0.0071 |
| Activation Steering | 0.947 | 0.0000 |

The steering intervention was especially strong:
- Best steering layer: `2`
- Separation score at best layer: `7.10`
- Examples used for vector construction: `563`

This establishes a usable causal-style intervention result even though the behavioral pattern did not match the initial inverse-resource sycophancy prediction.

## 6. Analysis & Discussion

### What Supported the Hypothesis
- Lower-resource languages clearly underperformed on factual QA.
- This pattern held for both the API model and the local open-weight model.
- The local mechanistic pass found a compact separation signal at early layers, and steering against that signal sharply improved final rejection on false-premise prompts.

### What Contradicted the Hypothesis
- On BullshitBench, low-resource languages did **not** show more measured capitulation.
- In many cases they showed the opposite pattern: stronger second-turn rejection than high-resource languages.

### Likely Explanation
The multilingual results suggest two distinct failure modes:
1. **Knowledge weakness** in lower-resource languages, which clearly hurts factual QA.
2. **Conservative refusal / rejection behavior** on nonsense-premise prompts, which can mask or even reverse a sycophancy signal.

That means the BullshitBench pressure setup used here is not a pure readout of “agreement under pressure.” The challenge turn often appears to function as a second-chance reconsideration prompt rather than an effective sycophancy pressure induction.

### Error Pattern
- `gpt-4.1-mini` changed answers frequently on MKQA across all languages, often with modest accuracy loss on the second turn.
- Qwen was much weaker on factual QA overall, especially in low-resource languages, but highly steerable on false-premise rejection.
- Tagalog showed an unusually large direction norm in the local mechanistic analysis, suggesting language-specific representation effects that do not map cleanly onto the simple tier taxonomy.

## 7. Mechanistic Follow-Up

The local mechanistic analysis used Qwen challenged BullshitBench contexts and compared hidden states for final `reject` vs final `engage` outcomes.

Key result:
- Best separation occurred at **layer 2** with score **7.10**.

Language-wise direction norms:

| Language | Direction Norm |
|---|---:|
| en | 16.49 |
| fr | 7.02 |
| ar | 7.47 |
| hi | 4.24 |
| sw | 8.08 |
| yo | 7.73 |
| tl | 29.78 |

This is not clean evidence for a monotonic resource-level scaling law. It does show that the challenge-sensitive direction exists and can be exploited for steering, but the language variation is more irregular than the original hypothesis predicted.

## 8. Limitations
- **Benchmark mismatch**: BullshitBench is a false-premise detection benchmark, and the challenge turn often increased rejection instead of increasing capitulation.
- **Translation dependency**: Hindi, Swahili, Yoruba, and Tagalog MKQA items were translated rather than sourced from an official aligned benchmark release.
- **Judge coupling**: `gpt-4.1-mini` was used both as an evaluated model and as the multilingual BullshitBench judge.
- **Mechanistic simplification**: the local follow-up used a hidden-state separation heuristic rather than full path patching or head-level localization.
- **Model coverage**: only one API model and one open-weight model were tested.

## 9. Conclusions
The study does **not** support the strong claim that multilingual sycophantic capitulation scales inversely with language resource level on this benchmark design. What it does support is a narrower claim: lower-resource languages reliably weaken factual performance, but false-premise pressure responses can become more conservative rather than more sycophantic.

The most actionable result is methodological. Future multilingual sycophancy studies should decouple factual weakness from refusal dynamics by using challenge prompts that more directly pressure agreement with a specific wrong answer, especially outside English. The local mitigation results also show that activation steering is a promising mechanism-level intervention even when the cross-lingual behavioral story is more complicated than expected.

## 10. Reproducibility
- Environment: `.venv`
- Dependency manifests: `pyproject.toml`, `requirements.txt`
- Main command:

```bash
source .venv/bin/activate
python src/run_full_study.py
```

Primary outputs:
- Metrics: `results/analysis/metrics_summary.csv`
- Tier summaries: `results/analysis/tier_summaries.json`
- Fisher tests: `results/analysis/fisher_results.json`
- Raw responses: `results/model_outputs/`
- Mechanistic artifacts: `results/mechanistic/`
- Figures: `figures/`

## 11. References
- Sharma et al. 2024. *Towards Understanding Sycophancy in Language Models.*
- Wei et al. 2024. *Simple synthetic data reduces sycophancy in large language models.*
- Chen et al. 2024. *From Yes-Men to Truth-Tellers.*
- Chataigner et al. 2024. *Multilingual Hallucination Gaps in Large Language Models.*
- Longpre et al. 2020. *MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering.*
