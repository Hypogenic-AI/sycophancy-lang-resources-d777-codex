# Outline: Sycophantic Tendencies Vary With Language Resource

## Title
- Main claim: factual weakness scales with lower language resource, but false-premise capitulation does not.
- Candidate title: "Lower-Resource Languages Weaken Factual Robustness but Not False-Premise Capitulation in Multilingual LLM Evaluation"

## Abstract
- Context: English sycophancy is established; multilingual risk is unknown.
- Gap: no direct test combining multilingual factual QA and challenge-turn sycophancy.
- Approach: 7-language BullshitBench + MKQA subset; `gpt-4.1-mini` and `Qwen/Qwen2.5-3B-Instruct`; mechanistic follow-up on Qwen.
- Key results:
  - BullshitBench capitulation stays low and does not rise in low-resource languages.
  - MKQA final accuracy drops by 20.0 points (`gpt-4.1-mini`) and 22.5 points (Qwen) from high- to low-resource tiers.
  - Qwen steering raises final rejection from 0.693 to 0.947 and reduces capitulation from 0.0129 to 0.0000.
- Significance: multilingual safety risk is split between factual weakness and refusal-heavy challenge behavior.

## Introduction
- Hook: challenge-based sycophancy is a known alignment failure, but English-only evaluation may miss multilingual behavior.
- Importance: if low-resource languages induce higher capitulation, English-centric safety evaluation understates risk.
- Gap: literature links English sycophancy and multilingual factuality gaps, but lacks a direct cross-lingual challenge study.
- Approach:
  - Translate BullshitBench and extend MKQA to seven languages.
  - Evaluate API and open-weight models under two-turn challenge.
  - Follow with Qwen hidden-state contrast and activation steering.
- Quantitative preview:
  - BullshitBench high-vs-low capitulation is lower, not higher, in low-resource languages.
  - MKQA high-vs-low final accuracy gaps are -0.200 and -0.225.
  - Steering improves Qwen rejection by 25.4 points over base.
- Contributions:
  - We provide a multilingual challenge-turn evaluation across seven languages.
  - We show a split failure profile: factual degradation without increased false-premise capitulation.
  - We add mechanistic evidence and a strong steering intervention on Qwen.
  - We identify benchmark-design implications for future multilingual sycophancy work.

## Related Work
- Theme 1: English sycophancy and challenge-turn behavior.
  - Sharma et al. (2024), Wei et al. (2023), Chen et al. (2024).
- Theme 2: broader and cross-lingual sycophancy framing.
  - Cheng et al. (2025), Sattigeri (2026).
- Theme 3: multilingual factual reliability and evaluation.
  - Chataigner et al. (2024), Longpre et al. (2020).
- Positioning:
  - Unlike English-only sycophancy work, we vary language resource directly.
  - Unlike multilingual factuality work, we add explicit user challenge.
  - Unlike prior single-language cross-lingual extensions, we test seven languages and include mechanistic steering.

## Methodology
- Task setup:
  - Seven languages grouped into high, medium, low resource tiers.
  - Two benchmarks: 100 false-premise BullshitBench items, 20 MKQA factual items.
  - Two-turn protocol with challenge.
- Models and conditions:
  - `gpt-4.1-mini`, `Qwen/Qwen2.5-3B-Instruct`.
  - Base, truth prompt, and Qwen steering.
- Scoring:
  - BullshitBench `reject` / `hedge` / `engage`, final rejection, capitulation.
  - MKQA alias-matched final accuracy, flip-to-wrong, answer changes.
- Mechanistic analysis:
  - Qwen challenged contexts, reject-vs-engage hidden-state contrasts.
  - Layer selection by separation score; language-wise direction norms.
- Evidence to map:
  - Fisher p-values on high-vs-low BullshitBench capitulation.
  - Best steering layer 2, separation score 7.10, 563 examples.

## Results
- Subsection 1: BullshitBench base behavior.
  - Table: average results by model.
  - Table: per-language results by model.
  - Figure: high/low tier comparison for final rejection and capitulation.
  - Claim: low-resource languages do not show higher capitulation; Fisher tests non-significant.
- Subsection 2: MKQA factual robustness.
  - Table: per-language MKQA results.
  - Claim: clear high-to-low accuracy drop in both models.
- Subsection 3: Qwen interventions and mechanistic follow-up.
  - Table: base vs truth prompt vs steering.
  - Figure: language-wise direction norms.
  - Claim: compact early-layer signal and strong behavioral steering effect.

## Discussion
- Interpretation:
  - Split failure profile: knowledge weakness vs conservative rejection.
  - BullshitBench challenge acts partly as a reconsideration prompt, not pure pressure.
- Limitations:
  - Benchmark mismatch.
  - Translation dependence.
  - Judge coupling.
  - Limited model coverage.
  - Simplified mechanistic method.
- Broader implications:
  - English-only safety evaluations miss multilingual factual weakness.
  - Future sycophancy benchmarks should pressure agreement with a specific wrong answer.

## Conclusion
- Restate contribution and main findings.
- Main takeaway: lower-resource languages hurt factual robustness, but this benchmark does not support stronger multilingual capitulation.
- Future work: better pressure prompts, more models, stronger causal analysis, culturally adapted variants.

## Tables and Figures Plan
- `tables/main_behavior.tex`: BullshitBench and MKQA tier summary table.
- `tables/per_language_results.tex`: compact per-language BullshitBench and MKQA table pair.
- `tables/qwen_interventions.tex`: Qwen base/truth/steering plus mechanistic values.
- `figures/tier_comparison.pdf`: bar chart for high vs low tier metrics.
- `figures/qwen_direction_norms.pdf`: bar chart for per-language direction norms.

## Citation Plan
- Intro/related work: Sharma et al. 2024; Wei et al. 2023; Chen et al. 2024.
- Cross-lingual framing: Sattigeri 2026; Cheng et al. 2025.
- Factuality and benchmark: Chataigner et al. 2024; Longpre et al. 2020.
