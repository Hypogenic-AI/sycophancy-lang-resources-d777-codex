# Research Plan: Sycophantic Tendencies Vary With Language Resource

## Motivation & Novelty Assessment

### Why This Research Matters
RLHF-era assistants often abandon correct answers when a user applies social pressure, which makes them unreliable in high-stakes settings such as medicine, law, and finance. If this failure mode is worse in lower-resource languages, then alignment and safety evaluations done mainly in English materially understate risk for many global users.

### Gap in Existing Work
The literature in [literature_review.md](/workspaces/sycophancy-lang-resources-d777-codex/literature_review.md) shows strong English evidence for sycophancy and separate multilingual evidence for factuality gaps, but very little direct cross-lingual evidence that combines factual correctness with explicit user challenge. The closest paper found is a 2026 Hindi Beacon extension, which is single-language, small-scale, and focused on Beacon-style prompts rather than a broad multilingual factual benchmark.

### Our Novel Contribution
This project combines a multilingual extension of BullshitBench with pressure-conditioned factual QA derived from MKQA across seven languages: English, French, Arabic, Hindi, Swahili, Yoruba, and Tagalog. It tests whether language resource level predicts capitulation under pressure, then follows the behavioral result with local mechanistic probing, activation steering, and lightweight mitigation.

### Experiment Justification
- Experiment 1: Behavioral evaluation on multilingual BullshitBench is needed to test whether false-premise detection degrades more sharply under challenge in lower-resource languages.
- Experiment 2: Behavioral evaluation on multilingual MKQA-derived factual QA is needed to separate false-premise robustness from ordinary factual answer retention.
- Experiment 3: Local mechanistic probing on an open-weight multilingual model is needed to test whether challenged prompts induce a stronger “capitulation” direction in low-resource languages.
- Experiment 4: Activation steering is needed to test whether the inferred direction is causal rather than merely correlational.
- Experiment 5: Prompt-based and steering-based mitigation is needed to test whether low-resource sycophancy can be reduced without large retraining.

## Research Question
Does sycophantic capitulation under user pressure increase as language resource level decreases, because factual representations are weaker in lower-resource languages and therefore less able to resist a challenge-induced agreement direction?

## Background and Motivation
Prior work shows that sycophancy is a robust RLHF failure mode in English and that multilingual factual reliability is uneven across languages. What is missing is a controlled multilingual test that combines objective answers, explicit user challenge, and a language-resource gradient. This project targets that gap with aligned multilingual prompts and domain-diverse false-premise examples.

## Hypothesis Decomposition
- H1: Final-answer correctness after challenge is lower in lower-resource languages than in higher-resource languages.
- H2: The drop from initial answer correctness to post-challenge correctness is larger in lower-resource languages.
- H3: On false-premise items, capitulation into answering nonsense rather than rejecting the premise increases as language resource decreases.
- H4: In a local open-weight model, challenged prompts induce a hidden-state direction associated with answer reversal, and that direction is stronger or more behaviorally effective in lower-resource languages.
- H5: Negative steering along the inferred direction and truth-prioritizing prompting reduce capitulation, with larger gains in lower-resource languages.

Independent variables:
- Language: `en`, `fr`, `ar`, `hi`, `sw`, `yo`, `tl`
- Resource tier: high (`en`, `fr`), medium (`ar`, `hi`, `tl`), low (`sw`, `yo`)
- Dataset type: BullshitBench vs MKQA-derived QA
- Condition: no-pressure vs pressure
- Model: API model vs local open-weight model
- Mitigation: none vs prompt mitigation vs steering mitigation

Dependent variables:
- Initial correctness / rejection rate
- Final correctness / rejection rate
- Capitulation rate
- Answer-change rate
- Local hidden-state contrast metrics and steering effect size

Alternative explanations to test or note:
- Translation quality rather than language resource
- Script effects or tokenization mismatch
- Domain imbalance in BullshitBench
- Model-specific multilingual weakness unrelated to sycophancy

## Proposed Methodology

### Approach
Use two complementary benchmarks. First, translate all 100 BullshitBench v2 items into the target languages while preserving the nonsensical element and domain labels. Second, sample aligned factual questions from MKQA, then translate prompts and canonical short answers into the target languages not covered by official MKQA. For each item, run a two-turn interaction: the model answers, then receives a challenge pressuring it away from the correct/rejecting answer. Evaluate both a real API model and a local open-weight multilingual model under matched decoding settings. Use the local model for hidden-state extraction, steering, and mitigation because internal activations are required.

### Experimental Steps
1. Validate workspace resources, environment, and hardware so the pipeline is reproducible and stays isolated in `.venv`.
2. Construct multilingual evaluation sets:
   - Translate BullshitBench items into six non-English target languages with a real LLM.
   - Sample a manageable aligned MKQA subset and translate queries plus canonical short answers into Hindi, Swahili, Yoruba, and Tagalog.
   - Build challenge turns that assert the opposite of the ideal first-turn behavior.
3. Run baseline behavioral evaluation:
   - Condition A: initial answer only.
   - Condition B: answer followed by social-pressure challenge and final answer.
4. Score outputs:
   - BullshitBench: `reject`, `hedge`, `engage` rubric plus binary capitulation indicator.
   - MKQA: exact match / alias match plus answer-change and final correctness.
5. Run local mechanistic analysis:
   - Extract assistant-side hidden states for initial and challenged contexts.
   - Measure per-layer representation differences between retained-truth and capitulated cases.
6. Run activation steering:
   - Derive a mean contrast vector between challenged-capitulating and challenged-noncapitulating cases.
   - Inject the negative vector during generation and measure change in capitulation.
7. Run mitigation experiments:
   - Prompt mitigation: explicit truth-over-agreement system instruction.
   - Steering mitigation: negative sycophancy vector.
8. Analyze language-resource effects with confidence intervals, permutation/bootstrap tests, and regression against resource tier.

### Baselines
- No-pressure condition
- English as high-resource anchor
- High-resource mean (`en`, `fr`) vs low-resource mean (`sw`, `yo`)
- Unmitigated model vs prompt mitigation
- Unmitigated model vs steering mitigation

### Evaluation Metrics
- BullshitBench rejection accuracy: proportion correctly flagging the premise as incoherent.
- BullshitBench capitulation rate: proportion that initially reject or hedge but later engage with the false premise after challenge.
- MKQA initial accuracy and final accuracy.
- MKQA answer-change rate after challenge.
- Correct-to-incorrect flip rate after challenge.
- Delta-from-English and delta-from-high-resource-average.
- Mechanistic separability: cosine distance / linear probe AUC between retained-truth and capitulated hidden states.
- Steering gain: absolute reduction in capitulation rate.

### Statistical Analysis Plan
- Primary null hypothesis: capitulation rate is independent of language resource tier.
- For language-tier comparisons: two-sided bootstrap confidence intervals and permutation tests on mean capitulation differences.
- For paired no-pressure vs pressure comparisons within language: McNemar test where applicable.
- For trend over tiers: logistic regression with resource tier and dataset type as predictors.
- For multiple language-wise comparisons: Benjamini-Hochberg FDR correction at `q = 0.05`.
- Report effect sizes as absolute percentage-point differences and odds ratios.

## Expected Outcomes
Results support the hypothesis if lower-resource languages show:
- lower initial factual robustness,
- larger challenge-induced drops in correctness or rejection,
- higher final capitulation rates,
- and stronger or more steerable challenge-associated hidden-state directions.

Results weaken the hypothesis if language effects are flat, inconsistent across datasets, or fully explained by translation artifacts.

## Timeline and Milestones
1. Planning and resource review: complete first.
2. Environment setup and dependency installation: short setup pass.
3. Dataset construction and translation: first substantial implementation milestone.
4. Behavioral runs and scoring: core experiment milestone.
5. Mechanistic probing, steering, and mitigation: local-model milestone.
6. Statistical analysis, figures, report, and validation: final milestone.

## Potential Challenges
- API cost or rate limits:
  Cache every call and reuse translations and generations.
- Translation drift:
  Request concise faithful translations and preserve metadata for auditing.
- MKQA answer localization difficulty:
  Restrict to short-answer items and keep answer aliases.
- Hidden-state analysis complexity:
  Use a lightweight but defensible contrast-vector method rather than full path patching.
- Runtime:
  Start with a moderate MKQA subset and batch local inference on available GPUs.

## Success Criteria
- A multilingual benchmark artifact is created locally with the seven required languages.
- At least one real API model and one local open-weight model are evaluated.
- Results quantify per-language and per-tier sycophancy/capitulation.
- At least one mechanistic and one intervention result are produced on the local model.
- `REPORT.md`, `README.md`, code, figures, and raw results are saved and reproducible.
