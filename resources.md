# Resources Catalog

## Summary

This document catalogs the papers, datasets, and code repositories gathered for the project `Sycophantic tendencies vary with language resource`.

## Papers

Total papers downloaded: 7

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Towards Understanding Sycophancy in Language Models | Sharma et al. | 2024 | `papers/2310.13548v4_towards_understanding_sycophancy.pdf` | Foundational evidence that preference data and RLHF can incentivize sycophancy |
| Simple synthetic data reduces sycophancy in large language models | Wei et al. | 2024 | `papers/2308.03958v2_simple_synthetic_data_reduces_sycophancy.pdf` | Synthetic-data mitigation baseline |
| From Yes-Men to Truth-Tellers | Chen et al. | 2024 | `papers/2409.01658v3_yes_men_to_truth_tellers.pdf` | Mechanistic mitigation via pinpoint tuning |
| ELEPHANT | Cheng et al. | 2025 | `papers/2505.13995v2_elephant_social_sycophancy.pdf` | Social/moral sycophancy datasets and metrics |
| Extending Beacon to Hindi | Sattigeri | 2026 | `papers/2602.00046v1_extending_beacon_hindi.pdf` | Direct cross-lingual sycophancy evidence |
| Multilingual Hallucination Gaps in Large Language Models | Chataigner et al. | 2024 | `papers/2410.18270v1_multilingual_hallucination_gaps.pdf` | Factual reliability varies across languages |
| MKQA | Longpre et al. | 2020 | `papers/2007.15207v2_mkqa.pdf` | Primary multilingual factual QA benchmark |

See `papers/README.md` for the paper index and chunked-reading notes.

## Datasets

Total datasets downloaded or staged locally: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MKQA | `apple/ml-mkqa` | 10k examples, 26 languages | multilingual factual QA | `datasets/mkqa/` | Canonical factual benchmark; fully local |
| BullshitBench source questions | `petergpt/bullshit-benchmark` | v2 has 100 prompts | false-premise detection | `datasets/bullshitbench/` | English-only source bank for multilingual adaptation |
| ELEPHANT samples | `myracheng/elephant` | 5 CSV sample files | social/moral sycophancy | `datasets/elephant/` | Samples local; full archive remains on OSF |

See `datasets/README.md` for download instructions and schema notes.

## Code Repositories

Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| ml-mkqa | https://github.com/apple/ml-mkqa | official MKQA data and evaluation | `code/ml-mkqa/` | contains canonical dataset file |
| bullshit-benchmark | https://github.com/petergpt/bullshit-benchmark | false-premise benchmark and pipeline | `code/bullshit-benchmark/` | seed source for multilingual/custom BullshitBench |
| sycophancy-intervention | https://github.com/google/sycophancy-intervention | synthetic-data mitigation baseline | `code/sycophancy-intervention/` | lightweight intervention code |
| elephant | https://github.com/myracheng/elephant | social sycophancy metrics and samples | `code/elephant/` | full datasets external on OSF |
| sycophancy-interpretability | https://github.com/yellowtownhz/sycophancy-interpretability | path patching and pinpoint tuning | `code/sycophancy-interpretability/` | strong basis for mechanism-level follow-up |

See `code/README.md` for entry points and reuse suggestions.

## Resource Gathering Notes

### Search Strategy
- Started with paper-finder and manual arXiv queries centered on sycophancy, multilingual hallucination, and low-resource language performance.
- Prioritized papers that either directly measure sycophancy or provide multilingual factuality evidence needed to support the hypothesis.
- Pulled official benchmark and implementation repositories when available.

### Selection Criteria
- Direct relevance to sycophancy under challenge.
- Direct relevance to multilingual factual reliability.
- Availability of code or datasets for immediate experimentation.
- Preference for papers that could inform both benchmark design and mitigation baselines.

### Challenges Encountered
- The paper-finder helper required fallback/manual search.
- Hugging Face `load_dataset("mkqa")` failed with the current `datasets` release because the dataset still depends on an older script-based loader.
- The BullshitBench dataset is English-first and must be translated/adapted for the target hypothesis.
- The full ELEPHANT datasets are not stored in the GitHub repo and remain on OSF.

### Gaps and Workarounds
- No large ready-made multilingual sycophancy benchmark was found.
- Workaround: combine `MKQA` with pressure prompts and translate/adapt `BullshitBench` prompts across language-resource tiers.
- Direct code release for the Hindi Beacon extension was not identified during this pass; the paper still provides the design rationale.

## Recommendations for Experiment Design

1. Primary dataset(s): Use `MKQA` as the factual anchor and build pressure-conditioned prompt pairs on top of it; use translated `BullshitBench` items as explicit false-premise cases.
2. Baseline methods: include no-pressure prompting, literal translation, culturally adapted translation, synthetic-data intervention (`sycophancy-intervention`), and, if feasible, pinpoint tuning (`sycophancy-interpretability`).
3. Evaluation metrics: factual accuracy, final-answer flip rate, capitulation rate after challenge, and delta relative to English/high-resource languages.
4. Code to adapt/reuse: `code/ml-mkqa/` for scoring, `code/bullshit-benchmark/` for prompt seeds, `code/sycophancy-interpretability/` for evaluation structure and path patching, and `code/sycophancy-intervention/` for a mitigation baseline.
