# Cloned Repositories

## Repo 1: `ml-mkqa`
- URL: https://github.com/apple/ml-mkqa
- Purpose: official MKQA dataset and evaluation scripts
- Location: `code/ml-mkqa/`
- Key files:
  - `dataset/mkqa.jsonl.gz`
  - `mkqa_eval.py`
  - `mkqa_eval_all_languages.py`
- Notes: contains the canonical data file locally; no extra download was required beyond cloning.

## Repo 2: `bullshit-benchmark`
- URL: https://github.com/petergpt/bullshit-benchmark
- Purpose: nonsense-premise benchmark with question banks, collection pipeline, and public aggregate outputs
- Location: `code/bullshit-benchmark/`
- Key files:
  - `questions.json`
  - `questions.v2.json`
  - `scripts/run_end_to_end.sh`
  - `data/latest/`
  - `data/v2/latest/`
- Notes: useful as the English source set for a multilingual/custom BullshitBench derivative.

## Repo 3: `sycophancy-intervention`
- URL: https://github.com/google/sycophancy-intervention
- Purpose: code for the synthetic-data intervention from Wei et al. (2024)
- Location: `code/sycophancy-intervention/`
- Key files:
  - `code/dataset_pipeline.py`
  - `code/generate_data.py`
  - `code/pull_from_huggingface.py`
- Notes: minimal dependency footprint; primarily useful for generating intervention data rather than evaluation sets.

## Repo 4: `elephant`
- URL: https://github.com/myracheng/elephant
- Purpose: social and moral sycophancy metrics plus sample datasets
- Location: `code/elephant/`
- Key files:
  - `sycophancy_scorers.py`
  - `moral_sycophancy_scorer.py`
  - `sample_datasets/*.csv`
- Notes: full datasets are external on OSF; the repo itself provides sample CSVs and the scoring pipeline.

## Repo 5: `sycophancy-interpretability`
- URL: https://github.com/yellowtownhz/sycophancy-interpretability
- Purpose: evaluation datasets, path patching, and pinpoint tuning from Chen et al. (2024)
- Location: `code/sycophancy-interpretability/`
- Key files:
  - `evaluation/evaluate_sycophancy_chat_vllm.py`
  - `evaluation/datasets/sycophancy_eval/`
  - `path_patching/path_patching_hf.py`
  - `pinpoint_tuning/`
  - `prepare_training_data/`
- Notes: most complete reusable codebase here for mechanistic analysis and multi-turn sycophancy evaluation; expects vLLM / model-serving infrastructure for full use.

## Practical Reuse Recommendation

- Use `ml-mkqa` for multilingual factual answer checking.
- Use `bullshit-benchmark` as the English seed bank for false-premise prompts.
- Use `sycophancy-interpretability` if the experiment runner needs multi-turn challenge templates or head-level analysis.
- Use `sycophancy-intervention` for a lightweight mitigation baseline.
- Use `elephant` only if expanding into social or moral sycophancy beyond factual capitulation.
