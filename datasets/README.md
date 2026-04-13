# Downloaded Datasets

This directory contains the datasets and sample artifacts gathered for the project. Large dataset contents should not be committed to git; the `.gitignore` here keeps only documentation and lightweight samples.

## Dataset 1: MKQA

### Overview
- Source: `apple/ml-mkqa` and paper https://arxiv.org/abs/2007.15207
- Local file: `datasets/mkqa/mkqa.jsonl.gz`
- Size: 10,000 aligned QA items across 26 languages; 260,000 question-answer pairs total
- Format: gzipped JSONL
- Task: multilingual factual open-domain QA
- License: CC BY-SA 3.0 (per repository)

### Why it matters here
- Best factual-control dataset in this workspace for testing whether accuracy falls faster in lower-resource languages.
- Can be combined with social-pressure prompt wrappers to measure language-conditional capitulation.

### Download Instructions

Using git-cloned source (already done in this workspace):

```bash
git clone https://github.com/apple/ml-mkqa.git code/ml-mkqa
cp code/ml-mkqa/dataset/mkqa.jsonl.gz datasets/mkqa/
```

Alternative direct source:

```bash
wget https://raw.githubusercontent.com/apple/ml-mkqa/main/dataset/mkqa.jsonl.gz -O datasets/mkqa/mkqa.jsonl.gz
```

### Loading the Dataset

```python
import gzip, json

with gzip.open("datasets/mkqa/mkqa.jsonl.gz", "rt", encoding="utf-8") as f:
    first = json.loads(next(f))
```

### Validation Notes
- Verified locally: `10000` examples.
- Verified language keys: `ar, da, de, en, es, fi, fr, he, hu, it, ja, km, ko, ms, nl, no, pl, pt, ru, sv, th, tr, vi, zh_cn, zh_hk, zh_tw`.
- Sample schema saved to `datasets/mkqa/samples.json`.

## Dataset 2: BullshitBench Source Questions

### Overview
- Source: `petergpt/bullshit-benchmark`
- Local files:
  - `datasets/bullshitbench/questions.json`
  - `datasets/bullshitbench/questions.v2.json`
  - `datasets/bullshitbench/aggregate_summary_v1.json`
  - `datasets/bullshitbench/aggregate_summary_v2.json`
- Format: JSON
- Task: nonsense-premise detection / refusal to continue invalid assumptions
- Size:
  - v1: question bank embedded in `questions.json`
  - v2: `100` questions across software, finance, legal, medical, and physics

### Why it matters here
- Strong seed benchmark for factual robustness under misleading premises.
- Not multilingual by default; this project will likely need a translated/custom multilingual derivative.
- Useful for constructing pressure prompts where the user insists on a false or nonsensical premise.

### Download Instructions

Using git-cloned source (already done in this workspace):

```bash
git clone https://github.com/petergpt/bullshit-benchmark.git code/bullshit-benchmark
cp code/bullshit-benchmark/questions.json datasets/bullshitbench/
cp code/bullshit-benchmark/questions.v2.json datasets/bullshitbench/
cp code/bullshit-benchmark/data/latest/aggregate_summary.json datasets/bullshitbench/aggregate_summary_v1.json
cp code/bullshit-benchmark/data/v2/latest/aggregate_summary.json datasets/bullshitbench/aggregate_summary_v2.json
```

### Loading the Dataset

```python
import json

with open("datasets/bullshitbench/questions.v2.json") as f:
    data = json.load(f)

techniques = data["techniques"]
questions = [q for t in techniques for q in t["questions"]]
```

### Validation Notes
- `questions.json` and `questions.v2.json` are metadata-rich JSON objects, not flat arrays.
- Question samples extracted to:
  - `datasets/bullshitbench/questions.json.sample.json`
  - `datasets/bullshitbench/questions.v2.json.sample.json`
- For the current hypothesis, recommended next step is to translate a balanced subset across language-resource tiers rather than use the full English bank unchanged.

## Dataset 3: ELEPHANT Samples

### Overview
- Source: `myracheng/elephant`
- Local files:
  - `datasets/elephant/OEQ_sample.csv`
  - `datasets/elephant/AITA-YTA_sample.csv`
  - `datasets/elephant/AITA-NTA-OG_sample.csv`
  - `datasets/elephant/AITA-NTA-FLIP_sample.csv`
  - `datasets/elephant/SS_sample.csv`
- Format: CSV
- Task: social and moral sycophancy measurement
- Size: sample-only in this workspace (10 examples per file from the repo)

### Why it matters here
- Covers social agreement behavior that is adjacent to factual capitulation.
- Useful if the project expands beyond fact-based sycophancy into advice/opinion framing.

### Download Instructions

Sample files from cloned repo:

```bash
git clone https://github.com/myracheng/elephant.git code/elephant
cp code/elephant/sample_datasets/*.csv datasets/elephant/
```

Full datasets are hosted externally by the authors:

```bash
# See README for the OSF datasets.zip link:
open https://osf.io/r3dmj/?view_only=37ee66a8020a45c29a38bd704ca61067
```

### Loading the Dataset

```python
import pandas as pd

df = pd.read_csv("datasets/elephant/OEQ_sample.csv")
```

### Validation Notes
- Sample schemas and example rows saved to `datasets/elephant/samples.json`.
- Full ELEPHANT payload was not mirrored locally because the repo itself ships only samples and points to OSF for the complete archive.

## Recommendations

- Primary factual benchmark: MKQA.
- Primary misleading-premise benchmark seed: BullshitBench v2 question bank.
- Optional social-sycophancy extension: ELEPHANT.
- For the stated hypothesis, build a multilingual evaluation set by translating BullshitBench-style false-premise prompts and wrapping MKQA questions with social-pressure challenge turns.
