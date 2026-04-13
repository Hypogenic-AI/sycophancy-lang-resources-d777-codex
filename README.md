# Sycophantic Tendencies Vary With Language Resource

This workspace runs a multilingual sycophancy study over 7 languages: English, French, Arabic, Hindi, Swahili, Yoruba, and Tagalog. It combines translated BullshitBench false-premise items with MKQA-derived factual QA, evaluates `gpt-4.1-mini` and `Qwen/Qwen2.5-3B-Instruct`, and adds a local mechanistic/steering pass on Qwen.

Key findings:
- Low-resource languages hurt factual QA much more than high-resource languages on both models.
- The main hypothesis was only partially supported: factual weakness scaled with resource level, but BullshitBench capitulation did not increase in low-resource languages and often moved in the opposite direction.
- On Qwen, prompt mitigation improved BullshitBench final rejection from `69.3%` to `78.0%`, and activation steering raised it to `94.7%` while driving measured capitulation to `0%`.
- The pressure turn on BullshitBench often triggered second-turn self-correction rather than second-turn capitulation, which is a design limitation for this benchmark setup.

Reproduce:
```bash
source .venv/bin/activate
python src/run_full_study.py
```

Important outputs:
- Full report: `REPORT.md`
- Planning document: `planning.md`
- Main script: `src/run_full_study.py`
- Metrics: `results/analysis/metrics_summary.csv`
- Merged outputs: `results/analysis/merged_outputs.csv`
- Mechanistic artifacts: `results/mechanistic/`
- Figures: `figures/`

File structure:
- `datasets/`: source datasets
- `results/model_outputs/`: raw model responses
- `results/analysis/`: judged outputs and summary tables
- `results/mechanistic/`: steering vector and layer scores
- `figures/`: generated plots
