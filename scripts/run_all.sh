#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

$PYTHON scripts/collect_expert.py --config configs/sft.yaml
$PYTHON scripts/show_dataset_stats.py --dataset_dir results/expert_dataset
$PYTHON -m src.sft.train_sft --config configs/sft.yaml
$PYTHON -m src.grpo.train_grpo_action --config configs/grpo_action.yaml
$PYTHON -m src.grpo.train_grpo_text_action --config configs/grpo_text_action.yaml
$PYTHON scripts/make_plots.py --results_dir results --output_dir report/figures
$PYTHON scripts/summarize_results.py --results_dir results --output_dir report/tables

if command -v pdflatex >/dev/null 2>&1; then
  (cd report && pdflatex -interaction=nonstopmode main.tex >/tmp/report_build.log && pdflatex -interaction=nonstopmode main.tex >>/tmp/report_build.log)
  echo "[run_all] PDF built: report/main.pdf"
else
  echo "[run_all] WARNING: pdflatex not found, skipping report build"
fi
