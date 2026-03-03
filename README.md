# NanoVLM MiniGrid Pipeline

Reproducible Python repository for:

- **SFT** on expert trajectories
- **GRPO (action)**
- **GRPO (text+action)**

Default training/eval environment: `MiniGrid-Empty-Random-6x6-v0` + `RGBImgObsWrapper`.
Expert data collection mixes random EmptyEnv sizes via `data.collect_envs` in `configs/sft.yaml`.

## Features

- End-to-end pipeline from zero with one script.
- Deterministic seeding (as far as practical with PyTorch/CUDA).
- Pure PyTorch RL loops (no SB3, no RLlib).
- Unified artifacts in `results/*`:
  - `metrics.jsonl`
  - checkpoints (`best`, `last`)
  - plots and summary table
  - optional PDF report.

## Repository layout

```text
configs/
  sft.yaml
  grpo_action.yaml
  grpo_text_action.yaml
src/
  common/
  env/
  data/
  model/
  sft/
  grpo/
scripts/
  collect_expert.py
  show_dataset_stats.py
  evaluate.py
  make_plots.py
  summarize_results.py
  smoke_test.py
  run_all.sh
report/
  main.tex
  refs.bib
  figures/
  tables/
results/
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Quick start (full pipeline)

```bash
bash scripts/run_all.sh
```

This performs:

1. Expert dataset collection.
2. Dataset stats check (action/env distribution).
3. SFT training.
4. GRPO-action training.
5. GRPO-text+action training.
6. Plot generation.
7. Results summary table.
8. PDF report build (if `pdflatex` is installed).

If `pdflatex` is missing, pipeline continues and prints a warning.

## Step-by-step commands

### 1) Collect expert data

```bash
python scripts/collect_expert.py --config configs/sft.yaml
```

Output default: `results/expert_dataset/`

Optional quick sanity check:

```bash
python scripts/show_dataset_stats.py --dataset_dir results/expert_dataset
```

### 2) Train SFT

```bash
python -m src.sft.train_sft --config configs/sft.yaml
```

Output default: `results/sft/`

### 3) Train GRPO-action

```bash
python -m src.grpo.train_grpo_action --config configs/grpo_action.yaml
```

Output default: `results/grpo_action/`

### 4) Train GRPO-text+action

```bash
python -m src.grpo.train_grpo_text_action --config configs/grpo_text_action.yaml
```

Output default: `results/grpo_text_action/`

### 5) Build plots

```bash
python scripts/make_plots.py --results_dir results --output_dir report/figures
```

### 6) Build summary table

```bash
python scripts/summarize_results.py --results_dir results --output_dir report/tables
```

### 7) Build PDF report

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cd ..
```

## Evaluation

Evaluate any checkpoint:

```bash
python scripts/evaluate.py --mode sft --checkpoint results/sft/checkpoints/best
python scripts/evaluate.py --mode grpo_action --checkpoint results/grpo_action/checkpoints/best
python scripts/evaluate.py --mode grpo_text_action --checkpoint results/grpo_text_action/checkpoints/best
```

## Smoke test

```bash
python scripts/smoke_test.py --config configs/sft.yaml
```

Checks:

1. Env reset/step with wrappers.
2. One model batch/action forward.
3. Two expert runs and deterministic first trajectory.

## Action mapping

| action_id | token |
|---:|---|
| 0 | `<ACT_LEFT>` |
| 1 | `<ACT_RIGHT>` |
| 2 | `<ACT_FORWARD>` |
| 3 | `<ACT_PICKUP>` |
| 4 | `<ACT_DROP>` |
| 5 | `<ACT_TOGGLE>` |
| 6 | `<ACT_DONE>` |

## Dataset schema (`metadata.jsonl`)

Each step record contains:

- `episode_id`
- `step_idx`
- `seed`
- `env_id`
- `env_size`
- `mission_text`
- `image_path`
- `action_id`
- `action_token`
- `prompt_variant_id`

Train/val split files:

- `splits/train_ids.txt`
- `splits/val_ids.txt`

## Logging format

`results/<run>/metrics.jsonl` includes key fields:

- `timestamp`
- `phase`
- `global_step`
- `env_steps`
- `episodes`
- `success_rate`
- `avg_return`
- `loss`

## Reproducibility notes

- Seeds are set for Python, NumPy, and Torch.
- Deterministic Torch mode is enabled with `warn_only=True`.
- Exact bitwise reproducibility may still vary by CUDA/cuDNN version and hardware.

## Compute profile

Default configs target **Balanced 1 GPU (8–16 GB)** with CPU fallback for smoke tests and tiny runs.

## NanoVLM pinning

This project vendors NanoVLM `v0.1` model code (tag commit `6ba9082e16f1fc8c21a1f8d0c54b26c9233c8771`)
under `src/model/vendor/nanovlm_v0_1/models`, and uses HF weights from:

- `lusxvr/nanoVLM-222M`

First run downloads model/backbone weights automatically.

## Troubleshooting

### No GPU

Set `experiment.device: cpu` in config. Training will be much slower.

### HF download failed

Check internet, HF availability, and local cache permissions. Re-run command.

### `pdflatex` missing

Install TeX Live (or equivalent), or skip PDF build; plots/tables are still produced.
