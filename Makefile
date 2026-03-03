PYTHON ?= python3

.PHONY: install collect dataset_stats sft grpo_action grpo_text_action plots summary report smoke run_all

install:
	$(PYTHON) -m pip install -r requirements.txt

collect:
	$(PYTHON) scripts/collect_expert.py --config configs/sft.yaml

dataset_stats:
	$(PYTHON) scripts/show_dataset_stats.py --dataset_dir results/expert_dataset

sft:
	$(PYTHON) -m src.sft.train_sft --config configs/sft.yaml

grpo_action:
	$(PYTHON) -m src.grpo.train_grpo_action --config configs/grpo_action.yaml

grpo_text_action:
	$(PYTHON) -m src.grpo.train_grpo_text_action --config configs/grpo_text_action.yaml

plots:
	$(PYTHON) scripts/make_plots.py --results_dir results --output_dir report/figures

summary:
	$(PYTHON) scripts/summarize_results.py --results_dir results --output_dir report/tables

report:
	@if command -v pdflatex >/dev/null 2>&1; then \
		cd report && pdflatex -interaction=nonstopmode main.tex >/tmp/report_build.log && pdflatex -interaction=nonstopmode main.tex >>/tmp/report_build.log; \
		echo "Built report/main.pdf"; \
	else \
		echo "pdflatex not found, skipping PDF build"; \
	fi

smoke:
	$(PYTHON) scripts/smoke_test.py --config configs/sft.yaml

run_all:
	bash scripts/run_all.sh
