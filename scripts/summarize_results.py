#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.metrics import first_step_to_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize experiment results into CSV and LaTeX table")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="report/tables")
    return parser.parse_args()


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def method_output_format(method: str) -> str:
    if method == "grpo_text_action":
        return "Plan + Action token"
    return "Action token"


def build_summary_row(method: str, rows):
    eval_rows = [r for r in rows if r.get("phase") in {"eval", "final_eval"}]
    if not eval_rows:
        eval_rows = rows

    eval_rows.sort(key=lambda r: (r.get("env_steps", 0), r.get("global_step", 0)))
    final = eval_rows[-1] if eval_rows else {}

    return {
        "method": method,
        "output_format": method_output_format(method),
        "env_steps_to_0.8": first_step_to_threshold(eval_rows, 0.8),
        "env_steps_to_0.9": first_step_to_threshold(eval_rows, 0.9),
        "final_success": final.get("success_rate", 0.0),
        "final_return": final.get("avg_return", final.get("mean_return", 0.0)),
        "notes": "",
    }


def write_latex(rows, path: Path):
    lines = [
        "\\begin{tabular}{l l r r r r l}",
        "\\hline",
        "method & output\\_format & env\\_steps\\_to\\_0.8 & env\\_steps\\_to\\_0.9 & final\\_success & final\\_return & notes \\\\",
        "\\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['method']} & {r['output_format']} & {r['env_steps_to_0.8'] if r['env_steps_to_0.8'] is not None else '-'} & "
            f"{r['env_steps_to_0.9'] if r['env_steps_to_0.9'] is not None else '-'} & "
            f"{r['final_success']:.4f} & {r['final_return']:.4f} & {r['notes']} \\\\" 
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(results_dir.glob("*/metrics.jsonl")):
        method = p.parent.name
        rows.append(build_summary_row(method, load_rows(p)))

    if not rows:
        print("No result metrics found")
        return

    csv_path = out_dir / "results_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "output_format",
                "env_steps_to_0.8",
                "env_steps_to_0.9",
                "final_success",
                "final_return",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    tex_path = out_dir / "results_table.tex"
    write_latex(rows, tex_path)

    print(f"Wrote {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()
