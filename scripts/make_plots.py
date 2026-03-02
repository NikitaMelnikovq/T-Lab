#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.metrics import moving_average


def parse_args():
    parser = argparse.ArgumentParser(description="Build training curves from metrics.jsonl")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="report/figures")
    parser.add_argument("--smoothing", type=int, default=5)
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


def pick_curve_rows(rows):
    curve = [r for r in rows if "env_steps" in r and "success_rate" in r and r.get("phase") in {"eval", "final_eval", "train"}]
    curve.sort(key=lambda x: (x.get("env_steps", 0), x.get("global_step", 0)))
    return curve


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = {}
    for p in results_dir.glob("*/metrics.jsonl"):
        method = p.parent.name
        rows = pick_curve_rows(load_rows(p))
        if rows:
            series[method] = rows

    if not series:
        print("No metrics.jsonl files found for plotting")
        return

    # success plot
    plt.figure(figsize=(8, 5))
    for method, rows in series.items():
        xs = [float(r.get("env_steps", 0)) for r in rows]
        ys = moving_average([float(r.get("success_rate", 0.0)) for r in rows], args.smoothing)
        plt.plot(xs, ys, label=method)
    plt.title("Success Rate vs Env Steps")
    plt.xlabel("env_steps")
    plt.ylabel("success_rate")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "success_vs_env_steps.png")
    plt.close()

    # return plot
    plt.figure(figsize=(8, 5))
    for method, rows in series.items():
        xs = [float(r.get("env_steps", 0)) for r in rows]
        ys = moving_average([float(r.get("avg_return", r.get("mean_return", 0.0))) for r in rows], args.smoothing)
        plt.plot(xs, ys, label=method)
    plt.title("Average Return vs Env Steps")
    plt.xlabel("env_steps")
    plt.ylabel("avg_return")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "return_vs_env_steps.png")
    plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
