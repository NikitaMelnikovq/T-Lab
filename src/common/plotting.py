from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def plot_lines(
    x_key: str,
    y_key: str,
    rows_by_method: Dict[str, List[dict]],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for method, rows in rows_by_method.items():
        xs = [r.get(x_key, 0) for r in rows]
        ys = [r.get(y_key, 0.0) for r in rows]
        if not xs:
            continue
        plt.plot(xs, ys, label=method)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
