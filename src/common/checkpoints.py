from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def save_checkpoint(
    model,
    optimizer,
    step: int,
    payload: Dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "payload": payload,
        },
        path,
    )


def save_last(model, optimizer, step: int, payload: Dict[str, Any], ckpt_dir: Path) -> Path:
    path = ckpt_dir / "last" / "trainer_state.pt"
    save_checkpoint(model, optimizer, step, payload, path)
    model.save_pretrained(str(ckpt_dir / "last"))
    return path


def save_best(
    model,
    optimizer,
    step: int,
    metric_name: str,
    metric_value: float,
    ckpt_dir: Path,
) -> Path:
    path = ckpt_dir / "best" / "trainer_state.pt"
    payload = {"best_metric_name": metric_name, "best_metric_value": metric_value}
    save_checkpoint(model, optimizer, step, payload, path)
    model.save_pretrained(str(ckpt_dir / "best"))
    with (ckpt_dir / "best" / "best_metric.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return path


def load_checkpoint(path: Path, optimizer=None) -> Tuple[int, Dict[str, Any]]:
    state = torch.load(path, map_location="cpu")
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step", 0)), state.get("payload", {})


def load_best_or_last(run_dir: Path) -> Optional[Path]:
    best = run_dir / "checkpoints" / "best"
    last = run_dir / "checkpoints" / "last"
    if best.exists():
        return best
    if last.exists():
        return last
    return None
