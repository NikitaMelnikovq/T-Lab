from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlLogger:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, Any]) -> None:
        row = dict(payload)
        row.setdefault("timestamp", time.time())
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


class ExperimentLogger:
    def __init__(
        self,
        run_dir: Path,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = JsonlLogger(self.run_dir / "metrics.jsonl")

        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception as exc:
                print(f"[warning] tensorboard disabled: {exc}")

        self.wandb_run = None
        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project or "nanovlm-minigrid",
                    name=wandb_run_name,
                    dir=str(self.run_dir),
                )
            except Exception as exc:
                print(f"[warning] wandb disabled: {exc}")

    def log(self, payload: Dict[str, Any]) -> None:
        self.metrics.log(payload)
        step = payload.get("global_step")
        if self.tb_writer is not None:
            for k, v in payload.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, step if step is not None else 0)

        if self.wandb_run is not None:
            self.wandb_run.log(payload, step=step)

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def setup_console_logger(name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logger
