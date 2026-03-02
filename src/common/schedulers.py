from __future__ import annotations

import math


def cosine_with_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr_scale: float = 0.1) -> float:
    if total_steps <= 0:
        return base_lr
    min_lr = base_lr * min_lr_scale
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if step >= total_steps:
        return min_lr

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)
