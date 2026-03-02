from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.total += float(value)
        self.count += 1

    @property
    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@dataclass
class EpisodeStats:
    returns: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    successes: List[int] = field(default_factory=list)

    def add(self, episode_return: float, episode_steps: int, success: bool) -> None:
        self.returns.append(float(episode_return))
        self.steps.append(int(episode_steps))
        self.successes.append(1 if success else 0)

    def summary(self) -> Dict[str, float]:
        n = max(1, len(self.returns))
        return {
            "avg_return": sum(self.returns) / n,
            "avg_steps_to_goal": sum(self.steps) / n,
            "success_rate": sum(self.successes) / n,
            "episodes": len(self.returns),
        }


def moving_average(values: Iterable[float], window: int) -> List[float]:
    vals = list(values)
    if window <= 1:
        return vals
    out: List[float] = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        chunk = vals[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def first_step_to_threshold(metrics_rows: List[Dict], threshold: float, value_key: str = "success_rate") -> Optional[int]:
    for row in metrics_rows:
        if float(row.get(value_key, 0.0)) >= threshold:
            return int(row.get("env_steps", 0))
    return None
