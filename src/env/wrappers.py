from __future__ import annotations

from typing import Dict


def normalize_obs(obs: Dict) -> Dict:
    """Keep a strict observation contract used by the pipeline."""
    return {
        "image": obs["image"],
        "mission": obs.get("mission", ""),
    }
