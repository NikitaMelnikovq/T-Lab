from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper


def _coerce_optional_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"" , "none", "null"}:
            return None
        # Accept accidental trailing text by taking the first token.
        head = v.split()[0]
        return int(head)
    return int(value)


def make_env(
    env_id: str = "MiniGrid-Empty-8x8-v0",
    seed: int = 0,
    tile_size: int = 8,
    max_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
):
    kwargs = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    parsed_max_steps = _coerce_optional_int(max_steps)
    if parsed_max_steps is not None:
        kwargs["max_steps"] = parsed_max_steps

    env = gym.make(env_id, **kwargs)
    env = RGBImgObsWrapper(env, tile_size=tile_size)
    obs, info = env.reset(seed=seed)
    return env, obs, info


def obs_to_image_mission(obs: dict) -> Tuple:
    return obs["image"], obs.get("mission", "")
