#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import ensure_dir, load_yaml
from src.common.seed import set_seed
from src.data.transforms import choose_prompt_variant
from src.env.expert import evaluate_expert_success, oracle_action
from src.env.make_env import _coerce_optional_int, make_env
from src.model.action_tokenizer import action_id_to_token


@dataclass
class EnvSpec:
    env_id: str
    weight: float
    max_steps: Optional[int]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect expert trajectories for MiniGrid EmptyEnv")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--check_expert", action="store_true")
    return parser.parse_args()


def _build_env_specs(env_cfg: Dict, data_cfg: Dict) -> List[EnvSpec]:
    specs_cfg = data_cfg.get("collect_envs")
    if not specs_cfg:
        return [
            EnvSpec(
                env_id=str(env_cfg.get("env_id", "MiniGrid-Empty-8x8-v0")),
                weight=1.0,
                max_steps=_coerce_optional_int(env_cfg.get("max_steps", None)),
            )
        ]

    out: List[EnvSpec] = []
    for item in specs_cfg:
        env_id = str(item.get("env_id"))
        if not env_id:
            continue
        weight = float(item.get("weight", 1.0))
        max_steps = _coerce_optional_int(item.get("max_steps", env_cfg.get("max_steps", None)))
        out.append(EnvSpec(env_id=env_id, weight=weight, max_steps=max_steps))

    if not out:
        raise ValueError("data.collect_envs is present but no valid env specs were parsed")
    return out


def _sample_env_spec(rng: random.Random, specs: List[EnvSpec]) -> EnvSpec:
    weights = [max(0.0, s.weight) for s in specs]
    if sum(weights) <= 0:
        weights = [1.0 for _ in specs]
    return rng.choices(specs, weights=weights, k=1)[0]


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed, deterministic=True)

    env_cfg = cfg["env"]
    data_cfg = cfg["data"]

    num_episodes = int(args.episodes or data_cfg.get("num_expert_episodes", 2000))
    output_dir = Path(args.output_dir or data_cfg["dataset_dir"])
    ensure_dir(output_dir / "images")
    splits_dir = ensure_dir(output_dir / "splits")

    metadata_path = output_dir / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    tile_size = int(env_cfg.get("tile_size", 8))
    use_prompt_variants = bool(data_cfg.get("use_prompt_variants", True))
    prompt_variant_count = int(data_cfg.get("prompt_variant_count", 3))
    split_seed = int(data_cfg.get("split_seed", seed))
    train_split = float(data_cfg.get("train_split", 0.9))

    env_specs = _build_env_specs(env_cfg=env_cfg, data_cfg=data_cfg)
    rng = random.Random(seed)

    env_cache: Dict[Tuple[str, int, Optional[int]], object] = {}

    def get_env(spec: EnvSpec, episode_seed: int):
        key = (spec.env_id, tile_size, spec.max_steps)
        if key not in env_cache:
            env, _, _ = make_env(
                env_id=spec.env_id,
                seed=episode_seed,
                tile_size=tile_size,
                max_steps=spec.max_steps,
            )
            env_cache[key] = env
        return env_cache[key]

    episode_ids = []
    with metadata_path.open("a", encoding="utf-8") as fout:
        for episode_id in tqdm(range(num_episodes), desc="Collecting expert data"):
            episode_seed = seed + episode_id
            spec = _sample_env_spec(rng, env_specs)

            env = get_env(spec, episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            env_size = int(env.unwrapped.width)

            done = False
            step_idx = 0
            episode_ids.append(episode_id)

            while not done:
                action = int(oracle_action(env.unwrapped))
                image_rel_path = f"images/ep_{episode_id:06d}_step_{step_idx:04d}.png"
                image_abs_path = output_dir / image_rel_path
                Image.fromarray(obs["image"]).save(image_abs_path)

                prompt_variant_id = choose_prompt_variant(
                    use_prompt_variants=use_prompt_variants,
                    prompt_variant_count=prompt_variant_count,
                    rng=rng,
                )

                row = {
                    "episode_id": episode_id,
                    "step_idx": step_idx,
                    "seed": episode_seed,
                    "env_id": spec.env_id,
                    "env_size": env_size,
                    "mission_text": obs.get("mission", ""),
                    "image_path": image_rel_path,
                    "action_id": action,
                    "action_token": action_id_to_token(action),
                    "prompt_variant_id": prompt_variant_id,
                }
                fout.write(json.dumps(row, ensure_ascii=True) + "\n")

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step_idx += 1

    for env in env_cache.values():
        env.close()

    split_rng = random.Random(split_seed)
    split_rng.shuffle(episode_ids)
    cut = int(len(episode_ids) * train_split)
    train_ids = sorted(episode_ids[:cut])
    val_ids = sorted(episode_ids[cut:])

    with (splits_dir / "train_ids.txt").open("w", encoding="utf-8") as f:
        for i in train_ids:
            f.write(f"{i}\n")

    with (splits_dir / "val_ids.txt").open("w", encoding="utf-8") as f:
        for i in val_ids:
            f.write(f"{i}\n")

    print(f"Saved dataset to: {output_dir}")
    print(f"Train episodes: {len(train_ids)}, Val episodes: {len(val_ids)}")

    if args.check_expert:
        eval_env_id = str(env_cfg.get("env_id", env_specs[0].env_id))
        rate = evaluate_expert_success(
            env_id=eval_env_id,
            episodes=200,
            seed=seed,
            tile_size=tile_size,
            max_steps=env_cfg.get("max_steps", None),
        )
        print(f"Expert success over 200 eval episodes on {eval_env_id}: {rate:.4f}")


if __name__ == "__main__":
    main()
