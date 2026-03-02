#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import ensure_dir, load_yaml
from src.common.seed import set_seed
from src.data.transforms import choose_prompt_variant
from src.env.expert import evaluate_expert_success, oracle_action
from src.env.make_env import make_env
from src.model.action_tokenizer import action_id_to_token


def parse_args():
    parser = argparse.ArgumentParser(description="Collect expert trajectories for MiniGrid EmptyEnv")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--check_expert", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed, deterministic=True)

    env_cfg = cfg["env"]
    data_cfg = cfg["data"]

    num_episodes = int(args.episodes or data_cfg.get("num_expert_episodes", 2000))
    output_dir = Path(args.output_dir or data_cfg["dataset_dir"])
    images_dir = ensure_dir(output_dir / "images")
    splits_dir = ensure_dir(output_dir / "splits")

    metadata_path = output_dir / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    env_id = env_cfg.get("env_id", "MiniGrid-Empty-8x8-v0")
    tile_size = int(env_cfg.get("tile_size", 8))
    max_steps = env_cfg.get("max_steps", None)
    use_prompt_variants = bool(data_cfg.get("use_prompt_variants", True))
    prompt_variant_count = int(data_cfg.get("prompt_variant_count", 3))
    split_seed = int(data_cfg.get("split_seed", seed))
    train_split = float(data_cfg.get("train_split", 0.9))

    rng = random.Random(seed)

    env, obs, _ = make_env(env_id=env_id, seed=seed, tile_size=tile_size, max_steps=max_steps)
    env_size = int(env.unwrapped.width)

    episode_ids = []
    with metadata_path.open("a", encoding="utf-8") as fout:
        for episode_id in tqdm(range(num_episodes), desc="Collecting expert data"):
            episode_seed = seed + episode_id
            obs, _ = env.reset(seed=episode_seed)
            done = False
            step_idx = 0
            episode_ids.append(episode_id)

            while not done:
                action = int(oracle_action(env.unwrapped))
                img = Image.fromarray(obs["image"])
                image_rel_path = f"images/ep_{episode_id:06d}_step_{step_idx:04d}.png"
                image_abs_path = output_dir / image_rel_path
                img.save(image_abs_path)

                prompt_variant_id = choose_prompt_variant(
                    use_prompt_variants=use_prompt_variants,
                    prompt_variant_count=prompt_variant_count,
                    rng=rng,
                )

                row = {
                    "episode_id": episode_id,
                    "step_idx": step_idx,
                    "seed": episode_seed,
                    "env_id": env_id,
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
        rate = evaluate_expert_success(
            env_id=env_id,
            episodes=200,
            seed=seed,
            tile_size=tile_size,
            max_steps=max_steps,
        )
        print(f"Expert success over 200 eval episodes: {rate:.4f}")


if __name__ == "__main__":
    main()
