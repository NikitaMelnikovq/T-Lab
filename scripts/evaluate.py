#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import get_device, load_yaml
from src.common.seed import set_seed
from src.data.transforms import format_text_action_prompt
from src.env.make_env import make_env
from src.grpo.rollout import preprocess_obs_image
from src.model.action_tokenizer import build_action_mapping
from src.model.generation import generate_plan_then_action
from src.model.nanovlm_loader import load_nanovlm
from src.sft.eval_sft import evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint for any method")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["sft", "grpo_action", "grpo_text_action"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def evaluate_text_action(
    model,
    tokenizer,
    action_mapping,
    env_id: str,
    eval_episodes: int,
    seed: int,
    device: str,
    tile_size: int,
    max_steps,
    max_plan_tokens: int = 24,
):
    success = 0
    total_return = 0.0
    total_steps = 0

    for ep in range(eval_episodes):
        env, obs, _ = make_env(env_id=env_id, seed=seed + ep, tile_size=tile_size, max_steps=max_steps)
        done = False
        while not done:
            prompt = format_text_action_prompt(obs.get("mission", ""))
            image = preprocess_obs_image(obs["image"])
            out = generate_plan_then_action(
                model=model,
                tokenizer=tokenizer,
                action_mapping=action_mapping,
                prompt_text=prompt,
                image_tensor=image,
                device=device,
                max_plan_tokens=max_plan_tokens,
                temperature=1.0,
                greedy=True,
            )
            obs, reward, terminated, truncated, _ = env.step(int(out["action_id"]))
            if out["parse_failed"]:
                reward -= 0.1
            total_return += float(reward)
            total_steps += 1
            done = terminated or truncated
            if terminated and reward > 0:
                success += 1
        env.close()

    n = max(1, eval_episodes)
    return {
        "success_rate": success / n,
        "avg_return": total_return / n,
        "avg_steps_to_goal": total_steps / n,
    }


def main():
    args = parse_args()

    default_cfg = {
        "sft": "configs/sft.yaml",
        "grpo_action": "configs/grpo_action.yaml",
        "grpo_text_action": "configs/grpo_text_action.yaml",
    }
    cfg_path = args.config or default_cfg[args.mode]
    cfg = load_yaml(cfg_path)

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    device = get_device(cfg["experiment"].get("device", "auto"))
    env_cfg = cfg["env"]
    model_cfg = cfg["model"]

    model, tokenizer = load_nanovlm(
        source=args.checkpoint,
        tokenizer_name=model_cfg.get("tokenizer_name", "HuggingFaceTB/cosmo2-tokenizer"),
        device=device,
        freeze_backbones=False,
    )
    model.eval()
    action_mapping = build_action_mapping(tokenizer)

    eval_episodes = int(args.episodes or env_cfg.get("eval_episodes", 500))

    if args.mode in {"sft", "grpo_action"}:
        metrics = evaluate_policy(
            model=model,
            tokenizer=tokenizer,
            action_mapping=action_mapping,
            env_id=env_cfg["env_id"],
            eval_episodes=eval_episodes,
            seed=seed + 70_000,
            device=device,
            tile_size=int(env_cfg.get("tile_size", 8)),
            max_steps=env_cfg.get("max_steps", None),
        )
    else:
        metrics = evaluate_text_action(
            model=model,
            tokenizer=tokenizer,
            action_mapping=action_mapping,
            env_id=env_cfg["env_id"],
            eval_episodes=eval_episodes,
            seed=seed + 70_000,
            device=device,
            tile_size=int(env_cfg.get("tile_size", 8)),
            max_steps=env_cfg.get("max_steps", None),
            max_plan_tokens=int(cfg["train"].get("max_plan_tokens", 24)),
        )

    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
