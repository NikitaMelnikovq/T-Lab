from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from src.common.config import get_device, load_yaml
from src.common.seed import set_seed
from src.data.transforms import format_action_prompt
from src.env.make_env import make_env
from src.model.action_tokenizer import build_action_mapping
from src.model.generation import choose_action
from src.model.nanovlm_loader import load_nanovlm


def evaluate_policy(
    model,
    tokenizer,
    action_mapping,
    env_id: str,
    eval_episodes: int,
    seed: int,
    device: str,
    tile_size: int = 8,
    max_steps=None,
    show_progress: bool = True,
) -> Dict[str, float]:
    successes = 0
    returns = []
    steps_to_goal = []

    iterator = tqdm(range(eval_episodes), desc="Evaluating", leave=False, disable=not show_progress)
    for ep in iterator:
        env, obs, _ = make_env(
            env_id=env_id,
            seed=seed + ep,
            tile_size=tile_size,
            max_steps=max_steps,
        )
        done = False
        ep_return = 0.0
        ep_steps = 0

        while not done:
            prompt = format_action_prompt(obs.get("mission", ""), variant_id=0)
            image = torch.tensor(obs["image"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze(0)
            image = (image - 0.5) / 0.5

            out = choose_action(
                model=model,
                tokenizer=tokenizer,
                action_mapping=action_mapping,
                prompt_text=prompt,
                image_tensor=image,
                device=device,
                greedy=True,
                temperature=1.0,
            )

            obs, reward, terminated, truncated, _ = env.step(int(out["action_id"]))
            ep_return += float(reward)
            ep_steps += 1
            done = terminated or truncated

            if terminated and reward > 0:
                successes += 1

        returns.append(ep_return)
        steps_to_goal.append(ep_steps)
        env.close()

    n = max(1, eval_episodes)
    return {
        "success_rate": successes / n,
        "avg_return": sum(returns) / n,
        "avg_steps_to_goal": sum(steps_to_goal) / n,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

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
    mapping = build_action_mapping(tokenizer)

    metrics = evaluate_policy(
        model=model,
        tokenizer=tokenizer,
        action_mapping=mapping,
        env_id=env_cfg["env_id"],
        eval_episodes=int(args.episodes or env_cfg.get("eval_episodes", 500)),
        seed=seed + 10_000,
        device=device,
        tile_size=int(env_cfg.get("tile_size", 8)),
        max_steps=env_cfg.get("max_steps", None),
    )
    print(metrics)


if __name__ == "__main__":
    main()
