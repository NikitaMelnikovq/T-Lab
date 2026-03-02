#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import get_device, load_yaml
from src.common.seed import set_seed
from src.data.transforms import format_action_prompt
from src.env.expert import oracle_action
from src.env.make_env import make_env
from src.grpo.rollout import preprocess_obs_image
from src.model.action_tokenizer import build_action_mapping
from src.model.generation import choose_action
from src.model.nanovlm_loader import load_nanovlm


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline smoke test")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def check_env(cfg):
    env, obs, _ = make_env(
        env_id=cfg["env"]["env_id"],
        seed=int(cfg["experiment"]["seed"]),
        tile_size=int(cfg["env"].get("tile_size", 8)),
        max_steps=cfg["env"].get("max_steps", None),
    )
    action = int(oracle_action(env.unwrapped))
    obs2, reward, term, trunc, _ = env.step(action)
    env.close()
    return {
        "first_action": action,
        "reward_after_step": float(reward),
        "done": bool(term or trunc),
        "mission": obs.get("mission", ""),
    }


def check_model_batch(cfg):
    device = get_device(cfg["experiment"].get("device", "auto"))
    model, tokenizer = load_nanovlm(
        source=cfg["model"].get("nanovlm_source", "lusxvr/nanoVLM-222M"),
        tokenizer_name=cfg["model"].get("tokenizer_name", "HuggingFaceTB/cosmo2-tokenizer"),
        device=device,
        freeze_backbones=True,
    )
    action_mapping = build_action_mapping(tokenizer)

    env, obs, _ = make_env(
        env_id=cfg["env"]["env_id"],
        seed=int(cfg["experiment"]["seed"]),
        tile_size=int(cfg["env"].get("tile_size", 8)),
        max_steps=cfg["env"].get("max_steps", None),
    )
    image = preprocess_obs_image(obs["image"])
    prompt = format_action_prompt(obs.get("mission", ""), variant_id=0)
    out = choose_action(
        model=model,
        tokenizer=tokenizer,
        action_mapping=action_mapping,
        prompt_text=prompt,
        image_tensor=image,
        device=device,
        greedy=False,
    )
    env.close()
    return {"sampled_action": int(out["action_id"]), "logprob": float(out["logprob"])}


def check_expert_two_eps(cfg):
    seed = int(cfg["experiment"]["seed"])
    actions_runs = []
    for _ in range(2):
        env, obs, _ = make_env(
            env_id=cfg["env"]["env_id"],
            seed=seed,
            tile_size=int(cfg["env"].get("tile_size", 8)),
            max_steps=cfg["env"].get("max_steps", None),
        )
        done = False
        seq = []
        while not done:
            a = int(oracle_action(env.unwrapped))
            seq.append(a)
            obs, reward, term, trunc, _ = env.step(a)
            done = term or trunc
        env.close()
        actions_runs.append(seq)

    return {
        "ep_len": len(actions_runs[0]),
        "deterministic": actions_runs[0] == actions_runs[1],
        "first_actions": actions_runs[0][:5],
    }


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["experiment"]["seed"]))

    env_res = check_env(cfg)
    model_res = check_model_batch(cfg)
    expert_res = check_expert_two_eps(cfg)

    print("[SMOKE] env:", env_res)
    print("[SMOKE] model:", model_res)
    print("[SMOKE] expert:", expert_res)

    if not expert_res["deterministic"]:
        raise SystemExit("Determinism check failed")


if __name__ == "__main__":
    main()
