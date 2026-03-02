from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from src.data.transforms import format_action_prompt, format_text_action_prompt
from src.env.make_env import make_env
from src.model.generation import choose_action, generate_plan_then_action


@dataclass
class RolloutStats:
    mean_return: float
    success_rate: float
    mean_episode_steps: float
    env_steps: int
    episodes: int
    parse_failures: int = 0


def preprocess_obs_image(obs_img) -> torch.Tensor:
    image = torch.tensor(obs_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)
    image = (image - 0.5) / 0.5
    return image


def _run_episode_action(policy_model, tokenizer, action_mapping, env_id, env_seed, device, tile_size, max_steps, temperature):
    env, obs, _ = make_env(env_id=env_id, seed=env_seed, tile_size=tile_size, max_steps=max_steps)
    done = False
    transitions = []
    total_return = 0.0
    ep_steps = 0
    success = 0

    while not done:
        prompt = format_action_prompt(obs.get("mission", ""), variant_id=0)
        image_t = preprocess_obs_image(obs["image"])

        out = choose_action(
            model=policy_model,
            tokenizer=tokenizer,
            action_mapping=action_mapping,
            prompt_text=prompt,
            image_tensor=image_t,
            device=device,
            greedy=False,
            temperature=temperature,
        )
        next_obs, reward, terminated, truncated, _ = env.step(out["action_id"])
        done = terminated or truncated

        transitions.append(
            {
                "prompt": prompt,
                "image": image_t,
                "action_id": int(out["action_id"]),
                "action_token_id": int(out["action_token_id"]),
                "old_logprob": float(out["logprob"]),
                "entropy": float(out["entropy"]),
            }
        )

        total_return += float(reward)
        ep_steps += 1
        if terminated and reward > 0:
            success = 1

        obs = next_obs

    env.close()
    return {
        "transitions": transitions,
        "episode_return": total_return,
        "episode_steps": ep_steps,
        "success": success,
        "parse_failures": 0,
    }


def _run_episode_text_action(policy_model, tokenizer, action_mapping, env_id, env_seed, device, tile_size, max_steps, temperature, max_plan_tokens):
    env, obs, _ = make_env(env_id=env_id, seed=env_seed, tile_size=tile_size, max_steps=max_steps)
    done = False
    transitions = []
    total_return = 0.0
    ep_steps = 0
    success = 0
    parse_failures = 0

    while not done:
        prompt = format_text_action_prompt(obs.get("mission", ""))
        image_t = preprocess_obs_image(obs["image"])

        out = generate_plan_then_action(
            model=policy_model,
            tokenizer=tokenizer,
            action_mapping=action_mapping,
            prompt_text=prompt,
            image_tensor=image_t,
            device=device,
            max_plan_tokens=max_plan_tokens,
            temperature=temperature,
            greedy=False,
        )

        next_obs, reward, terminated, truncated, _ = env.step(int(out["action_id"]))
        done = terminated or truncated

        if out["parse_failed"]:
            reward -= 0.1
            parse_failures += 1

        transitions.append(
            {
                "prompt": prompt,
                "image": image_t,
                "action_id": int(out["action_id"]),
                "generated_ids": list(out["generated_ids"]),
                "generated_old_logprobs": list(out["generated_logprobs"]),
                "is_action_token": list(out["is_action_token"]),
                "parse_failed": bool(out["parse_failed"]),
                "text": out["text"],
            }
        )

        total_return += float(reward)
        ep_steps += 1
        if terminated and reward > 0:
            success = 1

        obs = next_obs

    env.close()
    return {
        "transitions": transitions,
        "episode_return": total_return,
        "episode_steps": ep_steps,
        "success": success,
        "parse_failures": parse_failures,
    }


def collect_grouped_rollouts_action(
    policy_model,
    tokenizer,
    action_mapping,
    env_id: str,
    group_seeds: List[int],
    group_size: int,
    device: str,
    tile_size: int,
    max_steps,
    temperature: float,
):
    all_transitions = []
    all_returns = []
    all_success = []
    all_steps = []

    for base_seed in group_seeds:
        group_eps = []
        rewards = []
        for _ in range(group_size):
            ep = _run_episode_action(
                policy_model,
                tokenizer,
                action_mapping,
                env_id,
                base_seed,
                device,
                tile_size,
                max_steps,
                temperature,
            )
            group_eps.append(ep)
            rewards.append(ep["episode_return"])

        r_mean = float(torch.tensor(rewards).mean().item())
        r_std = float(torch.tensor(rewards).std(unbiased=False).item())

        for ep, r in zip(group_eps, rewards):
            advantage = (r - r_mean) / (r_std + 1e-8)
            for t in ep["transitions"]:
                t["advantage"] = float(advantage)
                all_transitions.append(t)

            all_returns.append(ep["episode_return"])
            all_success.append(ep["success"])
            all_steps.append(ep["episode_steps"])

    stats = RolloutStats(
        mean_return=float(sum(all_returns) / max(1, len(all_returns))),
        success_rate=float(sum(all_success) / max(1, len(all_success))),
        mean_episode_steps=float(sum(all_steps) / max(1, len(all_steps))),
        env_steps=int(sum(all_steps)),
        episodes=len(all_steps),
    )
    return all_transitions, stats


def collect_grouped_rollouts_text_action(
    policy_model,
    tokenizer,
    action_mapping,
    env_id: str,
    group_seeds: List[int],
    group_size: int,
    device: str,
    tile_size: int,
    max_steps,
    temperature: float,
    max_plan_tokens: int,
):
    all_transitions = []
    all_returns = []
    all_success = []
    all_steps = []
    total_parse_failures = 0

    for base_seed in group_seeds:
        group_eps = []
        rewards = []
        for _ in range(group_size):
            ep = _run_episode_text_action(
                policy_model,
                tokenizer,
                action_mapping,
                env_id,
                base_seed,
                device,
                tile_size,
                max_steps,
                temperature,
                max_plan_tokens,
            )
            group_eps.append(ep)
            rewards.append(ep["episode_return"])

        r_mean = float(torch.tensor(rewards).mean().item())
        r_std = float(torch.tensor(rewards).std(unbiased=False).item())

        for ep, r in zip(group_eps, rewards):
            advantage = (r - r_mean) / (r_std + 1e-8)
            for t in ep["transitions"]:
                t["advantage"] = float(advantage)
                all_transitions.append(t)

            all_returns.append(ep["episode_return"])
            all_success.append(ep["success"])
            all_steps.append(ep["episode_steps"])
            total_parse_failures += int(ep.get("parse_failures", 0))

    stats = RolloutStats(
        mean_return=float(sum(all_returns) / max(1, len(all_returns))),
        success_rate=float(sum(all_success) / max(1, len(all_success))),
        mean_episode_steps=float(sum(all_steps) / max(1, len(all_steps))),
        env_steps=int(sum(all_steps)),
        episodes=len(all_steps),
        parse_failures=total_parse_failures,
    )
    return all_transitions, stats
