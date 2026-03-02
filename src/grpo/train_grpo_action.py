from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from src.common.checkpoints import save_best, save_last
from src.common.config import ensure_dir, get_device, load_yaml
from src.common.logging import ExperimentLogger, setup_console_logger
from src.common.seed import set_seed
from src.grpo.grpo_core import update_policy_action
from src.grpo.rollout import collect_grouped_rollouts_action
from src.model.action_tokenizer import build_action_mapping
from src.model.nanovlm_loader import get_trainable_parameters, load_nanovlm
from src.sft.eval_sft import evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRPO (action-only)")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    exp_cfg = cfg["experiment"]
    env_cfg = cfg["env"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    log_cfg = cfg["logging"]

    seed = int(exp_cfg["seed"])
    set_seed(seed)

    run_dir = ensure_dir(exp_cfg["output_dir"])
    ensure_dir(run_dir / "checkpoints")
    (run_dir / "used_config.yaml").write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")

    logger = setup_console_logger("grpo_action")
    ex_logger = ExperimentLogger(
        run_dir=run_dir,
        use_tensorboard=bool(log_cfg.get("use_tensorboard", False)),
        use_wandb=bool(log_cfg.get("use_wandb", False)),
        wandb_run_name=exp_cfg.get("name", "grpo_action"),
    )

    device = get_device(exp_cfg.get("device", "auto"))
    logger.info(f"Using device: {device}")

    init_source = model_cfg.get("init_checkpoint") or model_cfg.get("nanovlm_source", "lusxvr/nanoVLM-222M")
    ref_source = model_cfg.get("reference_checkpoint") or init_source

    policy_model, tokenizer = load_nanovlm(
        source=init_source,
        tokenizer_name=model_cfg.get("tokenizer_name", "HuggingFaceTB/cosmo2-tokenizer"),
        device=device,
        freeze_backbones=bool(model_cfg.get("freeze_backbones", True)),
        unfreeze_embeddings=bool(model_cfg.get("unfreeze_embeddings", True)),
        unfreeze_lm_head=bool(model_cfg.get("unfreeze_lm_head", True)),
    )
    ref_model, _ = load_nanovlm(
        source=ref_source,
        tokenizer_name=model_cfg.get("tokenizer_name", "HuggingFaceTB/cosmo2-tokenizer"),
        device=device,
        freeze_backbones=False,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    action_mapping = build_action_mapping(tokenizer)
    params = get_trainable_parameters(policy_model)

    optimizer = torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    iterations = int(train_cfg.get("iterations", 120))
    group_size = int(train_cfg.get("group_size", 4))
    prompts_per_iter = int(train_cfg.get("prompts_per_iter", 8))
    ppo_epochs = int(train_cfg.get("ppo_epochs", 2))
    minibatch_size = int(train_cfg.get("minibatch_size", 32))
    epsilon = float(train_cfg.get("epsilon", 0.2))
    beta = float(train_cfg.get("beta", 0.02))
    temperature = float(train_cfg.get("temperature", 1.0))
    eval_every = int(train_cfg.get("eval_every_iters", 10))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    env_steps_total = 0
    episodes_total = 0
    best_success = -1.0
    steps_to_08 = None
    steps_to_09 = None
    start_time = time.time()

    for it in range(1, iterations + 1):
        group_seeds = [seed + it * 1000 + i for i in range(prompts_per_iter)]

        transitions, rollout_stats = collect_grouped_rollouts_action(
            policy_model=policy_model,
            tokenizer=tokenizer,
            action_mapping=action_mapping,
            env_id=env_cfg["env_id"],
            group_seeds=group_seeds,
            group_size=group_size,
            device=device,
            tile_size=int(env_cfg.get("tile_size", 8)),
            max_steps=env_cfg.get("max_steps", None),
            temperature=temperature,
        )

        update_stats = update_policy_action(
            model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            action_mapping=action_mapping,
            transitions=transitions,
            device=device,
            epsilon=epsilon,
            beta=beta,
            minibatch_size=minibatch_size,
            ppo_epochs=ppo_epochs,
            max_grad_norm=max_grad_norm,
        )

        env_steps_total += rollout_stats.env_steps
        episodes_total += rollout_stats.episodes

        payload = {
            "phase": "train",
            "iter": it,
            "global_step": it,
            "loss": update_stats["loss"],
            "kl_to_ref": update_stats["kl_to_ref"],
            "entropy": update_stats["entropy"],
            "clip_frac": update_stats["clip_frac"],
            "mean_return": rollout_stats.mean_return,
            "success_rate": rollout_stats.success_rate,
            "avg_return": rollout_stats.mean_return,
            "env_steps": env_steps_total,
            "episodes": episodes_total,
            "wallclock": time.time() - start_time,
        }
        ex_logger.log(payload)

        if it % int(log_cfg.get("console_every_steps", 1)) == 0:
            logger.info(
                "iter=%d loss=%.4f mean_return=%.4f success=%.4f env_steps=%d",
                it,
                update_stats["loss"],
                rollout_stats.mean_return,
                rollout_stats.success_rate,
                env_steps_total,
            )

        if it % eval_every == 0:
            policy_model.eval()
            eval_metrics = evaluate_policy(
                model=policy_model,
                tokenizer=tokenizer,
                action_mapping=action_mapping,
                env_id=env_cfg["env_id"],
                eval_episodes=int(env_cfg.get("eval_episodes", 500)),
                seed=seed + 50_000 + it,
                device=device,
                tile_size=int(env_cfg.get("tile_size", 8)),
                max_steps=env_cfg.get("max_steps", None),
            )
            policy_model.train()

            if steps_to_08 is None and eval_metrics["success_rate"] >= 0.8:
                steps_to_08 = env_steps_total
            if steps_to_09 is None and eval_metrics["success_rate"] >= 0.9:
                steps_to_09 = env_steps_total

            eval_payload = {
                "phase": "eval",
                "iter": it,
                "global_step": it,
                "loss": update_stats["loss"],
                "success_rate": eval_metrics["success_rate"],
                "avg_return": eval_metrics["avg_return"],
                "avg_steps_to_goal": eval_metrics["avg_steps_to_goal"],
                "env_steps": env_steps_total,
                "episodes": episodes_total,
                "env_steps_to_0.8": -1 if steps_to_08 is None else steps_to_08,
                "env_steps_to_0.9": -1 if steps_to_09 is None else steps_to_09,
                "wallclock": time.time() - start_time,
            }
            ex_logger.log(eval_payload)
            logger.info(f"eval iter={it} metrics={eval_metrics}")

            if eval_metrics["success_rate"] > best_success:
                best_success = eval_metrics["success_rate"]
                save_best(
                    model=policy_model,
                    optimizer=optimizer,
                    step=it,
                    metric_name="success_rate",
                    metric_value=best_success,
                    ckpt_dir=run_dir / "checkpoints",
                )

    save_last(model=policy_model, optimizer=optimizer, step=iterations, payload={}, ckpt_dir=run_dir / "checkpoints")

    summary = {
        "env_steps": env_steps_total,
        "episodes": episodes_total,
        "env_steps_to_0.8": steps_to_08,
        "env_steps_to_0.9": steps_to_09,
        "best_success_rate": best_success,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    logger.info(f"GRPO action summary: {summary}")
    ex_logger.close()


if __name__ == "__main__":
    main()
