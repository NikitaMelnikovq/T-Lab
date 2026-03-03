from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.common.checkpoints import save_best, save_last
from src.common.config import ensure_dir, get_device, load_yaml
from src.common.logging import ExperimentLogger, setup_console_logger
from src.common.schedulers import cosine_with_warmup_lr
from src.common.seed import set_seed
from src.data.collate import collate_action_batch
from src.data.dataset import ExpertStepDataset
from src.model.action_tokenizer import build_action_mapping
from src.model.nanovlm_loader import get_trainable_parameters, load_nanovlm
from src.model.policy_head import gather_action_logits
from src.sft.eval_sft import evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Train SFT policy on expert trajectories")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def compute_class_weights(train_ds, device: str):
    counts = train_ds.action_counts()
    total = sum(counts.values())
    weights = []
    for k in range(7):
        c = counts.get(k, 1)
        w = total / max(1, 7 * c)
        weights.append(w)
    return torch.tensor(weights, device=device, dtype=torch.float32)


def build_dataloader(cfg, dataset, tokenizer, action_mapping, split_name: str):
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    collate_fn = partial(
        collate_action_batch,
        tokenizer=tokenizer,
        max_prompt_length=int(model_cfg.get("max_prompt_length", 160)),
        action_mapping=action_mapping,
    )

    if split_name == "train" and data_cfg.get("balance_actions", False) and data_cfg.get("balance_mode", "weighted_ce") == "resample":
        counts = dataset.action_counts()
        sample_weights = []
        for row in dataset.rows:
            sample_weights.append(1.0 / max(1, counts[int(row.action_id)]))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = split_name == "train"

    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader


def evaluate_val_loss(model, val_loader, action_mapping, device: str, class_weights=None, max_batches: int | None = None):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            action_ids = batch["action_ids"].to(device)

            hidden, _ = model(input_ids, images, attention_mask=attention_mask, targets=None)
            vocab_logits = model.decoder.head(hidden[:, -1, :])
            action_logits = gather_action_logits(vocab_logits, action_mapping.action_token_ids)
            loss = torch.nn.functional.cross_entropy(action_logits, action_ids, weight=class_weights)
            losses.append(float(loss.item()))
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    model.train()
    if not losses:
        return 0.0
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    exp_cfg = cfg["experiment"]
    env_cfg = cfg["env"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    log_cfg = cfg["logging"]

    seed = int(exp_cfg["seed"])
    set_seed(seed)

    run_dir = ensure_dir(exp_cfg["output_dir"])
    ensure_dir(run_dir / "checkpoints")
    (run_dir / "used_config.yaml").write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")

    logger = setup_console_logger("sft")
    ex_logger = ExperimentLogger(
        run_dir=run_dir,
        use_tensorboard=bool(log_cfg.get("use_tensorboard", False)),
        use_wandb=bool(log_cfg.get("use_wandb", False)),
        wandb_run_name=exp_cfg.get("name", "sft"),
    )

    device = get_device(exp_cfg.get("device", "auto"))
    logger.info(f"Using device: {device}")

    model, tokenizer = load_nanovlm(
        source=model_cfg.get("nanovlm_source", "lusxvr/nanoVLM-222M"),
        tokenizer_name=model_cfg.get("tokenizer_name", "HuggingFaceTB/cosmo2-tokenizer"),
        device=device,
        freeze_backbones=bool(model_cfg.get("freeze_backbones", True)),
        unfreeze_embeddings=bool(model_cfg.get("unfreeze_embeddings", True)),
        unfreeze_lm_head=bool(model_cfg.get("unfreeze_lm_head", True)),
    )
    action_mapping = build_action_mapping(tokenizer)

    train_ds = ExpertStepDataset(
        dataset_dir=data_cfg["dataset_dir"],
        split="train",
        image_size=int(data_cfg.get("image_size", 224)),
        use_prompt_variants=bool(data_cfg.get("use_prompt_variants", True)),
    )
    val_ds = ExpertStepDataset(
        dataset_dir=data_cfg["dataset_dir"],
        split="val",
        image_size=int(data_cfg.get("image_size", 224)),
        use_prompt_variants=bool(data_cfg.get("use_prompt_variants", True)),
    )

    train_loader = build_dataloader(cfg, train_ds, tokenizer, action_mapping, split_name="train")
    val_loader = build_dataloader(cfg, val_ds, tokenizer, action_mapping, split_name="val")

    train_counts = train_ds.action_counts()
    val_counts = val_ds.action_counts()
    logger.info(f"Train action counts: {train_counts}")
    logger.info(f"Val action counts  : {val_counts}")
    nonzero_actions = sum(1 for _, c in train_counts.items() if c > 0)
    if nonzero_actions < 3:
        logger.warning(
            "Low action diversity in train split (nonzero action classes=%d). "
            "Use Empty-Random envs or mixed sizes in data.collect_envs for a less trivial benchmark.",
            nonzero_actions,
        )

    class_weights = None
    if data_cfg.get("balance_actions", False) and data_cfg.get("balance_mode", "weighted_ce") == "weighted_ce":
        class_weights = compute_class_weights(train_ds, device=device)

    params = get_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg.get("epochs", 2))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    total_steps = max(1, (len(train_loader) * epochs) // max(1, grad_accum))
    warmup_steps = int(train_cfg.get("warmup_steps", 50))
    eval_every = int(train_cfg.get("eval_every_steps", 100))
    quick_eval_episodes = int(train_cfg.get("val_episodes", 20))
    final_eval_episodes = int(env_cfg.get("eval_episodes", 200))
    early_stop_success_rate = float(train_cfg.get("early_stop_success_rate", 1.0))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 3))
    eval_show_progress = bool(train_cfg.get("eval_show_progress", False))
    val_loss_max_batches = train_cfg.get("val_loss_max_batches", None)
    if val_loss_max_batches is not None:
        val_loss_max_batches = int(val_loss_max_batches)
    console_every = int(log_cfg.get("console_every_steps", 20))

    global_step = 0
    best_success = -1.0
    running_loss = 0.0
    train_start = time.time()
    perfect_eval_streak = 0
    should_stop = False

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"SFT epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            action_ids = batch["action_ids"].to(device)

            hidden, _ = model(input_ids, images, attention_mask=attention_mask, targets=None)
            vocab_logits = model.decoder.head(hidden[:, -1, :])
            action_logits = gather_action_logits(vocab_logits, action_mapping.action_token_ids)
            loss = torch.nn.functional.cross_entropy(action_logits, action_ids, weight=class_weights)
            (loss / grad_accum).backward()

            running_loss += float(loss.item())

            if (batch_idx + 1) % grad_accum == 0:
                cur_lr = cosine_with_warmup_lr(
                    step=global_step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    base_lr=float(train_cfg.get("lr", 2e-4)),
                )
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr

                torch.nn.utils.clip_grad_norm_(params, float(train_cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % console_every == 0:
                    avg_loss = running_loss / max(1, console_every)
                    running_loss = 0.0
                    logger.info(f"step={global_step} train_loss={avg_loss:.4f} lr={cur_lr:.6g}")

                payload = {
                    "phase": "train",
                    "global_step": global_step,
                    "loss": float(loss.item()),
                    "lr": cur_lr,
                    "env_steps": 0,
                    "episodes": 0,
                    "success_rate": 0.0,
                    "avg_return": 0.0,
                }
                ex_logger.log(payload)

                if global_step % eval_every == 0:
                    val_loss = evaluate_val_loss(
                        model,
                        val_loader,
                        action_mapping,
                        device,
                        class_weights=class_weights,
                        max_batches=val_loss_max_batches,
                    )
                    eval_episodes = max(1, quick_eval_episodes)
                    rollout = evaluate_policy(
                        model=model,
                        tokenizer=tokenizer,
                        action_mapping=action_mapping,
                        env_id=env_cfg["env_id"],
                        eval_episodes=eval_episodes,
                        seed=seed + 20_000 + global_step,
                        device=device,
                        tile_size=int(env_cfg.get("tile_size", 8)),
                        max_steps=env_cfg.get("max_steps", None),
                        show_progress=eval_show_progress,
                    )

                    eval_payload = {
                        "phase": "eval",
                        "global_step": global_step,
                        "loss": val_loss,
                        "val_loss": val_loss,
                        "val_success_rate": rollout["success_rate"],
                        "success_rate": rollout["success_rate"],
                        "avg_return": rollout["avg_return"],
                        "avg_steps_to_goal": rollout["avg_steps_to_goal"],
                        "env_steps": eval_episodes * int(env_cfg.get("max_steps") or 4 * 8 * 8),
                        "episodes": eval_episodes,
                        "wallclock": time.time() - train_start,
                    }
                    ex_logger.log(eval_payload)
                    logger.info(
                        "eval step=%d val_loss=%.4f val_success=%.4f avg_return=%.4f",
                        global_step,
                        val_loss,
                        rollout["success_rate"],
                        rollout["avg_return"],
                    )

                    if rollout["success_rate"] > best_success:
                        best_success = rollout["success_rate"]
                        save_best(
                            model=model,
                            optimizer=optimizer,
                            step=global_step,
                            metric_name="val_success_rate",
                            metric_value=best_success,
                            ckpt_dir=run_dir / "checkpoints",
                        )

                    if rollout["success_rate"] >= early_stop_success_rate - 1e-12:
                        perfect_eval_streak += 1
                    else:
                        perfect_eval_streak = 0

                    if early_stop_patience > 0 and perfect_eval_streak >= early_stop_patience:
                        logger.info(
                            "Early stopping at step=%d after %d consecutive evals with success_rate >= %.4f",
                            global_step,
                            perfect_eval_streak,
                            early_stop_success_rate,
                        )
                        ex_logger.log(
                            {
                                "phase": "early_stop",
                                "global_step": global_step,
                                "loss": val_loss,
                                "success_rate": rollout["success_rate"],
                                "avg_return": rollout["avg_return"],
                                "avg_steps_to_goal": rollout["avg_steps_to_goal"],
                                "env_steps": eval_episodes * int(env_cfg.get("max_steps") or 4 * 8 * 8),
                                "episodes": eval_episodes,
                                "wallclock": time.time() - train_start,
                            }
                        )
                        should_stop = True
                        break

        if should_stop:
            break

    save_last(model=model, optimizer=optimizer, step=global_step, payload={}, ckpt_dir=run_dir / "checkpoints")

    # Final full evaluation
    final_metrics = evaluate_policy(
        model=model,
        tokenizer=tokenizer,
        action_mapping=action_mapping,
        env_id=env_cfg["env_id"],
        eval_episodes=final_eval_episodes,
        seed=seed + 30_000,
        device=device,
        tile_size=int(env_cfg.get("tile_size", 8)),
        max_steps=env_cfg.get("max_steps", None),
        show_progress=True,
    )
    ex_logger.log(
        {
            "phase": "final_eval",
            "global_step": global_step,
            "loss": 0.0,
            "success_rate": final_metrics["success_rate"],
            "avg_return": final_metrics["avg_return"],
            "avg_steps_to_goal": final_metrics["avg_steps_to_goal"],
            "env_steps": final_eval_episodes * int(env_cfg.get("max_steps") or 4 * 8 * 8),
            "episodes": final_eval_episodes,
            "wallclock": time.time() - train_start,
        }
    )

    logger.info(f"Final SFT metrics: {final_metrics}")
    ex_logger.close()


if __name__ == "__main__":
    main()
