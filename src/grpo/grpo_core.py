from __future__ import annotations

import random
from typing import Dict, List

import torch

from src.grpo.losses import grpo_action_loss, grpo_text_action_token_loss
from src.model.generation import sequence_logprobs
from src.model.policy_head import gather_action_logits, masked_action_distribution


def _action_logprob_entropy(model, tokenizer, action_mapping, prompt: str, image: torch.Tensor, action_token_id: int, device: str):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attn = encoded["attention_mask"].to(device)
    img = image.unsqueeze(0).to(device)

    hidden, _ = model(input_ids, img, attention_mask=attn, targets=None)
    vocab_logits = model.decoder.head(hidden[:, -1, :])
    probs, log_probs = masked_action_distribution(vocab_logits, action_mapping.action_token_ids)

    local_idx = action_mapping.action_token_ids.index(int(action_token_id))
    lp = log_probs[0, local_idx]
    ent = -(probs * log_probs).sum(dim=-1)[0]
    return lp, ent


def update_policy_action(
    model,
    ref_model,
    optimizer,
    tokenizer,
    action_mapping,
    transitions: List[Dict],
    device: str,
    epsilon: float,
    beta: float,
    minibatch_size: int,
    ppo_epochs: int,
    max_grad_norm: float,
):
    model.train()
    losses = []
    entropies = []
    kls = []
    clip_hits = []

    for _ in range(ppo_epochs):
        random.shuffle(transitions)
        for i in range(0, len(transitions), minibatch_size):
            mb = transitions[i : i + minibatch_size]
            if not mb:
                continue

            optimizer.zero_grad(set_to_none=True)
            mb_loss = 0.0
            mb_entropy = 0.0
            mb_kl = 0.0
            mb_clip = 0.0

            for tr in mb:
                new_lp, entropy = _action_logprob_entropy(
                    model, tokenizer, action_mapping, tr["prompt"], tr["image"], tr["action_token_id"], device
                )
                old_lp = torch.tensor(tr["old_logprob"], device=device, dtype=new_lp.dtype)
                adv = torch.tensor(tr["advantage"], device=device, dtype=new_lp.dtype)

                if ref_model is not None:
                    with torch.no_grad():
                        ref_lp, _ = _action_logprob_entropy(
                            ref_model,
                            tokenizer,
                            action_mapping,
                            tr["prompt"],
                            tr["image"],
                            tr["action_token_id"],
                            device,
                        )
                else:
                    ref_lp = None

                loss, ppo_term, kl_term, ratio = grpo_action_loss(
                    new_logprob=new_lp,
                    old_logprob=old_lp,
                    advantage=adv,
                    epsilon=epsilon,
                    beta=beta,
                    ref_logprob=ref_lp,
                )

                mb_loss = mb_loss + loss
                mb_entropy = mb_entropy + entropy
                mb_kl = mb_kl + kl_term
                mb_clip = mb_clip + ((ratio < (1 - epsilon)) | (ratio > (1 + epsilon))).float()

            mb_loss = mb_loss / len(mb)
            mb_entropy = mb_entropy / len(mb)
            mb_kl = mb_kl / len(mb)
            mb_clip = mb_clip / len(mb)

            mb_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()

            losses.append(float(mb_loss.item()))
            entropies.append(float(mb_entropy.item()))
            kls.append(float(mb_kl.item()))
            clip_hits.append(float(mb_clip.item()))

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "entropy": sum(entropies) / max(1, len(entropies)),
        "kl_to_ref": sum(kls) / max(1, len(kls)),
        "clip_frac": sum(clip_hits) / max(1, len(clip_hits)),
    }


def update_policy_text_action(
    model,
    ref_model,
    optimizer,
    tokenizer,
    transitions: List[Dict],
    device: str,
    epsilon: float,
    beta: float,
    minibatch_size: int,
    ppo_epochs: int,
    max_grad_norm: float,
    text_loss_weight: float,
):
    model.train()

    losses = []
    kls = []
    clip_hits = []

    for _ in range(ppo_epochs):
        random.shuffle(transitions)
        for i in range(0, len(transitions), minibatch_size):
            mb = transitions[i : i + minibatch_size]
            if not mb:
                continue

            optimizer.zero_grad(set_to_none=True)
            mb_loss = 0.0
            mb_kl = 0.0
            mb_clip = 0.0

            for tr in mb:
                new_lps = sequence_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=tr["prompt"],
                    image_tensor=tr["image"],
                    generated_ids=tr["generated_ids"],
                    device=device,
                )

                if ref_model is not None:
                    with torch.no_grad():
                        ref_lps = sequence_logprobs(
                            model=ref_model,
                            tokenizer=tokenizer,
                            prompt_text=tr["prompt"],
                            image_tensor=tr["image"],
                            generated_ids=tr["generated_ids"],
                            device=device,
                        )
                else:
                    ref_lps = [None for _ in tr["generated_ids"]]

                adv = torch.tensor(tr["advantage"], device=device, dtype=new_lps[0].dtype if new_lps else torch.float32)
                token_losses = []
                token_kls = []
                token_clips = []

                for j, new_lp in enumerate(new_lps):
                    old_lp = torch.tensor(tr["generated_old_logprobs"][j], device=device, dtype=new_lp.dtype)
                    ref_lp = ref_lps[j] if ref_lps[j] is not None else None
                    is_action = int(tr["is_action_token"][j])
                    w = 1.0 if is_action else float(text_loss_weight)
                    w_t = torch.tensor(w, device=device, dtype=new_lp.dtype)

                    loss_t, _, kl_t, ratio_t = grpo_text_action_token_loss(
                        new_logprob=new_lp,
                        old_logprob=old_lp,
                        ref_logprob=ref_lp,
                        advantage=adv,
                        epsilon=epsilon,
                        beta=beta,
                        token_weight=w_t,
                    )
                    token_losses.append(loss_t)
                    token_kls.append(kl_t)
                    token_clips.append(((ratio_t < (1 - epsilon)) | (ratio_t > (1 + epsilon))).float())

                if token_losses:
                    mb_loss = mb_loss + torch.stack(token_losses).mean()
                    mb_kl = mb_kl + torch.stack(token_kls).mean()
                    mb_clip = mb_clip + torch.stack(token_clips).float().mean()

            mb_loss = mb_loss / len(mb)
            mb_kl = mb_kl / len(mb)
            mb_clip = mb_clip / len(mb)

            mb_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()

            losses.append(float(mb_loss.item()))
            kls.append(float(mb_kl.item()))
            clip_hits.append(float(mb_clip.item()))

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "entropy": 0.0,
        "kl_to_ref": sum(kls) / max(1, len(kls)),
        "clip_frac": sum(clip_hits) / max(1, len(clip_hits)),
    }
