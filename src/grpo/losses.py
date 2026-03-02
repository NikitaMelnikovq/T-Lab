from __future__ import annotations

import torch

from src.model.policy_head import kl_schulman_approx, ppo_clipped_loss


def grpo_action_loss(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantage: torch.Tensor,
    epsilon: float,
    beta: float,
    ref_logprob: torch.Tensor | None = None,
):
    ratio = torch.exp(new_logprob - old_logprob)
    ppo = ppo_clipped_loss(ratio, advantage, epsilon)

    if ref_logprob is None or beta <= 0:
        kl = torch.zeros_like(ppo)
    else:
        kl = kl_schulman_approx(ref_logprob, new_logprob)

    loss = ppo + beta * kl
    return loss, ppo, kl, ratio


def grpo_text_action_token_loss(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    ref_logprob: torch.Tensor | None,
    advantage: torch.Tensor,
    epsilon: float,
    beta: float,
    token_weight: torch.Tensor,
):
    ratio = torch.exp(new_logprob - old_logprob)
    ppo = ppo_clipped_loss(ratio, advantage, epsilon) * token_weight

    if ref_logprob is None or beta <= 0:
        kl = torch.zeros_like(ppo)
    else:
        kl = kl_schulman_approx(ref_logprob, new_logprob) * token_weight

    loss = ppo + beta * kl
    return loss, ppo, kl, ratio
