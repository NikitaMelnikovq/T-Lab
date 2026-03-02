from __future__ import annotations

from typing import List, Tuple

import torch


def gather_action_logits(vocab_logits: torch.Tensor, action_token_ids: List[int]) -> torch.Tensor:
    idx = torch.tensor(action_token_ids, device=vocab_logits.device, dtype=torch.long)
    return torch.index_select(vocab_logits, dim=-1, index=idx)


def masked_action_distribution(vocab_logits: torch.Tensor, action_token_ids: List[int], temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    action_logits = gather_action_logits(vocab_logits, action_token_ids)
    if temperature <= 0:
        temperature = 1.0
    scaled = action_logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    log_probs = torch.log_softmax(scaled, dim=-1)
    return probs, log_probs


def sample_action(
    vocab_logits: torch.Tensor,
    action_token_ids: List[int],
    greedy: bool,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs, log_probs = masked_action_distribution(vocab_logits, action_token_ids, temperature=temperature)

    if greedy:
        local_idx = torch.argmax(probs, dim=-1)
    else:
        local_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)

    chosen_logprob = log_probs.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return local_idx, chosen_logprob, entropy


def ppo_clipped_loss(ratio: torch.Tensor, advantage: torch.Tensor, epsilon: float) -> torch.Tensor:
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
    return -torch.minimum(unclipped, clipped)


def kl_schulman_approx(logp_ref: torch.Tensor, logp_cur: torch.Tensor) -> torch.Tensor:
    ref_over_cur = torch.exp(logp_ref - logp_cur)
    return ref_over_cur - (logp_ref - logp_cur) - 1.0
