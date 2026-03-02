from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple

import torch

from src.model.action_tokenizer import ACTION_TOKENS
from src.model.policy_head import masked_action_distribution, sample_action


def forward_next_token_logits(model, input_ids: torch.Tensor, image: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden, _ = model(input_ids, image, attention_mask=attention_mask, targets=None)
    last_hidden = hidden[:, -1, :]
    vocab_logits = model.decoder.head(last_hidden)
    return vocab_logits


def choose_action(
    model,
    tokenizer,
    action_mapping,
    prompt_text: str,
    image_tensor: torch.Tensor,
    device: str,
    greedy: bool = False,
    temperature: float = 1.0,
):
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    image = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        vocab_logits = forward_next_token_logits(model, input_ids, image, attention_mask)
        local_idx, logprob, entropy = sample_action(
            vocab_logits,
            action_mapping.action_token_ids,
            greedy=greedy,
            temperature=temperature,
        )

    local_idx = int(local_idx.item())
    action_token_id = action_mapping.action_token_ids[local_idx]
    action_id = action_mapping.token_id_to_action_id[action_token_id]
    return {
        "action_id": action_id,
        "action_token_id": action_token_id,
        "logprob": float(logprob.item()),
        "entropy": float(entropy.item()),
    }


def parse_action_token(text: str) -> str | None:
    m = re.search(r"Action:\s*(<ACT_[A-Z]+>)", text)
    if m is not None and m.group(1) in ACTION_TOKENS:
        return m.group(1)

    for tok in ACTION_TOKENS:
        if tok in text:
            return tok
    return None


def generate_plan_then_action(
    model,
    tokenizer,
    action_mapping,
    prompt_text: str,
    image_tensor: torch.Tensor,
    device: str,
    max_plan_tokens: int,
    temperature: float,
    greedy: bool = False,
):
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    image = image_tensor.unsqueeze(0).to(device)

    generated_ids: List[int] = []
    generated_logprobs: List[float] = []
    is_action_token: List[int] = []

    cur_ids = input_ids
    cur_mask = attention_mask

    # Plan token generation (unconstrained)
    for _ in range(max_plan_tokens):
        with torch.no_grad():
            logits = forward_next_token_logits(model, cur_ids, image, cur_mask)
            if greedy:
                next_id = torch.argmax(logits, dim=-1)
                next_logprob = torch.log_softmax(logits, dim=-1).gather(-1, next_id.unsqueeze(-1)).squeeze(-1)
            else:
                probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                next_logprob = torch.log(probs.gather(-1, next_id.unsqueeze(-1)).squeeze(-1) + 1e-12)

        token_id = int(next_id.item())
        if token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(token_id)
        generated_logprobs.append(float(next_logprob.item()))
        is_action_token.append(0)

        cur_ids = torch.cat([cur_ids, next_id.unsqueeze(-1)], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_id).unsqueeze(-1)], dim=1)

    # Constrained action token
    with torch.no_grad():
        logits = forward_next_token_logits(model, cur_ids, image, cur_mask)
        local_idx, action_logprob, _ = sample_action(
            logits,
            action_mapping.action_token_ids,
            greedy=greedy,
            temperature=temperature,
        )

    local_idx = int(local_idx.item())
    action_token_id = action_mapping.action_token_ids[local_idx]
    action_id = action_mapping.token_id_to_action_id[action_token_id]

    generated_ids.append(action_token_id)
    generated_logprobs.append(float(action_logprob.item()))
    is_action_token.append(1)

    plan_text = tokenizer.decode(generated_ids[:-1], skip_special_tokens=True).strip()
    action_token = tokenizer.convert_ids_to_tokens(action_token_id)
    output_text = f"Plan: {plan_text}\nAction: {action_token}"

    parsed = parse_action_token(output_text)
    parse_failed = parsed is None

    if parse_failed:
        action_id = random.randint(0, 6)
        action_token = ACTION_TOKENS[action_id]

    return {
        "text": output_text,
        "action_id": action_id,
        "action_token": action_token,
        "generated_ids": generated_ids,
        "generated_logprobs": generated_logprobs,
        "is_action_token": is_action_token,
        "parse_failed": parse_failed,
    }


def sequence_logprobs(model, tokenizer, prompt_text: str, image_tensor: torch.Tensor, generated_ids: List[int], device: str):
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True)
    cur_ids = encoded["input_ids"].to(device)
    cur_mask = encoded["attention_mask"].to(device)
    image = image_tensor.unsqueeze(0).to(device)

    out = []
    for token_id in generated_ids:
        logits = forward_next_token_logits(model, cur_ids, image, cur_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        lp = log_probs[0, int(token_id)]
        out.append(lp)

        next_t = torch.tensor([[int(token_id)]], device=device, dtype=cur_ids.dtype)
        cur_ids = torch.cat([cur_ids, next_t], dim=1)
        cur_mask = torch.cat([cur_mask, torch.ones((1, 1), device=device, dtype=cur_mask.dtype)], dim=1)

    return out
