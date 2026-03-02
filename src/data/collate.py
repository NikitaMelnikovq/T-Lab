from __future__ import annotations

from typing import Dict, List

import torch


def collate_action_batch(samples: List[Dict], tokenizer, max_prompt_length: int, action_mapping) -> Dict:
    prompts = [s["prompt"] for s in samples]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )

    images = torch.stack([s["image"] for s in samples], dim=0)
    action_ids = torch.tensor([int(s["action_id"]) for s in samples], dtype=torch.long)
    target_token_ids = torch.tensor(
        [action_mapping.action_id_to_token_id[int(s["action_id"])] for s in samples], dtype=torch.long
    )

    return {
        "images": images,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "action_ids": action_ids,
        "target_token_ids": target_token_ids,
        "prompts": prompts,
    }


def collate_text_generation_batch(samples: List[Dict], tokenizer, max_prompt_length: int) -> Dict:
    prompts = [s["prompt"] for s in samples]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )

    images = torch.stack([s["image"] for s in samples], dim=0)
    return {
        "images": images,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "prompts": prompts,
    }
