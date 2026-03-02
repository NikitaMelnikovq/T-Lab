from __future__ import annotations

import random
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROMPT_VARIANTS: List[str] = [
    "You control an agent in MiniGrid. Mission: {mission}\nRespond with exactly one token for the next action.\nAction:",
    "Task: {mission}\nChoose the immediate next move token for the agent.\nAction:",
    "Environment mission is: {mission}\nPredict one next action token from <ACT_...>.\nAction:",
]

TEXT_ACTION_PROMPT = (
    "You control an agent in MiniGrid. Mission: {mission}\n"
    "Provide a short plan (1-2 sentences) and one action token.\n"
    "Use exact format:\n"
    "Plan: <text>\n"
    "Action: <ACT_LEFT|ACT_RIGHT|ACT_FORWARD|ACT_PICKUP|ACT_DROP|ACT_TOGGLE|ACT_DONE>\n"
    "Now respond."
)


class ImageTransform:
    def __init__(self, image_size: int = 224):
        self.t = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.t(image)


def format_action_prompt(mission: str, variant_id: int = 0) -> str:
    idx = int(variant_id) % len(PROMPT_VARIANTS)
    return PROMPT_VARIANTS[idx].format(mission=mission)


def format_text_action_prompt(mission: str) -> str:
    return TEXT_ACTION_PROMPT.format(mission=mission)


def choose_prompt_variant(use_prompt_variants: bool, prompt_variant_count: int, rng: random.Random) -> int:
    if not use_prompt_variants:
        return 0
    return rng.randrange(max(1, prompt_variant_count))
