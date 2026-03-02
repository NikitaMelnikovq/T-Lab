from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

ACTION_ID_TO_TOKEN: Dict[int, str] = {
    0: "<ACT_LEFT>",
    1: "<ACT_RIGHT>",
    2: "<ACT_FORWARD>",
    3: "<ACT_PICKUP>",
    4: "<ACT_DROP>",
    5: "<ACT_TOGGLE>",
    6: "<ACT_DONE>",
}

ACTION_TOKEN_TO_ID = {v: k for k, v in ACTION_ID_TO_TOKEN.items()}
ACTION_TOKENS: List[str] = [ACTION_ID_TO_TOKEN[i] for i in range(7)]


@dataclass
class ActionTokenMapping:
    action_token_ids: List[int]
    action_id_to_token_id: Dict[int, int]
    token_id_to_action_id: Dict[int, int]


def build_tokenizer(tokenizer_name: str, add_action_tokens: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if add_action_tokens:
        existing = set(tokenizer.get_vocab().keys())
        to_add = [t for t in ACTION_TOKENS if t not in existing]
        if to_add:
            tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    return tokenizer


def build_action_mapping(tokenizer) -> ActionTokenMapping:
    action_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in ACTION_TOKENS]
    action_id_to_token_id = {a: tokenizer.convert_tokens_to_ids(tok) for a, tok in ACTION_ID_TO_TOKEN.items()}
    token_id_to_action_id = {v: k for k, v in action_id_to_token_id.items()}
    return ActionTokenMapping(
        action_token_ids=action_token_ids,
        action_id_to_token_id=action_id_to_token_id,
        token_id_to_action_id=token_id_to_action_id,
    )


def action_id_to_token(action_id: int) -> str:
    return ACTION_ID_TO_TOKEN[int(action_id)]


def token_to_action_id(token: str) -> int:
    return ACTION_TOKEN_TO_ID[token]
