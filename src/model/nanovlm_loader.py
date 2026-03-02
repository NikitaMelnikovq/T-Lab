from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional

import torch

from src.model.action_tokenizer import build_tokenizer
from src.model.vendor.nanovlm_v0_1.models.config import VLMConfig
from src.model.vendor.nanovlm_v0_1.models.vision_language_model import VisionLanguageModel


def load_nanovlm(
    source: str,
    tokenizer_name: str,
    device: str,
    freeze_backbones: bool = True,
    unfreeze_embeddings: bool = True,
    unfreeze_lm_head: bool = True,
):
    tokenizer = build_tokenizer(tokenizer_name, add_action_tokens=True)

    source_path = Path(source)
    if source_path.exists() and (source_path / "config.json").exists():
        model = VisionLanguageModel.from_pretrained(str(source_path))
    else:
        cfg = VLMConfig()
        cfg.lm_vocab_size = len(tokenizer)
        cfg.lm_tokenizer = tokenizer_name
        model = VisionLanguageModel(cfg, load_backbone=True)

    if model.decoder.token_embedding.num_embeddings != len(tokenizer):
        # Existing checkpoint may have a smaller vocab; resize manually.
        _resize_token_embeddings(model, len(tokenizer))

    if freeze_backbones:
        apply_peft_lite(
            model,
            unfreeze_embeddings=unfreeze_embeddings,
            unfreeze_lm_head=unfreeze_lm_head,
        )

    model.to(device)
    return model, tokenizer


def _resize_token_embeddings(model, new_vocab_size: int) -> None:
    old_emb = model.decoder.token_embedding
    old_head = model.decoder.head
    old_vocab, hidden = old_emb.weight.shape
    if old_vocab == new_vocab_size:
        return

    device = old_emb.weight.device
    dtype = old_emb.weight.dtype

    new_emb = torch.nn.Embedding(new_vocab_size, hidden, device=device, dtype=dtype)
    torch.nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
    copy_n = min(old_vocab, new_vocab_size)
    with torch.no_grad():
        new_emb.weight[:copy_n].copy_(old_emb.weight[:copy_n])

    new_head = torch.nn.Linear(hidden, new_vocab_size, bias=False, device=device, dtype=dtype)
    torch.nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
    with torch.no_grad():
        new_head.weight[:copy_n].copy_(old_head.weight[:copy_n])

    model.decoder.token_embedding = new_emb
    model.decoder.head = new_head
    if getattr(model.cfg, "lm_tie_weights", False):
        model.decoder.head.weight = model.decoder.token_embedding.weight
    model.cfg.lm_vocab_size = new_vocab_size


def apply_peft_lite(model, unfreeze_embeddings: bool = True, unfreeze_lm_head: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = False

    for p in model.MP.parameters():
        p.requires_grad = True

    if unfreeze_embeddings:
        for p in model.decoder.token_embedding.parameters():
            p.requires_grad = True

    if unfreeze_lm_head:
        for p in model.decoder.head.parameters():
            p.requires_grad = True


def build_reference_model(policy_model, device: str):
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.to(device)
    return ref_model


def get_trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
