"""Phase-1 backbone families behind one interface.

Every family exposes the attributes the downstream phases (P2-P7) use on LegalLM:

    lm.cfg.d_model, lm.cfg.n_layer, lm.cfg.vocab_size, lm.cfg.max_seq_len
    lm.tok(ids)                     -> token (+position) embeddings, lm.tok.weight = embedding matrix
    lm.blocks                       -> nn.ModuleList; blk(x, cos, sin, attn_mask=None, cache=None)
                                       attn_mask=None means causal, a boolean mask means "attend where True"
    lm.norm, lm.head                -> final norm, LM head (tied to lm.tok.weight)
    lm.rope_cos, lm.rope_sin        -> RoPE tables (dummy (T, 1) buffers for GPT-2, which ignores them)
    blk.attn.qkv / blk.attn.proj / blk.mlp.w12 / blk.mlp.w3
                                    -> the nn.Linear layers that LoRA / LoRMA / IA3 / adapters wrap

so the chunk encoders, PEFT wrappers and pooling heads of every phase run unchanged on any family.

    from backbones import load_backbone, encode
    lm = load_backbone("path/to/model_dir")             # config.json + pytorch_model.bin
    h = encode(lm, ids, key_mask, bidirectional=True)   # (B, T, d_model)
"""
import json
import os

import torch

from .legallm import LegalLM, ModelConfig as LegalLMConfig
from .gpt2_baseline import GPT2LM, GPT2Config
from .swa_lite import SWALiteLM, SWALiteConfig
from .jurisnet import JurisNetLM, JurisNetConfig
from .encdec_hybrid import EncDecLM, EncDecConfig

REGISTRY = {
    "legallm": (LegalLM, LegalLMConfig),      # submitted backbone (shipped config.json has no "arch" key)
    "gpt2": (GPT2LM, GPT2Config),
    "swa_lite": (SWALiteLM, SWALiteConfig),
    "jurisnet": (JurisNetLM, JurisNetConfig),
    "encdec": (EncDecLM, EncDecConfig),
}


def build_backbone(cfg: dict):
    """cfg: model config dict. "arch" selects the family; a config without it is LegalLM."""
    cfg = dict(cfg)
    arch = cfg.pop("arch", "legallm")
    if arch not in REGISTRY:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(REGISTRY)}")
    model_cls, cfg_cls = REGISTRY[arch]
    return model_cls(cfg_cls(**cfg))


def load_backbone(model_dir, map_location="cpu"):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    model = build_backbone(cfg)
    sd = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=map_location)
    model.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd, strict=True)
    return model


def count_parameters(model):
    """Unique parameters (tied embedding / LM head counted once)."""
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())


def encode(lm, ids, key_mask=None, bidirectional=True):
    """Final hidden states of the encoder stack (the encoder for EncDecLM, the whole stack otherwise).

    bidirectional=True drops the causal mask (P2/P3 style); padding keys are masked via key_mask (B, T) bool.
    bidirectional=False keeps causal attention with right padding (P4/P5/P7 style).
    """
    T = ids.shape[1]
    cos, sin = lm.rope_cos[:T][None, None], lm.rope_sin[:T][None, None]
    mask = None
    if bidirectional:
        km = key_mask if key_mask is not None else torch.ones_like(ids, dtype=torch.bool)
        mask = km[:, None, None, :]
    x = lm.tok(ids)
    for blk in lm.blocks:
        x = blk(x, cos, sin, mask)
    return lm.norm(x)


__all__ = ["REGISTRY", "build_backbone", "load_backbone", "count_parameters", "encode",
           "LegalLM", "LegalLMConfig", "GPT2LM", "GPT2Config", "SWALiteLM", "SWALiteConfig",
           "JurisNetLM", "JurisNetConfig", "EncDecLM", "EncDecConfig"]
