"""Sliding-window lightweight decoder (SWA-Lite).

LegalLM's block (RMSNorm, RoPE, SwiGLU, no biases, tied embeddings) at roughly half the width, with every
token attending only to the previous `window` tokens (Longformer / Mistral-style local attention).
Motivation: legal text is dominated by local structure (citations, section numbers, party names), so a
narrower model with local attention should keep most of the modelling quality at ~half the parameters.

Default size (configs/swa_lite.json): d_model 512, 12 layers, 8 heads, d_ff 1536, window 256, context 1024
  = 32,000*512 + 12*3,408,896 + 512 = 57,291,264 parameters.

The window is a banded boolean mask passed to scaled_dot_product_attention; band distance is measured in
sequence indices, which equals position distance for right-padded training batches and for the
left-padded, KV-cached batches used in generation. Bidirectional use (downstream encoders) gets a
symmetric band |i - j| < window.
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .legallm import LegalLM, ModelConfig, Attention, Block, apply_rope


@dataclass
class SWALiteConfig:
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 8
    d_model: int = 512
    d_ff: int = 1536
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    window: int = 256

    def to_dict(self):
        return {"arch": "swa_lite", **asdict(self)}

    def base(self):
        d = asdict(self)
        d.pop("window")
        return ModelConfig(**d)


class SlidingWindowAttention(Attention):
    def __init__(self, cfg, window):
        super().__init__(cfg)
        self.window = window

    def forward(self, x, cos, sin, attn_mask=None, cache=None):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in self.qkv(x).split(C, dim=2))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        Tk = k.shape[2]
        qi = torch.arange(Tk - T, Tk, device=x.device)[:, None]
        kj = torch.arange(Tk, device=x.device)[None, :]
        band = (qi - kj).abs() < self.window
        mask = (band & (kj <= qi)) if attn_mask is None else (attn_mask & band)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SWALiteLM(LegalLM):
    def __init__(self, cfg: SWALiteConfig):
        super().__init__(cfg.base())
        self.cfg = cfg
        for blk in self.blocks:                                    # swap in the local-attention module
            old = blk.attn
            blk.attn = SlidingWindowAttention(cfg, cfg.window)
            blk.attn.load_state_dict(old.state_dict())             # keeps the scaled residual init
