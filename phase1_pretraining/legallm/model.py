"""LegalLM: a decoder-only transformer written from scratch for CS781 Phase 1.

Design (modern "Llama-style" block, all implemented here):
  * RMSNorm (pre-norm) instead of LayerNorm
  * Rotary position embeddings (RoPE) instead of learned absolute positions
  * SwiGLU feed-forward instead of GELU MLP
  * No bias terms, tied input/output embeddings
  * PyTorch scaled_dot_product_attention (memory-efficient kernel on T4)
  * KV cache + left-padding support for fast batched generation
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight).to(x.dtype)


def rope_tables(cfg):
    head_dim = cfg.d_model // cfg.n_head
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(cfg.max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, head_dim); cos/sin: (B or 1, 1, T, head_dim/2)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, attn_mask=None, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w12 = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask=None, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, attn_mask, cache)
        return x + self.mlp(self.norm2(x))


class LegalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight                      # tied embeddings
        cos, sin = rope_tables(cfg)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():                 # scaled init of residual projections
            if name.endswith("attn.proj.weight") or name.endswith("mlp.w3.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, positions=None, attn_mask=None, caches=None, last_only=False):
        B, T = idx.shape
        if positions is None:
            cos = self.rope_cos[:T][None, None]
            sin = self.rope_sin[:T][None, None]
        else:
            cos = self.rope_cos[positions].unsqueeze(1)
            sin = self.rope_sin[positions].unsqueeze(1)
        x = self.tok(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, attn_mask, None if caches is None else caches[i])
        x = self.norm(x)
        if targets is not None:
            logits = self.head(x)
            return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        if last_only:
            x = x[:, -1:]
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
