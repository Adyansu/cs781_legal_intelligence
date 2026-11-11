"""GPT-2-style decoder: the first Phase-1 baseline (59.46M parameters).

Classic GPT-2 block (Radford et al., 2019): learned absolute positions, pre-LayerNorm, GELU MLP, biases,
tied input/output embeddings, standard 50,257-token GPT-2 BPE. Trained on Supreme Court text only
(200M train / 5M validation tokens).

Default size (configs/gpt2_baseline.json): d_model 576, 9 layers, 9 heads, d_ff 1728, context 1024
  = 50,257*576 (wte) + 1,024*576 (wpe) + 9*3,324,672 (blocks) + 1,152 (ln_f) = 59,461,056 parameters.

Module names (tok / blocks / attn.qkv / attn.proj / mlp.w12 / mlp.w3 / norm / head) mirror LegalLM so the
downstream PEFT wrappers apply unchanged; mlp.w12 is the plain d->d_ff projection here (no gating).
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_layer: int = 9
    n_head: int = 9
    d_model: int = 576
    d_ff: int = 1728
    max_seq_len: int = 1024
    dropout: float = 0.0
    norm_eps: float = 1e-5

    def to_dict(self):
        return {"arch": "gpt2", **asdict(self)}


class TokPosEmbedding(nn.Module):
    """wte(ids) + wpe(positions); `.weight` is the token matrix (used for tying and prompt-tuning init)."""

    def __init__(self, cfg):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.max_seq_len, cfg.d_model)

    @property
    def weight(self):
        return self.wte.weight

    def forward(self, idx, positions=None):
        if positions is None:
            positions = torch.arange(idx.shape[1], device=idx.device)[None]
        return self.wte(idx) + self.wpe(positions.clamp(max=self.wpe.num_embeddings - 1))


class GPT2Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = cfg.dropout

    def forward(self, x, cos=None, sin=None, attn_mask=None, cache=None):     # cos/sin unused (learned positions)
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in self.qkv(x).split(C, dim=2))
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        p = self.drop if self.training else 0.0
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class GPT2MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w12 = nn.Linear(cfg.d_model, cfg.d_ff)          # "c_fc"
        self.w3 = nn.Linear(cfg.d_ff, cfg.d_model)           # "c_proj"

    def forward(self, x):
        return self.w3(F.gelu(self.w12(x), approximate="tanh"))


class GPT2Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = GPT2Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = GPT2MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos=None, sin=None, attn_mask=None, cache=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, attn_mask, cache))
        return x + self.drop(self.mlp(self.norm2(x)))


class GPT2LM(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.tok = TokPosEmbedding(cfg)
        self.blocks = nn.ModuleList([GPT2Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.wte.weight
        # dummy RoPE tables so code written for LegalLM (rope_cos[:T], rope_cos[positions]) still runs
        self.register_buffer("rope_cos", torch.zeros(cfg.max_seq_len, 1), persistent=False)
        self.register_buffer("rope_sin", torch.zeros(cfg.max_seq_len, 1), persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():               # GPT-2 scaled init of residual projections
            if name.endswith("attn.proj.weight") or name.endswith("mlp.w3.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, positions=None, attn_mask=None, caches=None, last_only=False):
        x = self.tok(idx, positions)
        for i, blk in enumerate(self.blocks):
            x = blk(x, None, None, attn_mask, None if caches is None else caches[i])
        x = self.norm(x)
        if targets is not None:
            logits = self.head(x)
            return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return self.head(x[:, -1:] if last_only else x)
