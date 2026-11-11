"""Encoder-decoder hybrid (prefix-LM).

A bidirectional encoder reads the prompt, a causal decoder continues it while cross-attending to the
encoder (T5-style split, LegalLM-style blocks: RMSNorm, RoPE, SwiGLU, no biases). Embeddings are shared by
encoder, decoder and the tied LM head.

Pretraining objective (prefix-LM on raw text): every 1024-token window is cut at a prefix length p drawn
from `split_choices`; the encoder sees tokens [0, p), the decoder starts from token p-1 and predicts
tokens p .. T. RoPE positions continue across the cut, so the decoder knows where the prompt ended. Phase 1
is exactly this setting at test time (prompt -> continuation).

Default size (configs/encdec_hybrid.json): 6 encoder + 6 decoder layers, d_model 768, 12 heads, d_ff 2048
  = 24,576,000 (embedding) + 6*7,079,424 (encoder) + 6*9,439,488 (decoder) + 2*768 = 123,691,008 parameters.

Downstream use: `lm.tok / lm.blocks / lm.norm` are the ENCODER, so classification, tagging and retrieval
heads (P2-P5, P7) consume encoder states. Generation (P1, P6) goes through prefill/step (see generate.py).
"""
from dataclasses import dataclass, asdict, field
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .legallm import ModelConfig, RMSNorm, Attention, SwiGLU, Block, rope_tables


@dataclass
class EncDecConfig:
    vocab_size: int = 32000
    n_enc_layer: int = 6
    n_dec_layer: int = 6
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    split_choices: list = field(default_factory=lambda: [256, 384, 512, 640, 768])

    @property
    def n_layer(self):                     # encoder depth; what downstream code means by "layers"
        return self.n_enc_layer

    def to_dict(self):
        return {"arch": "encdec", **asdict(self)}

    def base(self):
        return ModelConfig(vocab_size=self.vocab_size, n_layer=self.n_enc_layer, n_head=self.n_head,
                           d_model=self.d_model, d_ff=self.d_ff, max_seq_len=self.max_seq_len,
                           rope_theta=self.rope_theta, norm_eps=self.norm_eps)


class CrossAttention(nn.Module):
    """Decoder queries over encoder memory (no RoPE: content-based alignment). Memory K/V are cached."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, memory, mem_mask, cache=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if cache is not None and "mk" in cache:
            k, v = cache["mk"], cache["mv"]
        else:
            k, v = (t.view(B, -1, self.n_head, self.head_dim).transpose(1, 2) for t in self.kv(memory).split(C, dim=2))
            if cache is not None:
                cache["mk"], cache["mv"] = k, v
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mem_mask)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)                         # causal self-attention with RoPE + KV cache
        self.norm_x = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.cross = CrossAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, self_mask, memory, mem_mask, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, self_mask, cache)
        x = x + self.cross(self.norm_x(x), memory, mem_mask, cache)
        return x + self.mlp(self.norm2(x))


class EncDecLM(nn.Module):
    def __init__(self, cfg: EncDecConfig):
        super().__init__()
        self.cfg = cfg
        base = cfg.base()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(base) for _ in range(cfg.n_enc_layer)])       # encoder
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)                                    # encoder final norm
        self.dec_blocks = nn.ModuleList([DecoderBlock(base) for _ in range(cfg.n_dec_layer)])
        self.dec_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        cos, sin = rope_tables(base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        depth = cfg.n_enc_layer + cfg.n_dec_layer
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "cross.proj.weight", "mlp.w3.weight")):
                nn.init.normal_(p, 0.0, 0.02 / (2 * depth) ** 0.5)

    def _rope(self, positions, T, device):
        if positions is None:
            return self.rope_cos[:T][None, None], self.rope_sin[:T][None, None]
        positions = positions.clamp(max=self.cfg.max_seq_len - 1)
        if positions.dim() == 1:
            return self.rope_cos[positions][None, None], self.rope_sin[positions][None, None]
        return self.rope_cos[positions].unsqueeze(1), self.rope_sin[positions].unsqueeze(1)

    def encode_ids(self, ids, valid=None, positions=None):
        """Bidirectional encoder. valid: (B, T) bool key mask (left or right padding)."""
        cos, sin = self._rope(positions, ids.shape[1], ids.device)
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        x = self.tok(ids)
        for blk in self.blocks:
            x = blk(x, cos, sin, valid[:, None, None, :])
        return self.norm(x)

    def decode(self, dec_ids, memory, mem_valid=None, positions=None, caches=None):
        T = dec_ids.shape[1]
        cos, sin = self._rope(positions, T, dec_ids.device)
        mem_mask = None if mem_valid is None else mem_valid[:, None, None, :]
        # a cached single-token step attends to every cached key; otherwise causal
        self_mask = torch.ones(1, 1, 1, 1, dtype=torch.bool, device=dec_ids.device) if (caches is not None and T == 1) else None
        x = self.tok(dec_ids)
        for i, blk in enumerate(self.dec_blocks):
            x = blk(x, cos, sin, self_mask, memory, mem_mask, None if caches is None else caches[i])
        return self.dec_norm(x)

    def forward(self, idx, targets=None, split=None, **_):
        if targets is None:
            raise NotImplementedError("EncDecLM: use encode_ids()/decode() (or generate.py) for inference")
        T = idx.shape[1]
        choices = [p for p in self.cfg.split_choices if 1 < p < T] or [T // 2]
        p = split or random.choice(choices)
        memory = self.encode_ids(idx[:, :p])
        h = self.decode(idx[:, p - 1:], memory, positions=torch.arange(p - 1, T, device=idx.device))
        logits = self.head(h).float()
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[:, p - 1:].reshape(-1))
