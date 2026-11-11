"""JurisNetLM: citation-aware decoder.

Same transformer as LegalLM (so any difference comes from citation awareness, not size), plus:
  * type embedding   - every input token gets one of {text, case citation, statute reference}
                       (backbones/citations.py, tagged on the raw text) added to its token embedding;
  * next-type head   - an auxiliary classifier predicts the type of the NEXT token, i.e. learns where a
                       citation is about to start or end;
  * citation-weighted LM loss - next-token CE on citation/statute targets is up-weighted (cite_weight).

loss = weighted_CE(next token) + aux_weight * CE(next token type)

Default size (configs/jurisnet.json): LegalLM dims (12 x 768, d_ff 2048, 32K vocab)
  = 109,529,856 + 3*768 (type embedding) + 768*3 + 3 (next-type head) = 109,534,467 parameters.

Without `type_ids` (e.g. inside the downstream notebooks, which call lm.tok / lm.blocks directly) it runs as
a plain LegalLM-shaped decoder, so the downstream heads and PEFT wrappers apply unchanged.
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .legallm import LegalLM, ModelConfig
from .citations import N_TYPES


@dataclass
class JurisNetConfig:
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    n_types: int = N_TYPES
    cite_weight: float = 1.5
    aux_weight: float = 0.1

    def to_dict(self):
        return {"arch": "jurisnet", **asdict(self)}

    def base(self):
        d = asdict(self)
        for k in ("n_types", "cite_weight", "aux_weight"):
            d.pop(k)
        return ModelConfig(**d)


class JurisNetLM(LegalLM):
    def __init__(self, cfg: JurisNetConfig):
        super().__init__(cfg.base())
        self.cfg = cfg
        self.type_emb = nn.Embedding(cfg.n_types, cfg.d_model)
        nn.init.zeros_(self.type_emb.weight)                     # starts as plain LegalLM
        self.type_head = nn.Linear(cfg.d_model, cfg.n_types)
        nn.init.normal_(self.type_head.weight, 0.0, 0.02)
        nn.init.zeros_(self.type_head.bias)

    def forward(self, idx, targets=None, positions=None, attn_mask=None, caches=None, last_only=False,
                type_ids=None, type_targets=None, return_type_logits=False):
        T = idx.shape[1]
        if positions is None:
            cos, sin = self.rope_cos[:T][None, None], self.rope_sin[:T][None, None]
        else:
            cos, sin = self.rope_cos[positions].unsqueeze(1), self.rope_sin[positions].unsqueeze(1)
        x = self.tok(idx)
        if type_ids is not None:
            x = x + self.type_emb(type_ids.long())
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, attn_mask, None if caches is None else caches[i])
        x = self.norm(x)
        if targets is None:
            x = x[:, -1:] if last_only else x
            return (self.head(x), self.type_head(x)) if return_type_logits else self.head(x)
        logits = self.head(x).float()
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if type_targets is None:
            return ce.mean()
        tt = type_targets.reshape(-1).long()
        w = torch.where(tt > 0, torch.full_like(ce, self.cfg.cite_weight), torch.ones_like(ce))
        lm_loss = (ce * w).sum() / w.sum()
        aux = F.cross_entropy(self.type_head(x).float().view(-1, self.cfg.n_types), tt)
        return lm_loss + self.cfg.aux_weight * aux
