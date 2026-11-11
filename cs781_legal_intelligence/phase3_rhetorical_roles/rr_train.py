"""CS781 Phase 3 - Rhetorical Role prediction on top of the Phase-1 LegalLM.

One script, one experiment per call, one output directory per experiment:
    python rr_train.py --name e1_full --method full
    python rr_train.py --name e1_lorma --method lorma --hp '{"lr": 1e-3}'
    python rr_train.py --name smoke_full --method full --smoke        (2 docs/domain, a few steps)

Model ("packed-context hierarchical encoder"):
  consecutive sentences of a document are packed into one window (<= `window` tokens, an EOS between
  sentences) -> Phase-1 LegalLM with the causal mask removed (bidirectional=True), so every token sees
  the neighbouring sentences -> mean-pool each sentence's tokens -> + document-position bucket
  embedding + domain (CL/IT) embedding -> BiGRU over the sentences of the window -> 13-way softmax.
  Inference: overlapping windows (50% sentence overlap), centre-weighted average of probabilities.

Selection: official dev Macro-F1 from YOUR evaluate_phase3.py (run as a subprocess every epoch);
the in-script sklearn mirror must agree with it to 1e-6 or the run stops.
Test labels are never read. Submission: <out>/submission_<name>.zip with prediction.json at its root.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_data import LABELS, DOMAINS, load_phase3, describe  # noqa: E402

L2I = {l: i for i, l in enumerate(LABELS)}
N_LABELS = len(LABELS)
DOM2I = {d: i for i, d in enumerate(DOMAINS)}

DEFAULTS = dict(
    bidirectional=True,       # drop the causal mask (Phase 2: bidirectional beat causal on this model)
    pooling="mean",           # "mean" | "last" (last is the natural choice when bidirectional=False)
    window=1024,              # tokens per packed window (Phase-1 max_seq_len = 1024)
    max_sent_tokens=256,      # a single sentence is truncated to this many tokens
    context="gru",            # "gru" | "none"   (BiGRU over the sentences of a window)
    gru_hidden=256,
    pos_buckets=16,           # document-position buckets (0 = off)
    domain_emb=True,
    epochs=12, patience=4, batch_size=4, eval_batch_size=8,
    lr=3e-5,                  # encoder (or adapter) learning rate
    head_lr=1e-3,             # pooling extras, GRU, classifier
    weight_decay=0.01, warmup=0.06, grad_clip=1.0, dropout=0.1,
    loss="ce",                # "ce" | "sqrt_weighted" (sqrt inverse-frequency, mean 1, clipped)
    label_smoothing=0.0,
    fp16=True, seed=42,
    max_steps_per_epoch=0,    # 0 = full epoch (smoke tests set this)
)
METHOD_DEFAULTS = {
    "full":   dict(lr=3e-5),
    "lora":   dict(lr=5e-4, rank=16, alpha=32, targets=["qkv", "proj", "w12", "w3"]),
    "rslora": dict(lr=5e-4, rank=16, alpha=32, targets=["qkv", "proj", "w12", "w3"]),
    # LoRMA diverged at lr 1e-3 with 1024-token windows (non-finite losses in epoch 2) -> 2e-4, fp32 update path
    "lorma":  dict(lr=2e-4, rank=16, alpha=32, targets=["qkv", "proj", "w12", "w3"]),
    "ia3":    dict(lr=3e-3),
    "frozen": dict(lr=0.0),   # linear-probe control: encoder frozen, only GRU/head train
    # knowledge distillation: smaller student (subset of Phase-1 layers, fully fine-tuned) learns from the
    # gold labels and from a finished teacher run's sentence-level probabilities (same windows, same batch)
    "kd":     dict(lr=5e-5, teacher="", student_layers=None, kd_alpha=0.5, kd_T=2.0),
}


def log(*a):
    print(*a, flush=True)


# ============================================================================ encoder (= Phase-1 weights)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


def rope_tables(cfg):
    hd = cfg["d_model"] // cfg["n_head"]
    inv = 1.0 / (cfg.get("rope_theta", 10000.0) ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(cfg["max_seq_len"]).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg["n_head"], cfg["d_model"] // cfg["n_head"]
        self.qkv = nn.Linear(cfg["d_model"], 3 * cfg["d_model"], bias=False)
        self.proj = nn.Linear(cfg["d_model"], cfg["d_model"], bias=False)

    def forward(self, x, cos, sin, mask, causal):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, self.hd).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if causal:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # right padding never reaches real tokens
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)   # (B,1,1,T) True = attend
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w12 = nn.Linear(cfg["d_model"], 2 * cfg["d_ff"], bias=False)
        self.w3 = nn.Linear(cfg["d_ff"], cfg["d_model"], bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg["d_model"], cfg.get("norm_eps", 1e-5))
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg["d_model"], cfg.get("norm_eps", 1e-5))
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask, causal):
        x = x + self.attn(self.norm1(x), cos, sin, mask, causal)
        return x + self.mlp(self.norm2(x))


class LegalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.norm = RMSNorm(cfg["d_model"], cfg.get("norm_eps", 1e-5))
        cos, sin = rope_tables(cfg)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, ids, key_mask, causal):
        T = ids.shape[1]
        cos, sin = self.rope_cos[:T][None, None], self.rope_sin[:T][None, None]
        mask = None if causal else key_mask[:, None, None, :]
        x = self.tok(ids)
        for b in self.blocks:
            x = b(x, cos, sin, mask, causal)
        return self.norm(x)


# ============================================================================ PEFT modules
class LoRALinear(nn.Module):
    """W0 x + s * B A x ; s = alpha/r (LoRA) or alpha/sqrt(r) (rsLoRA). B = 0 at init."""
    def __init__(self, base, r, alpha, rs=False):
        super().__init__()
        self.base, self.s = base, (alpha / math.sqrt(r) if rs else alpha / r)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + self.s * F.linear(F.linear(x, self.A), self.B)


class LoRMALinear(nn.Module):
    """LoRMA+: (I + s B A) W0 x, computed as y0 + s B (A y0). B = 0 at init (same as your Phase 2 code)."""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.s = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        y0 = self.base(x)
        with torch.autocast("cuda", enabled=False):                       # low-rank update in fp32
            d = F.linear(F.linear(y0.float(), self.A.float()), self.B.float()) * self.s
        return y0 + d.to(y0.dtype)


class IA3QKV(nn.Module):
    """(IA)^3 on attention: rescale keys and values (queries untouched). Vectors start at 1."""
    def __init__(self, base, d):
        super().__init__()
        self.base = base
        self.l_k = nn.Parameter(torch.ones(d))
        self.l_v = nn.Parameter(torch.ones(d))
        self.register_buffer("ones_q", torch.ones(d), persistent=False)

    def forward(self, x):
        return self.base(x) * torch.cat([self.ones_q, self.l_k, self.l_v]).to(x.dtype)


class IA3In(nn.Module):
    """(IA)^3 on the FFN: rescale the d_ff activations entering w3."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l_ff = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l_ff.to(x.dtype))


def attach_peft(encoder, method, hp, d_model):
    n = 0
    for blk in encoder.blocks:
        if method == "ia3":
            blk.attn.qkv = IA3QKV(blk.attn.qkv, d_model)
            blk.mlp.w3 = IA3In(blk.mlp.w3)
            n += 2
            continue
        for parent, name in [(blk.attn, "qkv"), (blk.attn, "proj"), (blk.mlp, "w12"), (blk.mlp, "w3")]:
            if name not in hp["targets"]:
                continue
            base = getattr(parent, name)
            if method == "lora":
                mod = LoRALinear(base, hp["rank"], hp["alpha"])
            elif method == "rslora":
                mod = LoRALinear(base, hp["rank"], hp["alpha"], rs=True)
            elif method == "lorma":
                mod = LoRMALinear(base, hp["rank"], hp["alpha"])
            else:
                raise ValueError(method)
            setattr(parent, name, mod)
            n += 1
    return n


# ============================================================================ RR model
class RRModel(nn.Module):
    def __init__(self, cfg, hp):
        super().__init__()
        d = cfg["d_model"]
        self.hp = hp
        self.encoder = LegalEncoder(cfg)
        self.causal = not hp["bidirectional"]
        self.drop = nn.Dropout(hp["dropout"])
        self.pos_emb = nn.Embedding(hp["pos_buckets"], d) if hp["pos_buckets"] else None
        self.dom_emb = nn.Embedding(len(DOMAINS), d) if hp["domain_emb"] else None
        for e in (self.pos_emb, self.dom_emb):
            if e is not None:
                nn.init.normal_(e.weight, std=0.02)
        if hp["context"] == "gru":
            self.ctx = nn.GRU(d, hp["gru_hidden"], batch_first=True, bidirectional=True)
            out = d + 2 * hp["gru_hidden"]
        elif hp["context"] == "none":
            self.ctx, out = None, d
        else:
            raise ValueError(hp["context"])
        self.head = nn.Linear(out, N_LABELS)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def new_module_names(self):
        return ("pos_emb", "dom_emb", "ctx", "head")

    def forward(self, ids, key_mask, slot, n_sent, pos, dom):
        """ids/key_mask/slot: (B,T); slot = sentence index of each token, -1 for EOS/pad.
        n_sent: (B,) sentences per window; pos: (B,S) position bucket; dom: (B,)  ->  logits (B,S,13)"""
        h = self.encoder(ids, key_mask, self.causal).float()
        B, T, D = h.shape
        S = int(n_sent.max())
        valid = (slot >= 0)
        idx = slot.clamp(min=0)
        if self.hp["pooling"] == "mean":
            w = valid.float()
            sums = torch.zeros(B, S, D, device=h.device).scatter_add_(1, idx[..., None].expand(-1, -1, D), h * w[..., None])
            cnt = torch.zeros(B, S, device=h.device).scatter_add_(1, idx, w)
            v = sums / cnt.clamp(min=1.0)[..., None]
        elif self.hp["pooling"] == "last":
            tpos = torch.arange(T, device=h.device)[None].expand(B, T)
            lastpos = torch.full((B, S), -1, dtype=torch.long, device=h.device)
            lastpos.scatter_reduce_(1, idx, torch.where(valid, tpos, torch.full_like(tpos, -1)), reduce="amax")
            v = h.gather(1, lastpos.clamp(min=0)[..., None].expand(-1, -1, D))
        else:
            raise ValueError(self.hp["pooling"])
        if self.pos_emb is not None:
            v = v + self.pos_emb(pos)
        if self.dom_emb is not None:
            v = v + self.dom_emb(dom)[:, None]
        v = self.drop(v)
        if self.ctx is not None:
            packed = nn.utils.rnn.pack_padded_sequence(v, n_sent.cpu(), batch_first=True, enforce_sorted=False)
            c, _ = self.ctx(packed)
            c, _ = nn.utils.rnn.pad_packed_sequence(c, batch_first=True, total_length=S)
            v = torch.cat([v, self.drop(c)], -1)
        return self.head(v)


def load_phase1(model, model_dir, layer_map=None):
    sd = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]                                          # a training checkpoint instead of final weights
    sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    new = {}
    for k, v in sd.items():
        if k.startswith("head."):
            continue                                              # LM head is tied to tok
        if k.startswith("blocks.") and layer_map is not None:     # KD student: keep a subset of layers
            i = int(k.split(".")[1])
            if i not in layer_map:
                continue
            k = f"blocks.{layer_map[i]}." + k.split(".", 2)[2]
        new["encoder." + k] = v
    missing, unexpected = model.load_state_dict(new, strict=False)
    new_mods = model.new_module_names()
    missing = [m for m in missing if not m.split(".")[0] in new_mods]
    if missing or unexpected:
        raise RuntimeError(f"Phase-1 weights do not match: missing={missing[:8]} unexpected={unexpected[:8]}")
    tied = torch.equal(sd["head.weight"], sd["tok.weight"]) if "head.weight" in sd else None
    return len(new), tied


def build_model(cfg, hp, method, model_dir):
    cfg = dict(cfg)
    layer_map = None
    if hp.get("student_layers") is not None or method == "kd":
        layers = list(hp.get("student_layers") or range(0, cfg["n_layer"], 2))   # default: every 2nd layer
        assert all(0 <= i < cfg["n_layer"] for i in layers) and len(set(layers)) == len(layers), layers
        hp["student_layers"] = layers
        layer_map = {src: dst for dst, src in enumerate(layers)}
        cfg["n_layer"] = len(layers)
    model = RRModel(cfg, hp)
    n_loaded, tied = load_phase1(model, model_dir, layer_map)
    if method not in ("full", "kd"):
        for p in model.encoder.parameters():
            p.requires_grad = False
    n_wrapped = attach_peft(model.encoder, method, hp, cfg["d_model"]) if method not in ("full", "frozen", "kd") else 0
    return model, n_loaded, tied, n_wrapped


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def load_trainable_state(model, state):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != names:
        raise RuntimeError(f"checkpoint/trainable mismatch: {sorted(set(state) ^ names)[:8]}")
    model.load_state_dict(state, strict=False)


# ============================================================================ data -> windows
class EncDoc:
    pass


def encode_docs(docs, tok, hp, eos_id):
    out = []
    for d in docs:
        e = EncDoc()
        e.id, e.domain, e.split = d["id"], d["domain"], d["split"]
        e.sent_ids = []
        for s in d["sentences"]:
            ids = tok.encode(s).ids[:hp["max_sent_tokens"]] if s.strip() else []
            e.sent_ids.append(ids if ids else [eos_id])          # empty sentence -> one placeholder token
        e.n = len(e.sent_ids)
        e.labels = np.array([L2I[l] for l in d["labels"]], dtype=np.int64) if d["labels"] is not None else None
        nb = hp["pos_buckets"] or 1
        e.pos = np.minimum((np.arange(e.n) * nb) // max(e.n, 1), nb - 1).astype(np.int64)
        out.append(e)
    return out


def pack_from(doc, start, budget):
    """Largest run of sentences starting at `start` that fits `budget` tokens (EOS after each sentence)."""
    used, end = 0, start
    while end < doc.n:
        need = len(doc.sent_ids[end]) + 1
        if end > start and used + need > budget:
            break
        used += need
        end += 1
    return end


def train_windows(docs, budget, rng):
    """Non-overlapping windows; random start offset per document and epoch -> varied context each epoch."""
    items = []
    for d in docs:
        first_end = pack_from(d, 0, budget)
        s = rng.randrange(0, max(1, first_end))
        if s > 0:
            items.append((d, 0, s))
        while s < d.n:
            e = pack_from(d, s, budget)
            items.append((d, s, e))
            s = e
    return items


def infer_windows(docs, budget):
    """Windows advance by half their sentence count -> every sentence is inside >= 1 window, mostly 2."""
    items = []
    for d in docs:
        s = 0
        while True:
            e = pack_from(d, s, budget)
            items.append((d, s, e))
            if e >= d.n:
                break
            s = max(s + 1, s + (e - s) // 2)
    return items


def collate(items, eos_id, with_labels=True):
    seqs, slots, ns = [], [], []
    for d, s, e in items:
        ids, sl = [], []
        for j, k in enumerate(range(s, e)):
            ids += d.sent_ids[k] + [eos_id]
            sl += [j] * len(d.sent_ids[k]) + [-1]
        seqs.append(ids); slots.append(sl); ns.append(e - s)
    B, T, S = len(items), max(map(len, seqs)), max(ns)
    ids = torch.full((B, T), eos_id, dtype=torch.long)
    km = torch.zeros((B, T), dtype=torch.bool)
    slot = torch.full((B, T), -1, dtype=torch.long)
    pos = torch.zeros((B, S), dtype=torch.long)
    lab = torch.full((B, S), -100, dtype=torch.long)
    dom = torch.tensor([DOM2I[d.domain] for d, _, _ in items], dtype=torch.long)
    for i, ((d, s, e), q, sl) in enumerate(zip(items, seqs, slots)):
        ids[i, :len(q)] = torch.tensor(q)
        km[i, :len(q)] = True
        slot[i, :len(sl)] = torch.tensor(sl)
        pos[i, :e - s] = torch.from_numpy(d.pos[s:e])
        if with_labels and d.labels is not None:
            lab[i, :e - s] = torch.from_numpy(d.labels[s:e])
    return ids, km, slot, torch.tensor(ns), pos, dom, lab


# ============================================================================ metrics + official evaluator
def sk_metrics(y_true, y_pred):
    from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, confusion_matrix
    m = dict(macro_f1=f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
             macro_p=precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
             macro_r=recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    m["per_class"] = {l: dict(p=float(p[i]), r=float(r[i]), f1=float(f[i]), support=int(s[i])) for i, l in enumerate(LABELS)}
    m["confusion"] = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    return m


def flatten(truth, pred, ids):
    yt, yp = [], []
    for d in ids:
        yt += truth[d]; yp += pred[d]
    return yt, yp


def full_report(dev_docs, preds):
    truth = {d.id: [LABELS[i] for i in d.labels] for d in dev_docs}
    rep = {"combined": sk_metrics(*flatten(truth, preds, list(truth)))}
    for dom in DOMAINS:
        ids = [d.id for d in dev_docs if d.domain == dom]
        if ids:
            rep[dom] = sk_metrics(*flatten(truth, preds, ids))
    return rep


class OfficialEvaluator:
    def __init__(self, evaluator_path, work_dir, dev_docs):
        self.path, self.dir = evaluator_path, work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.labels_path = os.path.join(work_dir, "labels.json")
        json.dump(LABELS, open(self.labels_path, "w"))
        self.truth = {}
        for dom in DOMAINS:
            p = os.path.join(work_dir, f"dev_truth_{dom.lower()}.json")
            json.dump({d.id: [LABELS[i] for i in d.labels] for d in dev_docs if d.domain == dom},
                      open(p, "w"), ensure_ascii=False)
            self.truth[dom] = p

    def __call__(self, preds, tag):
        pp = os.path.join(self.dir, f"dev_prediction_{tag}.json")
        json.dump(preds, open(pp, "w"), ensure_ascii=False)
        cmd = [sys.executable, self.path, "--truth-cl", self.truth["CL"], "--truth-it", self.truth["IT"],
               "--prediction", pp, "--labels", self.labels_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"official evaluator failed:\n{r.stdout}\n{r.stderr}")
        vals = {k: float(v) for k, v in re.findall(r"Macro-(F1|Precision|Recall):\s*([0-9.]+)", r.stdout)}
        if "F1" not in vals:
            raise RuntimeError(f"could not parse evaluator output:\n{r.stdout}")
        return vals, r.stdout


# ============================================================================ prediction
@torch.no_grad()
def predict(model, docs, hp, device, eos_id):
    model.eval()
    use_amp = hp["fp16"] and device.type == "cuda"
    items = infer_windows(docs, hp["window"])
    acc = {d.id: np.zeros((d.n, N_LABELS), dtype=np.float64) for d in docs}
    wsum = {d.id: np.zeros(d.n, dtype=np.float64) for d in docs}
    for k in range(0, len(items), hp["eval_batch_size"]):
        batch = items[k:k + hp["eval_batch_size"]]
        ids, km, slot, ns, pos, dom, _ = collate(batch, eos_id, with_labels=False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(ids.to(device), km.to(device), slot.to(device), ns.to(device), pos.to(device), dom.to(device))
        probs = torch.softmax(logits.float(), -1).cpu().numpy()
        for i, (d, s, e) in enumerate(batch):
            j = np.arange(e - s)
            left = j if s > 0 else np.full(e - s, 10 ** 6)
            right = (e - s - 1) - j if e < d.n else np.full(e - s, 10 ** 6)
            w = np.minimum(np.minimum(left, right), 10 ** 3) + 1.0
            acc[d.id][s:e] += probs[i, :e - s] * w[:, None]
            wsum[d.id][s:e] += w
    preds, probs_out = {}, {}
    for d in docs:
        assert (wsum[d.id] > 0).all(), f"{d.id}: a sentence was not covered by any window"
        p = acc[d.id] / wsum[d.id][:, None]
        probs_out[d.id] = p
        preds[d.id] = [LABELS[i] for i in p.argmax(1)]
        assert len(preds[d.id]) == d.n
    return preds, probs_out


def dev_nll(dev_docs, probs):
    tot, n = 0.0, 0
    for d in dev_docs:
        p = probs[d.id][np.arange(d.n), d.labels]
        tot += -np.log(np.clip(p, 1e-12, 1)).sum(); n += d.n
    return tot / max(n, 1)


def class_weights(train_docs, hp):
    if hp["loss"] == "ce":
        return None
    if hp["loss"] != "sqrt_weighted":
        raise ValueError(hp["loss"])
    c = np.bincount(np.concatenate([d.labels for d in train_docs]), minlength=N_LABELS).astype(float)
    w = np.where(c > 0, np.sqrt(c.sum() / np.maximum(c, 1)), 1.0)
    w = w / (w * c).sum() * c.sum()                     # sentence-weighted mean = 1
    return torch.tensor(np.clip(w, 0.25, 10.0), dtype=torch.float32)


# ============================================================================ main
def param_fingerprint(params):
    h = hashlib.sha1()
    for p in params:
        h.update(p.detach().float().cpu().numpy().tobytes()[:4096])
    return h.hexdigest()


def write_submission(preds, test_docs, out_dir, name):
    clean = {}
    for d in test_docs:
        lab = preds[d.id]
        assert len(lab) == d.n and all(l in L2I for l in lab), d.id
        clean[d.id] = lab
    pj = os.path.join(out_dir, "prediction.json")
    json.dump(clean, open(pj, "w"), ensure_ascii=False)
    zp = os.path.join(out_dir, f"submission_{name}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
    return zp, len(clean), sum(len(v) for v in clean.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--method", required=True, choices=sorted(METHOD_DEFAULTS))
    ap.add_argument("--hp", default="{}", help="JSON dict of overrides")
    ap.add_argument("--paths", default="/kaggle/working/rr_paths.json",
                    help="JSON with data_dir, model_dir, tokenizer, evaluator, out_root")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if the experiment finished before")
    ap.add_argument("--model_dir", default=None, help="Phase-1 model folder for this run (default: paths.json)")
    ap.add_argument("--allow_count_mismatch", action="store_true",
                    help="train even if sentence counts differ from the brief")
    args = ap.parse_args()

    paths = json.load(open(args.paths))
    if args.model_dir:
        paths["model_dir"] = args.model_dir
        if os.path.exists(os.path.join(args.model_dir, "tokenizer.json")):
            paths["tokenizer"] = os.path.join(args.model_dir, "tokenizer.json")
    hp = {**DEFAULTS, **METHOD_DEFAULTS[args.method], **json.loads(args.hp)}
    unknown = set(json.loads(args.hp)) - set(DEFAULTS) - set(METHOD_DEFAULTS[args.method])
    if unknown:
        raise SystemExit(f"unknown hp keys: {sorted(unknown)}")
    name = ("_smoke_" + args.name) if args.smoke else args.name
    out_dir = os.path.join(paths["out_root"], name)
    if args.smoke and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    if os.path.exists(os.path.join(out_dir, "DONE")) and not args.force:
        log(f"{name} already finished ({out_dir}); use a new --name, or --force to redo it")
        return
    os.makedirs(out_dir, exist_ok=True)

    random.seed(hp["seed"]); np.random.seed(hp["seed"]); torch.manual_seed(hp["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"[{name}] GPU {torch.cuda.get_device_name(0)} | visible GPUs {torch.cuda.device_count()} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    else:
        log(f"[{name}] running on CPU")

    # ---- data
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(paths["tokenizer"])
    eos_id = tok.token_to_id("<|endoftext|>")
    assert eos_id is not None, "tokenizer has no <|endoftext|>"
    docs, _ = load_phase3(paths["data_dir"], seed=hp["seed"], verbose=False)
    if not args.smoke:
        from rr_data import EXPECTED_SENTENCES
        got = {k: sum(len(d["sentences"]) for d in docs if (d["split"], d["domain"]) == k) for k in EXPECTED_SENTENCES}
        bad = {k: (got[k], v) for k, v in EXPECTED_SENTENCES.items() if got[k] != v}
        if bad and not args.allow_count_mismatch:
            raise SystemExit(f"sentence counts differ from the brief (got, expected): {bad}")
    if args.smoke:
        keep = []
        for split in ("train", "dev", "test"):
            for dom in DOMAINS:
                keep += [d for d in docs if d["split"] == split and d["domain"] == dom][:2]
        docs = keep
        hp.update(epochs=2, patience=2, max_steps_per_epoch=10)
    describe(docs)
    enc = encode_docs(docs, tok, hp, eos_id)
    train_docs = [d for d in enc if d.split == "train"]
    dev_docs = [d for d in enc if d.split == "dev"]
    test_docs = [d for d in enc if d.split == "test"]
    tl = [len(s) for d in train_docs for s in d.sent_ids]
    log(f"sentence tokens (train): median {int(np.median(tl))}, p95 {int(np.percentile(tl, 95))}, max {max(tl)} "
        f"(cap {hp['max_sent_tokens']}); truncated {sum(t >= hp['max_sent_tokens'] for t in tl)}")

    # ---- model
    cfg = json.load(open(os.path.join(paths["model_dir"], "config.json")))
    model, n_loaded, tied, n_wrapped = build_model(cfg, hp, args.method, paths["model_dir"])
    model.to(device)
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Phase-1 tensors loaded {n_loaded} (strict, LM head tied={tied}) | PEFT modules {n_wrapped} | "
        f"params {tot / 1e6:.2f}M, trainable {trn / 1e6:.3f}M ({100 * trn / tot:.2f}%)")
    frozen = [p for n, p in model.named_parameters() if not p.requires_grad]
    frozen_fp0 = param_fingerprint(frozen[:6]) if frozen else None
    json.dump({"name": name, "method": args.method, "hp": hp, "model_cfg": cfg, "paths": paths,
               "params_total": tot, "params_trainable": trn}, open(os.path.join(out_dir, "run_config.json"), "w"), indent=1)

    # ---- optimiser
    new_mods = model.new_module_names()
    enc_dec, enc_nodec, head = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.split(".")[0] in new_mods:
            head.append(p)
        elif p.ndim >= 2:
            enc_dec.append(p)
        else:
            enc_nodec.append(p)
    groups = [g for g in [{"params": enc_dec, "lr": hp["lr"], "weight_decay": hp["weight_decay"]},
                          {"params": enc_nodec, "lr": hp["lr"], "weight_decay": 0.0},
                          {"params": head, "lr": hp["head_lr"], "weight_decay": hp["weight_decay"]}] if g["params"]]
    opt = torch.optim.AdamW(groups)
    n0 = len(train_windows(train_docs, hp["window"], random.Random(0)))
    spe = math.ceil(n0 / hp["batch_size"])
    if hp["max_steps_per_epoch"]:
        spe = min(spe, hp["max_steps_per_epoch"])
    total = spe * hp["epochs"]
    warm = max(1, int(hp["warmup"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, min(1.0, (total - s) / max(1, total - warm))))
    use_amp = hp["fp16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=2.0 ** 12, growth_interval=200)  # default 2^16 skips the first steps
    teacher, teacher_info = None, None
    if args.method == "kd":
        tdir = os.path.join(paths["out_root"], hp["teacher"])
        assert hp["teacher"] and os.path.exists(os.path.join(tdir, "DONE")), f"teacher run {tdir} has not finished"
        tmeta = json.load(open(os.path.join(tdir, "run_config.json")))
        thp = tmeta["hp"]
        for k in ("window", "max_sent_tokens", "pos_buckets"):
            assert thp[k] == hp[k], f"teacher/student must share {k}: {thp[k]} vs {hp[k]}"
        tmodel_dir = tmeta["paths"]["model_dir"]
        assert os.path.realpath(tmeta["paths"]["tokenizer"]) == os.path.realpath(paths["tokenizer"]) or \
            open(tmeta["paths"]["tokenizer"]).read() == open(paths["tokenizer"]).read(), "teacher uses another tokenizer"
        teacher, _, _, _ = build_model(tmeta["model_cfg"], thp, tmeta["method"], tmodel_dir)
        load_trainable_state(teacher, torch.load(os.path.join(tdir, "best.pt"), map_location="cpu"))
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        tm = json.load(open(os.path.join(tdir, "metrics.json")))
        teacher_info = {"name": hp["teacher"], "method": tmeta["method"], "params": sum(p.numel() for p in teacher.parameters()),
                        "layers": tmeta["model_cfg"]["n_layer"], "dev_f1_official": tm["dev_f1_official"]}
        log(f"teacher {teacher_info} | student layers {hp['student_layers']} "
            f"({len(hp['student_layers'])} of {cfg['n_layer']}) | kd_alpha {hp['kd_alpha']} T {hp['kd_T']}")

    cw = class_weights(train_docs, hp)
    cw = cw.to(device) if cw is not None else None
    log(f"windows/epoch ~{n0} | steps/epoch {spe} | total {total} | warmup {warm} | class weights "
        f"{None if cw is None else [round(x, 2) for x in cw.tolist()]}")

    ev = OfficialEvaluator(paths["evaluator"], os.path.join(out_dir, "eval"), dev_docs)

    # ---- resume
    start_ep, history, best, bad = 1, [], None, 0
    last_p, best_p = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt")
    if os.path.exists(last_p):
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        load_trainable_state(model, ck["model"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, history, best, bad = ck["epoch"] + 1, ck["history"], ck["best"], ck["bad"]
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        log(f"RESUMED after epoch {ck['epoch']} (best so far {best and best['dev_f1_official']})")
        del ck

    rng = random.Random(hp["seed"] * 1000 + start_ep)
    trainable = [p for p in model.parameters() if p.requires_grad]
    adapter_checked = start_ep > 1                      # on resume the check already ran
    for ep in range(start_ep, hp["epochs"] + 1):
        if bad >= hp["patience"]:
            break
        model.train()
        items = train_windows(train_docs, hp["window"], rng)
        rng.shuffle(items)
        t0, tot_loss, nb, nonfinite, gnorms, opt_steps = time.time(), 0.0, 0, 0, [], 0
        for k in range(0, len(items), hp["batch_size"]):
            if hp["max_steps_per_epoch"] and nb >= hp["max_steps_per_epoch"]:
                break
            ids, km, slot, ns, pos, dom, lab = collate(items[k:k + hp["batch_size"]], eos_id)
            ids, km, slot, pos, dom, lab = (t.to(device) for t in (ids, km, slot, pos, dom, lab))
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(ids, km, slot, ns.to(device), pos, dom)
            logits = logits.float()
            valid = lab != -100
            loss = F.cross_entropy(logits[valid], lab[valid], weight=cw, label_smoothing=hp["label_smoothing"])
            if teacher is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    t_logits = teacher(ids, km, slot, ns.to(device), pos, dom)
                T = hp["kd_T"]
                kd = F.kl_div(F.log_softmax(logits[valid] / T, -1), F.softmax(t_logits.float()[valid] / T, -1),
                              reduction="batchmean") * T * T
                loss = (1 - hp["kd_alpha"]) * loss + hp["kd_alpha"] * kd
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                nonfinite += 1
                if nonfinite == 1:
                    log(f"  non-finite loss at epoch {ep} step {nb + 1}: |logits| max "
                        f"{logits.detach().abs().nan_to_num(posinf=1e30).max().item():.3g}, lr {[g['lr'] for g in opt.param_groups]}")
                if nonfinite > 20:
                    raise RuntimeError("more than 20 non-finite losses in one epoch - stopping")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(trainable, hp["grad_clip"])
            scale_before = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() >= scale_before:              # optimizer actually stepped
                sched.step()
                opt_steps += 1
            if torch.isfinite(gn):
                gnorms.append(float(gn))
            tot_loss += loss.item(); nb += 1
            if ep == start_ep and nb == 1:
                no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
                log(f"  first step: loss {loss.item():.4f} | grad-norm {float(gn):.3f}"
                    f"{' (fp16 loss-scale warm-up: this step is skipped, normal)' if not torch.isfinite(gn) else ''}"
                    f" | trainable params without grad: "
                    f"{len(no_grad)} {no_grad[:3]}")
        train_s = time.time() - t0
        if nb == 0:
            raise RuntimeError("no optimisation step was taken this epoch")

        t1 = time.time()
        preds, probs = predict(model, dev_docs, hp, device, eos_id)
        off, _ = ev(preds, f"ep{ep}")
        rep = full_report(dev_docs, preds)
        mine = rep["combined"]["macro_f1"]
        if abs(mine - off["F1"]) > 1e-5:
            raise RuntimeError(f"in-script Macro-F1 {mine:.6f} != official {off['F1']:.6f}")
        rec = {"epoch": ep, "train_loss": tot_loss / nb, "dev_nll": dev_nll(dev_docs, probs),
               "dev_f1_official": off["F1"], "dev_p": off["Precision"], "dev_r": off["Recall"],
               **{f"dev_f1_{dm}": rep[dm]["macro_f1"] for dm in DOMAINS if dm in rep},
               "grad_norm_mean": float(np.mean(gnorms)) if gnorms else None, "nonfinite_steps": nonfinite,
               "steps": nb, "opt_steps": opt_steps, "train_s": train_s, "eval_s": time.time() - t1,
               "lr_now": [g["lr"] for g in opt.param_groups]}
        improved = best is None or off["F1"] > best["dev_f1_official"]
        if improved:
            best, bad = dict(rec), 0
            torch.save(trainable_state(model), best_p)
            json.dump(preds, open(os.path.join(out_dir, "dev_predictions_best.json"), "w"), ensure_ascii=False)
            json.dump(rep, open(os.path.join(out_dir, "dev_report_best.json"), "w"), indent=1)
        else:
            bad += 1
        history.append(rec)
        log(f"  ep {ep:>2}/{hp['epochs']} loss {rec['train_loss']:.4f} dev-nll {rec['dev_nll']:.4f} | OFFICIAL dev "
            f"Macro-F1 {off['F1']:.4f} (P {off['Precision']:.3f} R {off['Recall']:.3f}) | CL {rec.get('dev_f1_CL', 0):.4f} "
            f"IT {rec.get('dev_f1_IT', 0):.4f}{'  *' if improved else ''}  [{train_s:.0f}s+{rec['eval_s']:.0f}s]")
        if opt_steps == 0:
            log(f"  WARNING: all {nb} steps of epoch {ep} were skipped by the fp16 loss scaler (overflow) - no update yet")
        if not adapter_checked and frozen_fp0 is not None and opt_steps > 0:
            adapter_checked = True
            assert param_fingerprint(frozen[:6]) == frozen_fp0, "frozen pretrained weights changed!"
            if args.method in ("lora", "rslora", "lorma"):
                bn = sum(float(m.B.detach().norm()) for m in model.modules() if isinstance(m, (LoRALinear, LoRMALinear)))
                assert bn > 0, "LoRA/LoRMA B matrices are still zero - adapters did not update"
                log(f"  check: frozen weights unchanged, adapter |B| sum = {bn:.4f}")
            elif args.method == "ia3":
                dv = sum(float((m.l_k.detach() - 1).abs().sum() + (m.l_v.detach() - 1).abs().sum()) for m in model.modules() if isinstance(m, IA3QKV))
                assert dv > 0, "IA3 vectors did not move"
                log(f"  check: frozen weights unchanged, IA3 |l-1| sum = {dv:.4f}")
        torch.save({"model": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "history": history, "best": best, "bad": bad,
                    "py_rng": random.getstate(), "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()},
                   last_p + ".tmp")
        os.replace(last_p + ".tmp", last_p)
        if bad >= hp["patience"]:
            log(f"  early stop: no official dev improvement for {hp['patience']} epochs")

    if frozen_fp0 is not None and not adapter_checked:
        raise RuntimeError("no optimizer step ever succeeded, so frozen/adapter checks never ran")

    # ---- best checkpoint -> dev report + test submission
    load_trainable_state(model, torch.load(best_p, map_location="cpu"))
    model.to(device)
    preds, _ = predict(model, dev_docs, hp, device, eos_id)
    off, off_txt = ev(preds, "best")
    if abs(off["F1"] - best["dev_f1_official"]) > 1e-6:      # fp16 kernels can flip a near-tie
        log(f"  NOTE: reloaded best checkpoint scores {off['F1']:.6f} vs {best['dev_f1_official']:.6f} during training")
    rep = json.load(open(os.path.join(out_dir, "dev_report_best.json")))
    t0 = time.time()
    test_preds, _ = predict(model, test_docs, hp, device, eos_id)
    zp, n_docs, n_sent = write_submission(test_preds, test_docs, out_dir, name)
    if n_docs != 10 and not args.smoke:
        log(f"  WARNING: submission has {n_docs} test documents (brief expects 10)")

    log("\n" + "=" * 90 + f"\n{name} | best epoch {best['epoch']} | OFFICIAL dev Macro-F1 {off['F1']:.4f}  "
        f"(P {off['Precision']:.4f} R {off['Recall']:.4f})")
    log("official evaluator output:\n" + off_txt.strip())
    for dm in ["combined"] + [d for d in DOMAINS if d in rep]:
        log(f"  {dm:<8} macro-F1 {rep[dm]['macro_f1']:.4f}  P {rep[dm]['macro_p']:.4f}  R {rep[dm]['macro_r']:.4f}")
    log(f"  {'label':<24}{'P':>7}{'R':>7}{'F1':>7}{'supp':>6} | CL F1  IT F1")
    for l in LABELS:
        c = rep["combined"]["per_class"][l]
        dom_f = "  ".join(f"{rep[d]['per_class'][l]['f1']:.3f}" if d in rep else "  -  " for d in DOMAINS)
        log(f"  {l:<24}{c['p']:>7.3f}{c['r']:>7.3f}{c['f1']:>7.3f}{c['support']:>6} | {dom_f}")
    log(f"  confusion (rows = gold, cols = predicted, order = {LABELS}):")
    for l, row in zip(LABELS, rep["combined"]["confusion"]):
        log(f"  {l[:12]:<12} " + " ".join(f"{x:>4}" for x in row))
    log(f"  submission: {zp}  ({n_docs} docs, {n_sent} sentences, prediction.json at zip root)")

    metrics = {"name": name, "method": args.method, "best_epoch": best["epoch"],
               "dev_f1_official": off["F1"], "dev_p_official": off["Precision"], "dev_r_official": off["Recall"],
               **{f"dev_f1_{d}": rep[d]["macro_f1"] for d in DOMAINS if d in rep},
               "per_class_f1": {l: rep["combined"]["per_class"][l]["f1"] for l in LABELS},
               "params_total": tot, "params_trainable": trn, "layers": len(model.encoder.blocks),
               "teacher": teacher_info, "model_dir": paths["model_dir"], "history": history, "hp": hp,
               "test_docs": n_docs, "test_sentences": n_sent, "test_infer_s": time.time() - t0,
               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None}
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    open(os.path.join(out_dir, "DONE"), "w").write(time.ctime())
    if args.smoke:
        log("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
