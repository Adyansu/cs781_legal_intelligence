"""CPU smoke test for the backbone families (python tests/smoke_test.py, ~1 min, no data needed).

1. full-size configs: parameter counts and the 50M-300M capstone rule
2. tiny configs: finite pretraining loss + backward for every family
3. KV-cached greedy decoding == greedy by full recomputation (decoders and encoder-decoder)
4. downstream compatibility: encode() both ways, a P4/P5/P7-style block loop, LoRA on qkv/proj/w12/w3
5. save -> load_backbone round trip; shipped tokenizer + citation tagger
"""
import json
import math
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase1_pretraining"))
from backbones import build_backbone, load_backbone, count_parameters, encode, EncDecLM  # noqa: E402
from generate import decode_batch, parse_spec  # noqa: E402

torch.manual_seed(0)
CFG_DIR = os.path.join(ROOT, "phase1_pretraining", "configs")
EXPECTED = {"legallm": 109_529_856, "gpt2_baseline": 59_461_056, "swa_lite": 57_291_264,
            "jurisnet": 109_534_467, "encdec_hybrid": 123_691_008}
TINY = {
    "legallm": dict(vocab_size=97, n_layer=2, n_head=2, d_model=32, d_ff=64, max_seq_len=64),
    "gpt2": dict(arch="gpt2", vocab_size=97, n_layer=2, n_head=2, d_model=32, d_ff=96, max_seq_len=64),
    "swa_lite": dict(arch="swa_lite", vocab_size=97, n_layer=2, n_head=2, d_model=32, d_ff=64, max_seq_len=64, window=8),
    "jurisnet": dict(arch="jurisnet", vocab_size=97, n_layer=2, n_head=2, d_model=32, d_ff=64, max_seq_len=64),
    "encdec": dict(arch="encdec", vocab_size=97, n_enc_layer=2, n_dec_layer=2, n_head=2, d_model=32, d_ff=64,
                   max_seq_len=64, split_choices=[8, 12]),
}

print("1. full-size parameter counts")
for name, n_exp in EXPECTED.items():
    with torch.device("meta"):
        m = build_backbone(json.load(open(os.path.join(CFG_DIR, name + ".json")))["model"])
    n = count_parameters(m)
    print(f"   {name:<14} {n:>12,}  ({n / 1e6:.2f}M)")
    assert n == n_exp and 50e6 <= n <= 300e6, (name, n)

print("2. tiny models: loss + backward")
models = {}
for name, cfg in TINY.items():
    m = build_backbone(cfg)
    x = torch.randint(1, 97, (3, 24)); y = torch.randint(1, 97, (3, 24))
    kw = {"type_ids": torch.randint(0, 3, (3, 24)), "type_targets": torch.randint(0, 3, (3, 24))} if name == "jurisnet" else {}
    loss = m(x, targets=y, **kw)
    loss.backward()
    assert math.isfinite(loss.item()) and abs(loss.item() - math.log(97)) < 1.0, (name, loss.item())
    models[name] = m.eval()
    print(f"   {name:<9} loss {loss.item():.3f} (ln V = {math.log(97):.3f})")

print("3. cached greedy == recomputed greedy")


@torch.no_grad()
def greedy_reference(m, prompt, n):
    seq = list(prompt)
    for _ in range(n):
        ids = torch.tensor([seq])
        if isinstance(m, EncDecLM):
            p = len(prompt)
            mem = m.encode_ids(ids[:, :p])
            h = m.decode(ids[:, p - 1:], mem, positions=torch.arange(p - 1, len(seq)))
            logits = m.head(h)[0, -1]
        else:
            logits = m(ids)[0, -1]
        logits[0] = float("-inf")                       # EOS (id 0) blocked, as in generate.py
        seq.append(int(logits.argmax()))
    return seq[len(prompt):]


for name, m in models.items():
    prompts = [list(range(5, 20)), list(range(30, 37))]      # different lengths -> left padding exercised
    got = decode_batch(m, prompts, parse_spec("greedy"), 0, 10, lambda ids: False)
    ref = [greedy_reference(m, p, 10) for p in prompts]
    assert got == ref, (name, got, ref)
    beam = decode_batch(m, prompts, parse_spec("beam3_nr2"), 0, 6, lambda ids: False)
    assert all(len(b) == 6 for b in beam)
    print(f"   {name:<9} ok (greedy matches, beam3_nr2 runs)")


class LoRALinear(nn.Module):                             # same as the downstream notebooks
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base, self.s = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + self.s * F.linear(F.linear(x, self.A), self.B)


print("4. downstream compatibility")
for name, m in models.items():
    ids = torch.randint(1, 97, (2, 16)); km = torch.ones(2, 16, dtype=torch.bool); km[1, 10:] = False
    hb, hc = encode(m, ids, km, True), encode(m, ids, km, False)
    T = ids.shape[1]
    cos, sin = m.rope_cos[:T][None, None], m.rope_sin[:T][None, None]
    x = m.tok(ids)
    for blk in m.blocks:                                 # exactly how P4/P5/P7 run the backbone
        blk.attn.qkv, blk.attn.proj = LoRALinear(blk.attn.qkv), LoRALinear(blk.attn.proj)
        blk.mlp.w12, blk.mlp.w3 = LoRALinear(blk.mlp.w12), LoRALinear(blk.mlp.w3)
        x = blk(x, cos, sin)
    assert hb.shape == hc.shape == x.shape == (2, 16, m.cfg.d_model)
    print(f"   {name:<9} encode bi/causal + LoRA block loop ok")

print("5. load_backbone round trip, tokenizer, citation tagger")
with tempfile.TemporaryDirectory() as d:
    m = build_backbone(TINY["swa_lite"])
    torch.save(m.state_dict(), os.path.join(d, "pytorch_model.bin"))
    json.dump(TINY["swa_lite"], open(os.path.join(d, "config.json"), "w"))
    m2 = load_backbone(d)
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
from tokenizers import Tokenizer  # noqa: E402
from backbones.citations import token_types, CASE, STATUTE  # noqa: E402
tok = Tokenizer.from_file(os.path.join(ROOT, "phase1_pretraining", "legallm", "tokenizer.json"))
text = "The appeal under Section 302 of the Indian Penal Code was dismissed; see AIR 1984 SC 1728 and (2007) 1 SCC 683."
enc = tok.encode(text)
ty = token_types(text, enc.offsets)
assert tok.get_vocab_size() == 32000 and tok.token_to_id("<|endoftext|>") == 0
assert (ty == CASE).sum() >= 6 and (ty == STATUTE).sum() >= 2, ty
print(f"   tokenizer vocab {tok.get_vocab_size()} | {len(enc.ids)} tokens, {(ty == CASE).sum()} case-citation, "
      f"{(ty == STATUTE).sum()} statute tokens")
print("ALL SMOKE TESTS PASSED")
