"""Phase 1 decoding: legal text continuation -> Codabench submission, plus the local decoding comparison.

Decoding specs (combine with "_"):  greedy | beam<K> | sample_t<temp>_p<top_p>,  rep<penalty>,  nr<n>
  greedy          plain greedy
  greedy_rep1.1   repetition penalty 1.1 on already-seen tokens (CTRL-style)
  greedy_nr4      greedy, no 4-gram may repeat (prompt included)        <- submitted LegalLM decoding
  beam4_nr4       beam search, 4 beams, 4-gram blocking
EOS is blocked and every continuation is clipped to --target_words words (the test asks for 128).

  # submission (writes prediction.json + submission.zip)
  python generate.py --model_dir legallm_final --test test.json --decoding greedy_nr4 --out_dir out/
  # local comparison on held-out pairs from prepare_data.py (BERTScore-F1 = the leaderboard metric)
  python generate.py --model_dir gpt2_final --tokenizer gpt2 --eval_pairs eval_pairs.json \
      --decodings greedy,greedy_rep1.1,beam4_nr4,greedy_nr4

Works for every family: decoder-only models (LegalLM, GPT-2, SWA-Lite, JurisNetLM) through their KV cache,
the encoder-decoder hybrid through encode-once / cached decoding.
"""
import argparse
import json
import os
import re
import sys
import time
import zipfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backbones import load_backbone, EncDecLM, JurisNetLM  # noqa: E402


def parse_spec(spec):
    p = dict(beams=1, greedy=True, temperature=1.0, top_p=1.0, rep=1.0, nr=0)
    for part in spec.split("_"):
        if part.startswith("beam"):
            p["beams"] = int(part[4:])
        elif part == "sample":
            p["greedy"] = False
        elif part.startswith("t") and not p["greedy"]:
            p["temperature"] = float(part[1:])
        elif part.startswith("p") and not p["greedy"]:
            p["top_p"] = float(part[1:])
        elif part.startswith("rep"):
            p["rep"] = float(part[3:])
        elif part.startswith("nr"):
            p["nr"] = int(part[2:])
        elif part != "greedy":
            raise ValueError(f"bad decoding spec part {part!r} in {spec!r}")
    return p


# ------------------------------------------------------------------ steppers: prefill once, then one token per call
class DecoderStepper:
    """Left-padded batch, per-row RoPE/learned positions, KV cache (LegalLM generate logic, any decoder)."""

    def __init__(self, model, eos_id, type_ids=None):
        self.m, self.eos, self.jur = model, eos_id, isinstance(model, JurisNetLM)
        self.prompt_types = type_ids

    def _call(self, ids, **kw):
        if self.jur:
            logits, tl = self.m(ids, return_type_logits=True, **kw)
            self.next_type = tl[:, -1].argmax(-1)             # predicted type of the token being generated
            return logits[:, -1].float()
        return self.m(ids, **kw)[:, -1].float()

    def prefill(self, prompts):
        dev = next(self.m.parameters()).device
        B, L = len(prompts), max(len(p) for p in prompts)
        ids = torch.full((B, L), self.eos, dtype=torch.long)
        valid = torch.zeros((B, L), dtype=torch.bool)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p)
            valid[i, L - len(p):] = True
        ids, valid = ids.to(dev), valid.to(dev)
        pos = (valid.long().cumsum(1) - 1).clamp(min=0)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
        mask = ((causal[None] & valid[:, None, :]) | torch.eye(L, dtype=torch.bool, device=dev)[None])[:, None]
        self.caches = [dict() for _ in self.m.blocks]
        kw = {}
        if self.jur and self.prompt_types is not None:
            ty = torch.zeros((B, L), dtype=torch.long)
            for i, t in enumerate(self.prompt_types):
                ty[i, L - len(t):] = torch.as_tensor(t, dtype=torch.long)
            kw["type_ids"] = ty.to(dev)
        logits = self._call(ids, positions=pos, attn_mask=mask, caches=self.caches, last_only=True, **kw)
        self.pos, self.key_valid = valid.sum(1), valid
        return logits

    def step(self, tok):
        self.key_valid = torch.cat([self.key_valid, torch.ones_like(tok, dtype=torch.bool)[:, None]], 1)
        kw = {"type_ids": self.next_type[:, None]} if self.jur else {}
        logits = self._call(tok[:, None], positions=self.pos.clamp(max=self.m.cfg.max_seq_len - 1)[:, None],
                            attn_mask=self.key_valid[:, None, None, :], caches=self.caches, last_only=True, **kw)
        self.pos = self.pos + 1
        return logits

    def reorder(self, idx):
        for c in self.caches:
            c["k"], c["v"] = c["k"][idx], c["v"][idx]
        self.pos, self.key_valid = self.pos[idx], self.key_valid[idx]
        if self.jur:
            self.next_type = self.next_type[idx]


class EncDecStepper:
    """Encode the prompt once (bidirectional); decoder starts from the last prompt token, cached."""

    def __init__(self, model, eos_id, type_ids=None):
        self.m, self.eos = model, eos_id

    def prefill(self, prompts):
        dev = next(self.m.parameters()).device
        B, L = len(prompts), max(len(p) for p in prompts)
        ids = torch.full((B, L), self.eos, dtype=torch.long)
        valid = torch.zeros((B, L), dtype=torch.bool)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p)
            valid[i, L - len(p):] = True
        ids, valid = ids.to(dev), valid.to(dev)
        pos = (valid.long().cumsum(1) - 1).clamp(min=0)
        self.memory, self.mem_valid = self.m.encode_ids(ids, valid, pos), valid
        self.caches = [dict() for _ in self.m.dec_blocks]
        self.pos = pos[:, -1]
        return self._dec(ids[:, -1])

    def _dec(self, tok):
        h = self.m.decode(tok[:, None], self.memory, self.mem_valid,
                          positions=self.pos.clamp(max=self.m.cfg.max_seq_len - 1)[:, None], caches=self.caches)
        return self.m.head(h[:, -1]).float()

    def step(self, tok):
        self.pos = self.pos + 1
        return self._dec(tok)

    def reorder(self, idx):
        for c in self.caches:
            for k in ("k", "v", "mk", "mv"):
                c[k] = c[k][idx]
        self.memory, self.mem_valid, self.pos = self.memory[idx], self.mem_valid[idx], self.pos[idx]


# ------------------------------------------------------------------ decoding loops
def banned_tokens(hist, n):
    """Tokens that would complete an n-gram already present in hist (prompt + generated)."""
    if n <= 0 or len(hist) < n - 1:
        return []
    key = tuple(hist[len(hist) - n + 1:]) if n > 1 else ()
    return [hist[j + n - 1] for j in range(len(hist) - n + 1) if tuple(hist[j:j + n - 1]) == key]


def penalise(logits, seen, rep):
    if rep == 1.0:
        return logits
    pen = torch.where(logits > 0, logits / rep, logits * rep)
    return torch.where(seen, pen, logits)


@torch.no_grad()
def decode_batch(model, prompts, p, eos_id, max_new, enough, type_ids=None, seed=0):
    """prompts: list of token-id lists -> list of generated token-id lists. `enough(ids)` = stop test."""
    Stepper = EncDecStepper if isinstance(model, EncDecLM) else DecoderStepper
    K, V = p["beams"], model.cfg.vocab_size
    if K > 1:
        prompts = [q for q in prompts for _ in range(K)]
        type_ids = None if type_ids is None else [t for t in type_ids for _ in range(K)]
    st = Stepper(model, eos_id, type_ids)
    dev = next(model.parameters()).device
    logits = st.prefill(prompts)
    N = len(prompts)
    hist = [list(q) for q in prompts]
    gen = [[] for _ in range(N)]
    seen = torch.zeros(N, V, dtype=torch.bool, device=dev)
    if p["rep"] != 1.0:
        for i, q in enumerate(prompts):
            seen[i, torch.as_tensor(q, device=dev)] = True
    scores = torch.zeros(N, device=dev)
    if K > 1:
        scores.view(-1, K)[:, 1:] = float("-inf")                 # all beams start identical
    g = torch.Generator(device=dev).manual_seed(seed)
    for t in range(max_new):
        logits[:, eos_id] = float("-inf")
        logits = penalise(logits, seen, p["rep"])
        for i in range(N):
            b = banned_tokens(hist[i], p["nr"])
            if b:
                logits[i, b] = float("-inf")
        if K > 1:
            cand = (scores[:, None] + torch.log_softmax(logits, -1)).view(-1, K * V)
            top, idx = cand.topk(K, dim=-1)
            origin = (idx // V + torch.arange(0, N, K, device=dev)[:, None]).view(-1)
            nxt = (idx % V).view(-1)
            scores = top.view(-1)
            st.reorder(origin)
            o = origin.tolist()
            hist, gen, seen = [list(hist[j]) for j in o], [list(gen[j]) for j in o], seen[origin]
        elif p["greedy"]:
            nxt = logits.argmax(-1)
        else:
            probs = torch.softmax(logits / p["temperature"], -1)
            sp, si = probs.sort(-1, descending=True)
            sp = sp.masked_fill(sp.cumsum(-1) - sp > p["top_p"], 0.0)
            nxt = si.gather(1, torch.multinomial(sp, 1, generator=g)).squeeze(1)
        for i, tok in enumerate(nxt.tolist()):
            hist[i].append(tok); gen[i].append(tok)
        seen[torch.arange(N, device=dev), nxt] = True
        if t + 1 >= 32 and (t + 1) % 16 == 0 and all(enough(x) for x in gen[::K]):
            break
        if t + 1 < max_new:
            logits = st.step(nxt)
    return gen[::K]                                                 # beams are sorted: index 0 is the best


def clip_words(text, n):
    return " ".join(text.split()[:n])


def run(model, tok, prompts_text, spec, args):
    p = parse_spec(spec)
    eos = tok.token_to_id("<|endoftext|>")
    max_new = min(args.max_new, model.cfg.max_seq_len - 64)
    keep = model.cfg.max_seq_len - max_new
    encs = tok.encode_batch([t.rstrip() for t in prompts_text])
    prompts = [(e.ids[-keep:] or [eos]) for e in encs]
    types = None
    if isinstance(model, JurisNetLM):
        from backbones.citations import token_types
        types = [token_types(t.rstrip(), e.offsets)[-keep:] for t, e in zip(prompts_text, encs)]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))      # length-sorted batches
    outs, bs = [None] * len(prompts), max(1, args.batch_size // p["beams"])
    enough = lambda ids: len(tok.decode(ids).split()) > args.target_words  # noqa: E731
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        gen = decode_batch(model, [prompts[i] for i in idx], p, eos, max_new, enough,
                           None if types is None else [types[i] for i in idx], seed=args.seed + s)
        for i, g in zip(idx, gen):
            outs[i] = clip_words(tok.decode(g), args.target_words)
        print(f"  [{spec}] {min(s + bs, len(order))}/{len(order)} prompts | {time.time() - t0:.0f}s", flush=True)
    return outs


def metrics(preds, refs, device):
    out = {}
    try:
        from bert_score import BERTScorer                     # roberta-large, layer 17: the leaderboard metric
        out["bertscore_f1"] = float(BERTScorer(lang="en", device=device).score(preds, refs)[2].mean())
    except ImportError:
        print("bert_score not installed -> BERTScore skipped")
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        out["rougeL_f1"] = sum(sc.score(r, h)["rougeL"].fmeasure for h, r in zip(preds, refs)) / len(preds)
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True, help="config.json + pytorch_model.bin (+ tokenizer.json)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json or 'gpt2' (default: <model_dir>/tokenizer.json)")
    ap.add_argument("--test", help="Phase-1 test.json ({id: {'text': prompt}})")
    ap.add_argument("--eval_pairs", help="eval_pairs.json from prepare_data.py")
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--decoding", default="greedy_nr4")
    ap.add_argument("--decodings", default=None, help="comma list, compared on --eval_pairs")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--target_words", type=int, default=128)
    ap.add_argument("--max_new", type=int, default=360)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tspec = args.tokenizer or os.path.join(args.model_dir, "tokenizer.json")
    tok = Tokenizer.from_pretrained("gpt2") if tspec == "gpt2" else Tokenizer.from_file(tspec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(args.model_dir).to(device).eval()
    if device == "cuda":
        model.half()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.eval_pairs:
        pairs = json.load(open(args.eval_pairs, encoding="utf-8"))[:args.n_eval]
        res = {}
        for spec in (args.decodings or args.decoding).split(","):
            preds = run(model, tok, [x["prompt"] for x in pairs], spec, args)
            res[spec] = metrics(preds, [x["reference"] for x in pairs], device)
            print(f"{spec:<16} {res[spec]}")
        json.dump(res, open(os.path.join(args.out_dir, "decoding_comparison.json"), "w"), indent=2)
    if args.test:
        test = json.load(open(args.test, encoding="utf-8"))
        ids = list(test)
        outs = run(model, tok, [test[i]["text"] for i in ids], args.decoding, args)
        pj = os.path.join(args.out_dir, "prediction.json")
        json.dump({str(i): {"text": o} for i, o in zip(ids, outs)}, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with zipfile.ZipFile(os.path.join(args.out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(pj, arcname="prediction.json")
        short = sum(len(o.split()) < args.target_words for o in outs)
        print(f"wrote {len(outs)} predictions ({short} shorter than {args.target_words} words) -> submission.zip")


if __name__ == "__main__":
    main()
