"""Phase 1 data pipeline: (optional) tokenizer training -> streamed corpus -> uint16 token memmaps.

Reads the four permitted sources (train_acts / train_high_courts / train_supreme_court / train_tribunals
.json, 0.7-10.7 GB each) with ijson, so no file is ever fully in memory. Per-document validation draw
(seed 42): 5% of Supreme Court documents, 1% of the others - the same held-out split in every run.

Outputs in --out_dir:
  train.bin / val.bin              uint16 token ids, documents separated by <|endoftext|>
  train_types.bin / val_types.bin  uint8 citation types aligned with the tokens (--citation_types, JurisNetLM)
  eval_pairs.json                  held-out judgment prompts (--prompt_words) + true next --target_words words
  tokenizer.json                   copy of the tokenizer used (or the one trained with --train_tokenizer)

Examples
  # LegalLM: train the 32K byte-level BPE, then build the token pool
  python prepare_data.py --hf --out_dir /tmp/cs781_data --train_tokenizer --vocab_size 32000 \
      --caps acts=60e6,high_court=800e6,supreme_court=300e6,tribunal=300e6 --eval_pairs 64
  # GPT-2 baseline: standard GPT-2 tokenizer, Supreme Court only, 200M train / 5M validation tokens
  python prepare_data.py --hf --out_dir /tmp/gpt2_data --tokenizer gpt2 --only supreme_court \
      --caps supreme_court=200e6 --val_tokens 5e6 --eval_pairs 64
  # JurisNetLM: LegalLM tokenizer + citation types
  python prepare_data.py --hf --out_dir /tmp/juris_data --tokenizer legallm/tokenizer.json --citation_types ...
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF_REPO = "Exploration-Lab/CS-781-Capstone"
HF_PREFIX = "Phase_1_Pretraining/Train/"
KINDS = ["acts", "high_court", "supreme_court", "tribunal"]
JUDGMENT_KINDS = {"high_court", "supreme_court", "tribunal"}
EOS = "<|endoftext|>"
BYTES_PER_TOKEN = 4.5          # rough, only used to choose a keep-probability per file


def file_kind(path):
    n = os.path.basename(path).lower()
    return next((k for k in KINDS if k in n or k.replace("_", "") in n.replace("_", "")), "other")


def list_files(args):
    if args.files:
        return sorted(args.files)
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get(args.hf_token_env)
    names = sorted(f for f in HfApi().list_repo_files(HF_REPO, repo_type="dataset", token=token)
                   if f.startswith(HF_PREFIX) and f.endswith(".json"))
    return [hf_hub_download(HF_REPO, f, repo_type="dataset", token=token) for f in names]


def iter_docs(path, val_frac, seed):
    """Yields (is_validation, text) per document, streaming; the draw depends only on the seed and order."""
    import ijson
    rng = random.Random(seed)
    with open(path, "rb") as fh:
        head = fh.read(4096).lstrip()
    with open(path, "rb") as fh:
        items = (ijson.items(fh, "item", use_float=True) if head[:1] == b"["
                 else (v for _, v in ijson.kvitems(fh, "", use_float=True)))
        for rec in items:
            text = rec.get("text") if isinstance(rec, dict) else None
            yield rng.random() < val_frac, (text if isinstance(text, str) else "")


def load_tokenizer(spec):
    from tokenizers import Tokenizer
    if spec == "gpt2":
        return Tokenizer.from_pretrained("gpt2")              # standard 50,257-token GPT-2 BPE
    return Tokenizer.from_file(spec)


def train_tokenizer(files, args, out_path):
    """32K byte-level BPE (no normaliser: case, punctuation and section symbols are kept)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    per_file = max(1, args.tokenizer_docs // len(files))

    def sample():
        for p in files:
            n = 0
            for is_val, text in iter_docs(p, val_frac(p, args), args.seed):
                if text and not is_val:                       # never train the tokenizer on held-out docs
                    yield text[:20000]
                    n += 1
                    if n >= per_file:
                        break
    tok.train_from_iterator(sample(), trainer=trainer)
    assert tok.token_to_id(EOS) == 0
    tok.save(out_path)
    print(f"trained tokenizer: vocab {tok.get_vocab_size()} -> {out_path}")
    return tok


def val_frac(path, args):
    return args.sc_val_frac if file_kind(path) == "supreme_court" else args.other_val_frac


def parse_caps(s):
    caps = {}
    for part in filter(None, (s or "").split(",")):
        k, v = part.split("=")
        caps[k.strip()] = int(float(v))
    return caps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--files", nargs="*", help="local train_*.json files (default: download with --hf)")
    ap.add_argument("--hf", action="store_true", help="download the Phase-1 train files from the course HF dataset")
    ap.add_argument("--hf_token_env", default="HF_TOKEN")
    ap.add_argument("--tokenizer", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "legallm", "tokenizer.json"),
                    help="tokenizer.json path or 'gpt2'")
    ap.add_argument("--train_tokenizer", action="store_true", help="train a new byte-level BPE first")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--tokenizer_docs", type=int, default=200_000)
    ap.add_argument("--only", nargs="*", choices=KINDS, help="restrict to these sources (GPT-2 baseline: supreme_court)")
    ap.add_argument("--caps", default="", help="train-token cap per source, e.g. acts=60e6,high_court=800e6")
    ap.add_argument("--val_tokens", type=float, default=3e6)
    ap.add_argument("--sc_val_frac", type=float, default=0.05)
    ap.add_argument("--other_val_frac", type=float, default=0.01)
    ap.add_argument("--sample_seed", type=int, default=7, help="which training documents are kept under a cap")
    ap.add_argument("--seed", type=int, default=42, help="validation draw (keep fixed across runs)")
    ap.add_argument("--citation_types", action="store_true")
    ap.add_argument("--eval_pairs", type=int, default=0, help="held-out (prompt, reference) pairs to write")
    ap.add_argument("--prompt_words", type=int, default=512)
    ap.add_argument("--target_words", type=int, default=128)
    args = ap.parse_args()
    if not args.files and not args.hf:
        ap.error("give --files or --hf")
    os.makedirs(args.out_dir, exist_ok=True)

    files = [p for p in list_files(args) if not args.only or file_kind(p) in args.only]
    assert files, "no input files"
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    if args.train_tokenizer:
        tok = train_tokenizer(files, args, tok_path)
    else:
        tok = load_tokenizer(args.tokenizer)
        tok.save(tok_path)
    eos_id = tok.token_to_id(EOS)
    assert eos_id is not None and tok.get_vocab_size() <= 65535, "need <|endoftext|> and a uint16-sized vocab"
    if args.citation_types:
        from backbones.citations import token_types
    encode = getattr(tok, "encode_batch_fast", tok.encode_batch) if not args.citation_types else tok.encode_batch

    caps = parse_caps(args.caps)
    val_per_file = int(args.val_tokens) // len(files)
    min_eval_words = args.prompt_words + args.target_words + 20
    per_kind_eval = -(-args.eval_pairs // max(1, len([f for f in files if file_kind(f) in JUDGMENT_KINDS])))  # ceil
    handles = {n: open(os.path.join(args.out_dir, n), "wb") for n in ("train.bin", "val.bin")}
    if args.citation_types:
        handles.update({n: open(os.path.join(args.out_dir, n), "wb") for n in ("train_types.bin", "val_types.bin")})

    def write(split, texts):
        if not texts:
            return 0
        encs = encode(texts)
        ids = np.concatenate([np.asarray(e.ids + [eos_id], dtype=np.uint16) for e in encs])
        ids.tofile(handles[f"{split}.bin"])
        if args.citation_types:
            ty = np.concatenate([np.append(token_types(t, e.offsets), 0).astype(np.uint8) for t, e in zip(texts, encs)])
            ty.tofile(handles[f"{split}_types.bin"])
        return len(ids)

    eval_docs, t0, tot = {}, time.time(), {"train": 0, "val": 0}
    for path in files:
        kind = file_kind(path)
        cap = caps.get(kind, caps.get("default", 10 ** 12))
        keep_p = min(1.0, 1.3 * cap * BYTES_PER_TOKEN / max(1, os.path.getsize(path)))
        rng_keep, rng_res = random.Random(args.sample_seed), random.Random(args.seed + 2)
        want_eval = 3 * per_kind_eval if (args.eval_pairs and kind in JUDGMENT_KINDS) else 0
        n_tr = n_va = n_long = 0
        tr_buf, va_buf, res = [], [], []
        for is_val, text in iter_docs(path, val_frac(path, args), args.seed):
            if not text:
                continue
            if is_val:
                if n_va < val_per_file:
                    va_buf.append(text)
                    if len(va_buf) >= 64:
                        n_va += write("val", va_buf); va_buf = []
                if want_eval and len(text.split()) >= min_eval_words:          # reservoir sample
                    n_long += 1
                    if len(res) < want_eval:
                        res.append(text)
                    elif (j := rng_res.randrange(n_long)) < want_eval:
                        res[j] = text
            elif n_tr < cap and rng_keep.random() <= keep_p:
                tr_buf.append(text)
                if len(tr_buf) >= 256:
                    n_tr += write("train", tr_buf); tr_buf = []
            elif n_tr >= cap and n_va >= val_per_file and len(res) >= want_eval:
                break
        n_tr += write("train", tr_buf)
        if n_va < val_per_file:
            n_va += write("val", va_buf)
        eval_docs[kind] = res
        tot["train"] += n_tr; tot["val"] += n_va
        print(f"  {os.path.basename(path)} [{kind}]: train {n_tr / 1e6:.1f}M | val {n_va / 1e6:.2f}M tokens | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    for h in handles.values():
        h.close()
    print(f"token pool: train {tot['train'] / 1e6:.1f}M | val {tot['val'] / 1e6:.2f}M -> {args.out_dir}")

    if args.eval_pairs:
        rng, by_kind = random.Random(args.seed), {}
        for kind, docs in sorted(eval_docs.items()):
            for text in docs[:per_kind_eval]:
                spans = [m.start() for m in re.finditer(r"\S+", text)]
                words = text.split()
                c = rng.randint(args.prompt_words, len(words) - args.target_words)
                by_kind.setdefault(kind, []).append({"kind": kind, "reference": " ".join(words[c:c + args.target_words]),
                                                     "prompt": text[spans[c - args.prompt_words]:spans[c]].rstrip()})
        pairs = [x for i in range(per_kind_eval) for k in sorted(by_kind) for x in by_kind[k][i:i + 1]]  # interleave courts
        with open(os.path.join(args.out_dir, "eval_pairs.json"), "w", encoding="utf-8") as f:
            json.dump(pairs[:args.eval_pairs], f, ensure_ascii=False)
        print(f"eval pairs: {min(len(pairs), args.eval_pairs)} -> eval_pairs.json")


if __name__ == "__main__":
    main()
