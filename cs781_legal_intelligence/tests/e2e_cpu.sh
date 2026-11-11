#!/usr/bin/env bash
# End-to-end CPU check of the Phase-1 tooling on a synthetic corpus (~2 min):
# prepare_data.py (tokenizer training + citation types + eval pairs) -> pretrain.py (every family, tiny)
# -> generate.py (eval pairs + test set -> submission.zip).   Usage: bash tests/e2e_cpu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
P1="$ROOT/phase1_pretraining"

python3 - "$W" <<'EOF'
import json, random, sys
w = sys.argv[1]; rng = random.Random(0)
words = "the court held that the appellant was entitled to relief under the act and the petition is allowed".split()
cites = ["AIR 1984 SC 1728", "(2007) 1 SCC 683", "Section 302 of the Indian Penal Code", "Article 14"]
def doc(n): return " ".join(rng.choice(words) if rng.random() > 0.05 else rng.choice(cites) for _ in range(n))
for kind in ["acts", "high_courts", "supreme_court", "tribunals"]:
    json.dump([{"id": i, "text": doc(800)} for i in range(60)], open(f"{w}/train_{kind}.json", "w"))
json.dump({str(i): {"text": doc(300)} for i in range(5)}, open(f"{w}/test.json", "w"))
EOF

python3 "$P1/prepare_data.py" --files "$W"/train_*.json --out_dir "$W/data" --train_tokenizer --vocab_size 400 \
    --tokenizer_docs 100 --citation_types --eval_pairs 4 --prompt_words 200 --target_words 32 \
    --sc_val_frac 0.2 --other_val_frac 0.2 --val_tokens 4e4

for arch in legallm gpt2 swa_lite jurisnet encdec; do
python3 - "$W" "$arch" <<'EOF'
import json, sys
w, arch = sys.argv[1], sys.argv[2]
m = dict(vocab_size=400, n_layer=2, n_head=2, d_model=32, d_ff=64, max_seq_len=128)
if arch != "legallm": m["arch"] = arch
if arch == "swa_lite": m["window"] = 32
if arch == "encdec":
    m.pop("n_layer"); m.update(n_enc_layer=2, n_dec_layer=2, split_choices=[32, 64])
cfg = dict(model=m, train_bin=f"{w}/data/train.bin", val_bin=f"{w}/data/val.bin", tokenizer=f"{w}/data/tokenizer.json",
           out_dir=f"{w}/run_{arch}", seed=42, micro_batch=4, grad_accum=2, peak_lr=1e-3, min_lr_ratio=0.1,
           warmup_steps=5, decay_fraction=0.2, weight_decay=0.1, compile=False, train_hours=0.004,
           eval_every_sec=5, ckpt_every_sec=7, log_every=10, eval_batches=1, allow_any_size=True)
if arch == "jurisnet":
    cfg.update(train_types_bin=f"{w}/data/train_types.bin", val_types_bin=f"{w}/data/val_types.bin")
json.dump(cfg, open(f"{w}/cfg_{arch}.json", "w"))
EOF
python3 "$P1/pretrain.py" "$W/cfg_$arch.json" | grep -E "parameters|finished"
python3 "$P1/generate.py" --model_dir "$W/run_$arch/final" --test "$W/test.json" --eval_pairs "$W/data/eval_pairs.json" \
    --decodings greedy,greedy_nr4,beam2_nr4 --decoding greedy_nr4 --target_words 32 --max_new 48 --out_dir "$W/out_$arch" \
    | grep -E "wrote|greedy_nr4 "
python3 -c "import zipfile,json,sys; z=zipfile.ZipFile(sys.argv[1]); p=json.loads(z.read('prediction.json')); assert len(p)==5 and all(len(v['text'].split())==32 for v in p.values()), p" "$W/out_$arch/submission.zip"
done
echo "E2E OK"
