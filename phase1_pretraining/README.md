# Phase 1 · Legal LM pretraining

Full documentation: [main README › Phase 1](../README.md#phase-1-legal-lm-pretraining).

| File | Purpose |
|---|---|
| `legallm/model.py` | LegalLM, the submitted 109.53M decoder (RMSNorm, RoPE, SwiGLU, tied embeddings, KV cache) |
| `legallm/train.py` + `train_config.json` | Time-budgeted DDP training script and config of the submitted run (launched by the notebook with an absolute `deadline`) |
| `legallm/config.json`, `legallm/tokenizer.json` | Model config and the 32K byte-level BPE (weights are not in git: `pytorch_model.bin` is ≈440 MB) |
| `prepare_data.py` | Tokenizer training, streamed corpus → `train.bin` / `val.bin`, citation types, held-out eval pairs |
| `pretrain.py` + `configs/*.json` | The same recipe for any family: `legallm`, `gpt2_baseline`, `swa_lite`, `jurisnet`, `encdec_hybrid` |
| `generate.py` | Decoding (greedy, `rep`, `nr`, `beam`, sampling) → `submission.zip`; local BERTScore comparison |
| `notebooks/phase1_legallm_v2.ipynb` | Kaggle: warm-restart training + decoding search (repetition-penalty sweep, BERTScore-MBR) |

```bash
python prepare_data.py --hf --out_dir /tmp/cs781_data --train_tokenizer --vocab_size 32000 \
    --caps acts=60e6,high_court=800e6,supreme_court=300e6,tribunal=300e6 --eval_pairs 64
python prepare_data.py --hf --out_dir /tmp/gpt2_data --tokenizer gpt2 --only supreme_court \
    --caps supreme_court=200e6 --val_tokens 5e6 --eval_pairs 64                 # GPT-2 baseline data
python prepare_data.py --hf --out_dir /tmp/juris_data --citation_types        # JurisNetLM data (+ type memmaps)
torchrun --standalone --nproc_per_node=2 pretrain.py configs/<family>.json      # edit paths / train_hours first
python generate.py --model_dir <out_dir>/final --test test.json --decoding greedy_nr4 --out_dir out/
```

Every exported model folder (`final/`) holds `config.json` (with `arch`), `pytorch_model.bin`, `tokenizer.json` and `training_state.json`, and loads with `backbones.load_backbone`.
