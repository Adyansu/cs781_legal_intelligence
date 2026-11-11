# Phase 4 · Court judgment prediction

Full documentation: [main README › Phase 4](../README.md#phase-4-court-judgment-prediction-cjpe).

**Run** `phase4_cjpe.ipynb` top to bottom on Kaggle (GPU T4 ×2). Attach the phase data and the Phase-1 model folder (`pytorch_model.bin`, `config.json`, `tokenizer.json`, `model.py`); the notebook finds them anywhere under `/kaggle/input`. All experiments are listed in `EXPS` at the top. Each one gets a minute budget, and a wall-clock guard (`total_hours=4.6`) skips optional runs that would not fit.

- Outputs go to `/kaggle/working`: one `<run>.zip` per experiment (with `prediction.json` at its root), `results.csv` (score, parameters, memory, time per run) and dev-side reports.
- `CJPE_SMOKE=1` runs a tiny dry run (a few documents, about 1 minute per experiment).
- To use another Phase-1 family, replace `fresh_lm()`'s `LegalLM(...)` with `backbones.build_backbone(cfg)`. The classes share the attribute names that this notebook uses.
