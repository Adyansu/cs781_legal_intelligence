# Phase 2 · Legal NER

Full documentation: [main README › Phase 2](../README.md#phase-2-legal-named-entity-recognition).

**Run** `phase2_lner.ipynb` on Kaggle (GPU T4 ×2) with the Phase-1 model folder and `evaluate_phase2.py` attached. The Phase-2 data is read from `/kaggle/input` or downloaded from Hugging Face with the secret `capstoned`. Section 9 runs the experiments; each `run_experiment(name, method, **overrides)` writes `lner/<name>/submission_<name>.zip`, `metrics.json` and `val_predictions.json`.

| Experiment | Call |
|---|---|
| full / lora / lorma / student / kd | `run_experiment("lorma", "lorma")`, `run_experiment("kd", "kd", teacher="full")` |
| causal mask | `run_experiment("lorma_causal", "lorma", bidirectional=False)` |
| long / weighted CE | `run_experiment("lorma_wce", "lorma", epochs=40, patience=10, loss="weighted")` |
| **submitted** | `run_experiment("lorma_h256_wce", "lorma", epochs=40, patience=10, header=256, loss="weighted")` |

Also available: `partial` (top-k blocks), `adapter` (Pfeiffer bottleneck), `refit_all=True` (retrain on all 70 docs for the selected number of epochs), and `error_report(name)` (boundary / label / missed / spurious breakdown).
