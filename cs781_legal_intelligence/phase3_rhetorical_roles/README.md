# Phase 3 · Rhetorical role prediction

Full documentation: [main README › Phase 3](../README.md#phase-3-rhetorical-role-prediction).

`rr_data.py` handles data discovery, format detection and validation. `rr_train.py` covers the model, training, official-evaluator selection and the submission; it runs one experiment per call and auto-resumes from `last.pt`. `phase3_rr.ipynb` writes both files, writes `/kaggle/working/rr_paths.json` (data, Phase-1 model, evaluator) and runs the rounds below in parallel on two GPUs.

```bash
python rr_train.py --name v2_ctx_full  --method full
python rr_train.py --name v2_sent_full --method full --hp '{"window": 1, "context": "none", "pos_buckets": 0, "domain_emb": false, "batch_size": 32, "eval_batch_size": 128}'
python rr_train.py --name v2_ctx_lorma --method lorma            # submitted
python rr_train.py --name v2_ctx_rslora --method rslora
python rr_train.py --name v2_ctx_ia3   --method ia3
python rr_train.py --name v2_ctx_kd6   --method kd --hp '{"teacher": "v2_ctx_full"}'
python rr_train.py --name smoke_full   --method full --smoke     # 2 docs per domain, a few steps
```

Outputs per run in `rr/<name>/`: `best.pt`, `metrics.json`, `dev_report_best.json` (per class, per domain, confusion matrix), `dev_predictions_best.json` and `submission_<name>.zip`. The method `frozen` (linear-probe control) is also available.
