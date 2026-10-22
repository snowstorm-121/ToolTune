# Measured evidence snapshots

These JSON files are verbatim copies of local T1/T2 artifacts at first publication, not new experiments. Model/data revisions and hashes are retained. Full local raw logs, model weights, synthetic smoke adapter, raw/derived dataset and environment remain excluded. T3 formal fine-tuning and T4 fair model comparison have not run.

`model-manifest.json` records the single quantized base snapshot. `t1-smoke-summary.json` describes 10 synthetic optimizer updates, not a Glaive quality score. `t1-download-issue.json` preserves a real failed download incident. `t2-balanced-summary.json` and the leakage audit describe the active balanced diagnostic split. The v1 files retain the prior sampling and reproducibility audit. File paths refer to the original project layout.
