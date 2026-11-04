# Measured evidence snapshots

These files contain measured T1–T4 evidence, including T3 checkpoint selection and the frozen T4 comparison. Model/data revisions and hashes are retained. Full local prediction logs, model weights, adapters, raw/derived dataset and environment remain excluded.

Run `python3 -m unittest discover -s tests -p test_public_evidence.py -v` from the repository root to check committed source hashes, ordered Test IDs, paired transitions and strict-match totals. This requires only Python's standard library and also runs in GitHub Actions. It checks the consistency of published evidence; it does not regenerate predictions or independently rescore outputs that are not published.

`model-manifest.json` records the single quantized base snapshot. `t1-smoke-summary.json` describes 10 synthetic optimizer updates, not a Glaive quality score. `t1-download-issue.json` preserves a real failed download incident. `t2-balanced-summary.json` and the leakage audit describe the active balanced diagnostic split. The v1 files retain the prior sampling and reproducibility audit. File paths refer to the original project layout.
