# ToolTune

Local Qwen3-4B-Instruct-2507 quantized LoRA tool-calling experiment on Apple M5 / 16 GB.

Work follows the existing detailed T1–T4 plan. T1 is a **synthetic resource and interface check**, not Glaive quality evaluation. T2 data processing, T3 formal training and T4 frozen Test comparison are separate acceptance stages. No fine-tuning improvement is claimed before the fair comparison exists.

Current status: **T1 and T2 passed**. Active evaluation protocol: `v2-balanced`, approved before model evaluation; Validation 100 calls + 100 noncalls, Test 95 + 95. Training retains the same 4,000 examples. Original v1 is preserved. T3/T4 have not started. Details: `reports/T1-resource-smoke.md`, `reports/T2-data-protocol.md`, `reports/T2-balanced-protocol.md`.

## Environment and T1 reproduction

Python 3.12.15. Existing Python/uv executables are read only; packages are installed exclusively in this project's `.venv`. The interpreter remains dependent on the existing Python installation under RetailFlow `.tools/python` (not its site-packages); do not remove that interpreter while using this environment.

```sh
cd /Users/yyy/code/ToolTune
/Users/yyy/code/RetailFlow/.tools/uv/bin/uv sync --locked --python /Users/yyy/code/RetailFlow/.tools/python/cpython-3.12-macos-aarch64-none/bin/python3.12 --cache-dir /Users/yyy/code/ToolTune/.cache/uv
.venv/bin/python -m pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u scripts/smoke.py
```

The smoke runner refuses to overwrite an existing `artifacts/t1-smoke` or `adapters/t1-smoke` directory. Preserve completed/failed runs under distinct names before an explicitly authorized rerun. A smoke run does not use or select on Glaive Validation/Test.

Model snapshot location: `artifacts/model_path.txt`. Restore this path from the pinned revision in `UPSTREAM.md` when reproducing on another computer. Model file hashes, source records and actual environment versions are stored under `artifacts/`; trained deltas under `adapters/`. Source repository: [snowstorm-121/ToolTune](https://github.com/snowstorm-121/ToolTune). No model weights or adapters are published in this repository.

## What this repository includes

Code, locked dependencies, configs, upstream licenses, source/split ID manifests and T1/T2 reports are versioned. Small measured summaries are copied verbatim into `reports/evidence/`; full local logs and tensors remain local. T1 is a synthetic smoke experiment, not a trained model quality result.

Model weights, raw/derived Glaive data, adapters, `.venv`, caches and `artifacts/` are excluded from Git. A fresh clone therefore does not include the assets required for model/template integration tests or full hash-contract validation. Obtain the pinned upstream model/data separately and restore the local asset manifests before running those commands. The absolute interpreter paths above document the original machine; use your own Python 3.12 and uv paths on another machine. The committed contracts describe the measured local experiment and are not a claim that a fresh clone alone passes full validation.

The data/grouping/sampling/contract unit tests can run independently of downloaded model weights after dependency installation on a supported Apple Silicon machine:

```sh
.venv/bin/python -m pytest -q tests/test_data.py tests/test_sampling.py tests/test_contract.py
```

## Evidence

- T1: `artifacts/t1-smoke/` — full synthetic prompts, targets and masks, raw base/adapter generations, training metrics, system memory samples, tensor hashes and reload check.
- Source and adaptation details: `UPSTREAM.md`.
- Dependencies: `pyproject.toml`, `uv.lock`.
- Limits: synthetic examples diagnose execution, not tool-use generalization. MLX allocator peak is not whole-system peak; system pressure and swap are sampled separately.

## T2 commands and data

```sh
cd /Users/yyy/code/ToolTune
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/prepare_balanced.py --validate
```

Active data/config: `data/derived/v2-balanced/`, `data/manifests/v2-balanced/`, `configs/evaluation-balanced.json`. This is a balanced diagnostic distribution, not original population accuracy. Held-out call examples each have only one candidate tool. The reviewed unavailable-tools and clarification subsets remain undefined.

The original deterministic preparation command is `PYTHONPATH=src .venv/bin/python -m tooltune.cli prepare`; it refuses to overwrite `data/derived/v1` or `data/manifests/v1`. Parent validation: `PYTHONPATH=src .venv/bin/python -m tooltune.cli validate`. Independent parent reproducibility: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/verify_t2.py`. After v1 preparation, `scripts/prepare_balanced.py` generates v2, independently reparsing held-out sources in reverse order to verify identical output. It refuses to overwrite v2. These commands use already downloaded local inputs and do not run model evaluation.

- `data/derived/v1/{train,valid,test}.jsonl`: upstream tools/messages payloads.
- `data/derived/v1/{train,valid,test}.records.jsonl`: scoring metadata and traceable IDs, never passed wholesale as model prompts.
- `data/derived/v1/quarantine_*.jsonl`: rejected source/decision IDs and reasons; no label repair.
- `data/derived/v1/smoke/`: Train-only 200+20 examples for later pipeline diagnostics; `valid.jsonl` here is a Train-side check, not formal Validation.
- `data/manifests/v1/`: complete source-group assignments, selected IDs and frozen hash contract.

Scope remains single assistant decision. Binary no_call scoring does not establish natural-language correctness, source truth or successful execution. There is no explicit clarification gold subset or reviewed unavailable-tools subset yet. Generated historical API responses are not live API results.
