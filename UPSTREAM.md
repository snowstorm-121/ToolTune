# Sources and scope

- Framework: https://github.com/ml-explore/mlx-lm ; installed `mlx-lm==0.32.2`, `mlx==0.32.4` (Apple, MIT). Exact dependency graph: `uv.lock`.
- Quantized base: https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit ; revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. Community conversion of Qwen's model, not an official Qwen weight release. 4-bit, group size 64. No conversion from Ollama GGUF.
- Official model and template: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 ; revision `cdbee75f17c01a7cc42f958dc650907174af0554` (Apache-2.0). `configs/chat_template.jinja` is extracted verbatim from this revision's `tokenizer_config.json`. The community snapshot contains an outdated thinking template; the project overrides it explicitly for both training and inference, without editing the snapshot. Original files are retained in `artifacts/upstream/`.
- Dataset (T2): https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 ; revision `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` (Apache-2.0). Dataset availability does not imply external APIs have been executed.

Project code is a thin orchestration/data/validation layer. It does not implement MLX quantization, optimizers, model architecture or automatic differentiation. Engineering was assisted by Codex; interview claims must reflect actual understanding and measured artifacts.

## Local compatibility changes

1. Use the pinned official Instruct template, as described above.
2. Supply `assistant_loss` to the upstream trainer through its supported loss callback. MLX-LM 0.32.2's default loss uses `positions <= length`, including one padding target when padding exists. Our bound is `positions < length`. No installed source is modified. Tests pin both observed upstream behavior and the corrected target selection.
3. Upstream training automatically calls `maybe_set_recommended_wired_limit`. The T1 and T3 runners temporarily replace only that call with a no-op for the duration of training. Inference uses upstream `generate_step` directly to avoid the higher-level wired-limit context. This changes no system or process allocator limits.
4. Explicit LoRA modules: last four transformer blocks, attention q/v projections, rank 8, MLX `scale=16` (a multiplier, not a PEFT `alpha` field). Actual tensor names/counts are saved, not inferred from defaults.

## Licenses

The official model license is preserved at `LICENSES/Qwen-Apache-2.0.txt`. Installed dependency license metadata and original model/data cards remain with their respective distributions/snapshots. No permission to publish weights or data is implied by this local project.
