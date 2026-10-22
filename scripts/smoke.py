"""T1 synthetic interface/training check. Not a quality benchmark."""
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.datasets import ChatDataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tooltune.formatting import TEMPLATE, format_example, load_tokenizer
from tooltune.train import assistant_loss, training_dataset

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def toy_examples():
    tool = {'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers.',
        'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'},
        'b': {'type': 'integer'}}, 'required': ['a', 'b'], 'additionalProperties': False}}}
    examples = []
    for i, (a, b) in enumerate([(2, 3), (10, -2), (7, 8)]):
        messages = [{'role': 'system', 'content': 'Use available tools when appropriate.'}]
        if i == 2:
            messages += [{'role': 'user', 'content': 'Add 1 and 1.'},
                {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function',
                 'function': {'name': 'add', 'arguments': {'a': 1, 'b': 1}}}]},
                {'role': 'tool', 'content': '{"result": 2}'},
                {'role': 'assistant', 'content': 'The result is 2.'}]
        messages += [{'role': 'user', 'content': f'Use the tool to add {a} and {b}.'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function',
             'function': {'name': 'add', 'arguments': {'a': a, 'b': b}}}]}]
        examples.append({'sample_id': f'toy-call-{i}', 'origin': 'handwritten_synthetic',
                         'tools': [tool], 'messages': messages})
    for name, user, target in [('greeting', 'Hello!', 'Hello!'),
        ('unavailable', 'Check the current weather in Paris.',
         'I cannot check live weather with the available addition tool.')]:
        examples.append({'sample_id': f'toy-{name}', 'origin': 'handwritten_synthetic',
            'tools': [tool], 'messages': [{'role': 'user', 'content': user},
                                        {'role': 'assistant', 'content': target}]})
    return examples


def tensor_hash(array):
    # Preserve raw storage, including bfloat16 and quantized uint32 weights.
    return hashlib.sha256(np.array(array.view(mx.uint8)).tobytes()).hexdigest()


def snapshot(model):
    return {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': tensor_hash(v)}
            for k, v in tree_flatten(model.parameters())}


def predictions(model, tokenizer, examples, config):
    rows = []
    model.eval()
    for e in examples:
        prompt = format_example(e, tokenizer)['prompt_ids']
        start = time.perf_counter()
        ids = []
        reason = 'length'
        # Lower-level upstream generation avoids automatic wired-limit adjustment.
        generator = generate_step(mx.array(prompt), model,
            max_tokens=config['max_output_tokens'], sampler=make_sampler(temp=0),
            prefill_step_size=256)
        for token, _ in generator:
            if token == tokenizer.eos_token_id:
                reason = 'eos'
                break
            ids.append(token)
        generator.close()
        rows.append({'sample_id': e['sample_id'], 'prompt_ids': prompt,
            'output_ids': ids, 'output': tokenizer.decode(ids), 'stop_reason': reason,
            'seconds': time.perf_counter()-start})
        print('generation', e['sample_id'], rows[-1]['output'], flush=True)
    return rows


def main():
    config = yaml.safe_load((ROOT/'configs/smoke.yaml').read_text())
    run = ROOT/'artifacts/t1-smoke'
    run.mkdir(parents=True, exist_ok=False)
    adapters = ROOT/'adapters/t1-smoke'
    adapters.mkdir(parents=True, exist_ok=False)
    model_path = (ROOT/'artifacts/model_path.txt').read_text().strip()
    write(run/'config.json', config)
    write(run/'environment.json', {'python': sys.version, 'executable': sys.executable,
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'device': mx.device_info(), 'template_sha256': hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()})
    examples = toy_examples()
    tokenizer = load_tokenizer(model_path)
    write(run/'synthetic_examples.json', examples)
    formatted = [format_example(e, tokenizer) for e in examples]
    write(run/'mask_examples.json', [dict(sample_id=e['sample_id'], **f,
        prompt=tokenizer.decode(f['prompt_ids']), target=tokenizer.decode(f['target_ids']),
        tokens=[{'id': t, 'piece': tokenizer.decode([t]), 'loss_mask': m}
                for t, m in zip(f['input_ids'], f['loss_mask'])])
        for e, f in zip(examples, formatted)])
    long = dict(examples[0], messages=[{'role': 'user', 'content': 'long ' * 2000},
                                     examples[0]['messages'][-1]])
    try:
        format_example(long, tokenizer)
        raise AssertionError('long sample accepted')
    except ValueError as exc:
        write(run/'long_rejection.json', {'reason': str(exc), 'whole_sample_rejected': True})
    dataset = ChatDataset(examples, tokenizer, mask_prompt=True)
    for e, f in zip(examples, formatted):
        tokens, offset = dataset.process(e)
        assert tokens == f['input_ids'] and offset == len(f['prompt_ids'])
    stop = threading.Event()
    red = threading.Event()
    def monitor():
        with (run/'system_memory.jsonl').open('w') as log:
            while not stop.is_set():
                row = {'time': time.time()}
                for name, cmd in [('pressure', ['sysctl','kern.memorystatus_vm_pressure_level']),
                                  ('swap', ['sysctl','vm.swapusage']),
                                  ('memory', ['memory_pressure','-Q'])]:
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    row[name] = r.stdout.strip() or r.stderr.strip()
                if row['pressure'].endswith(': 4'): red.set()
                log.write(json.dumps(row)+'\n');log.flush()
                stop.wait(2)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        mx.random.seed(config['seed']); np.random.seed(config['seed'])
        model, _ = load(model_path, trust_remote_code=False)
        baseline = predictions(model, tokenizer, examples, config)
        write(run/'baseline_predictions.json', baseline)
        model.freeze()
        linear_to_lora_layers(model, config['num_layers'], config['lora_parameters'])
        before = snapshot(model)
        trainable = dict(tree_flatten(model.trainable_parameters()))
        assert trainable and all(k.endswith(('.lora_a', '.lora_b')) for k in trainable)
        assert all(32 <= int(k.split('.')[2]) <= 35 for k in trainable)
        write(run/'parameters_before.json', before)
        write(run/'trainable.json', {k: list(v.shape) for k,v in trainable.items()})
        adapter_config = {'fine_tune_type': 'lora', 'num_layers': config['num_layers'],
                          'lora_parameters': config['lora_parameters']}
        write(adapters/'adapter_config.json', adapter_config)
        class Callback:
            def on_train_loss_report(self, info):
                with (run/'train_metrics.jsonl').open('a') as log:
                    log.write(json.dumps(info)+'\n')
                if not math.isfinite(info['train_loss']): raise RuntimeError('non-finite loss')
                if red.is_set(): raise RuntimeError('red memory pressure observed')
            def on_val_loss_report(self, info):
                raise AssertionError('T1 has no validation/test benchmark')
        args = TrainingArgs(batch_size=1,
            iters=config['optimizer_updates']*config['grad_accumulation_steps'],
            grad_accumulation_steps=config['grad_accumulation_steps'],
            grad_checkpoint=True, max_seq_length=config['max_seq_length'],
            steps_per_report=4, steps_per_save=10000,
            adapter_file=str(adapters/'adapters.safetensors'))
        train_start = time.perf_counter()
        # Upstream otherwise changes the process wired limit. No system/allocator limits changed.
        with patch('mlx_lm.tuner.trainer.maybe_set_recommended_wired_limit', return_value=None):
            train(model, optim.Adam(config['learning_rate']), training_dataset(examples, tokenizer), args=args,
                  loss=assistant_loss, training_callback=Callback())
        train_seconds = time.perf_counter()-train_start
        after = snapshot(model)
        write(run/'parameters_after.json', after)
        changed = [k for k in before if before[k] != after[k]]
        assert changed and all(k in trainable for k in changed)
        assert all(before[k] == after[k] for k in before if k not in trainable)
        write(run/'trained_predictions.json', predictions(model, tokenizer, examples, config))
        probe = mx.array([formatted[0]['prompt_ids']])
        expected = model(probe)[:, -1, :]
        mx.eval(expected)
        peak = mx.get_peak_memory()
        del model, trainable
        gc.collect();mx.clear_cache()
        model, _ = load(model_path, adapter_path=str(adapters), trust_remote_code=False)
        actual = model(probe)[:, -1, :]
        diff = mx.max(mx.abs(expected.astype(mx.float32)-actual.astype(mx.float32))).item()
        assert diff < 1e-5
        reloaded = snapshot(model)
        assert reloaded == after
        summary = {'status': 'passed', 'synthetic_only': True,
            'optimizer_updates': config['optimizer_updates'], 'micro_steps': args.iters,
            'effective_batch': config['grad_accumulation_steps'],
            'trainable_parameters': sum(np.prod(v['shape']).item() for k,v in before.items()
                                         if k.endswith(('.lora_a','.lora_b'))),
            'changed_adapter_tensors': changed, 'frozen_base_unchanged': True,
            'reload_max_logit_difference': diff, 'reload_all_tensor_hashes_match': True,
            'training_seconds': train_seconds, 'mlx_peak_bytes_before_reload': peak,
            'mlx_peak_bytes_entire_run': mx.get_peak_memory(),
            'total_seconds': time.perf_counter()-start,
            'supervised_tokens_per_example': [sum(f['loss_mask']) for f in formatted]}
        write(run/'summary.json', summary)
        print(json.dumps(summary,indent=2), flush=True)
    except BaseException as exc:
        write(run/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                   'seconds': time.perf_counter()-start})
        raise
    finally:
        stop.set();thread.join(timeout=5)


if __name__ == '__main__':
    main()
