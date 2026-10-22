"""Fresh-process T3 acceptance: reload serialized selected adapter, verify actual tensors."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import mlx.core as mx
from mlx_lm import load

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tooltune.evaluate import generation_input
from tooltune.formatting import load_tokenizer
from smoke import snapshot,tensor_hash


def main():
    run=ROOT/'artifacts/t3-r8-v1'
    selected=json.loads((run/'selection.json').read_text())['selected']
    adapter=ROOT/selected['path']
    if not (adapter/'adapters.safetensors').is_file():raise FileNotFoundError('selected adapter missing')
    with (adapter/'adapters.safetensors').open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==selected['sha256']
    training=json.loads((run/'formal/training_summary.json').read_text())
    metrics=[json.loads(l) for l in (run/'formal/metrics.jsonl').read_text().splitlines()]
    assert training['optimizer_updates']==200 and training['micro_steps']==800
    assert training['supervised_tokens_exposed']==metrics[-1]['trained_tokens']
    trainable=json.loads((run/'formal/trainable.json').read_text())
    weights=mx.load(str(adapter/'adapters.safetensors'))
    assert set(weights)==set(trainable)
    assert all(list(w.shape)==trainable[k] for k,w in weights.items())
    base_path=(ROOT/'artifacts/model_path.txt').read_text().strip()
    model,_=load(base_path,adapter_path=str(adapter),trust_remote_code=False)
    model.eval();actual=snapshot(model)
    before=json.loads((run/'formal/parameters_before.json').read_text())
    assert set(actual)==set(before)
    assert all(actual[k]==before[k] for k in actual if k not in trainable)
    assert all(actual[k]['sha256']==tensor_hash(w) for k,w in weights.items())
    tokenizer=load_tokenizer(base_path)
    # An observation-only Train probe, not a retried Validation/Test prediction.
    first=json.loads((ROOT/'data/derived/v2-balanced/train.records.jsonl').open().readline())
    inputs=generation_input(first)
    prompt=tokenizer.apply_chat_template(inputs['messages'],tools=inputs['tools'] or None,
                                         add_generation_prompt=True,return_dict=False)
    adapted=model(mx.array([prompt]))[:,-1,:].astype(mx.float32);mx.eval(adapted)
    del model,weights;gc.collect();mx.clear_cache()
    model,_=load(base_path,trust_remote_code=False);model.eval()
    plain=model(mx.array([prompt]))[:,-1,:].astype(mx.float32)
    difference=mx.max(mx.abs(adapted-plain)).item()
    assert difference>0
    del model;gc.collect();mx.clear_cache()
    report={'status':'passed','selected_adapter':selected['path'],
        'adapter_tensor_count':len(trainable),'all_adapter_tensors_reloaded_exactly':True,
        'all_frozen_base_tensors_match_pretraining_snapshot':True,
        'train_probe_sample_id':first['sample_id'],'train_probe_max_logit_change':difference,
        'test_model_evaluation_run':False,'checks':{}}
    for name,command in [('pytest',[sys.executable,'-m','pytest','-q']),
        ('frozen_T2',[sys.executable,'scripts/prepare_balanced.py','--validate']),
        ('recompute',[sys.executable,'scripts/recompute_validation.py'])]:
        result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),
                              capture_output=True,text=True)
        (run/f'final-{name}.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(name+' failed: '+result.stdout+result.stderr)
        report['checks'][name]={'exit_code':result.returncode,'stdout':result.stdout.strip()}
    report['time_unix']=time.time()
    (run/'final_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
