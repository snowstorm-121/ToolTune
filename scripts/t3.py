"""T3-only offline validation and quantized LoRA; Test is never loaded."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import yaml
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
from mlx_lm.tuner.utils import linear_to_lora_layers

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tooltune.data import digest
from tooltune.evaluate import recompute, run_evaluation
from tooltune.formatting import TEMPLATE, load_tokenizer
from tooltune.metrics import select_checkpoint
from tooltune.train import assistant_loss, training_dataset
from smoke import snapshot

CONFIG=ROOT/'configs/train.yaml'
RUN=ROOT/'artifacts/t3-r8-v1'
ADAPTERS=ROOT/'adapters/t3-r8-v1'
DATA=ROOT/'data/derived/v2-balanced'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


class Resources:
    def __init__(self,path,minutes):
        self.path=path;self.minutes=minutes;self.stop=threading.Event();self.problem=None
        self.samples=[];self.start=time.monotonic()
    def __enter__(self):
        def monitor():
            red=0
            with self.path.open('a') as log:
                while not self.stop.is_set():
                    row={'time':time.time()}
                    for name,command in [('pressure',['sysctl','kern.memorystatus_vm_pressure_level']),
                        ('swap',['sysctl','vm.swapusage']),('memory',['memory_pressure','-Q'])]:
                        try:
                            r=subprocess.run(command,capture_output=True,text=True,timeout=5)
                            row[name]=r.stdout.strip() or r.stderr.strip()
                        except subprocess.TimeoutExpired:
                            row[name]='probe timed out';self.problem='system resource probe timed out'
                    red=red+1 if row['pressure'].endswith(': 4') else 0
                    if red>=3:self.problem='persistent red memory pressure'
                    self.samples.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    self.stop.wait(2)
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start();return self
    def check(self):
        if self.problem:raise RuntimeError(self.problem)
        if time.monotonic()-self.start>self.minutes*60:raise RuntimeError('wall time ceiling reached')
    def __exit__(self,*_):
        self.stop.set();self.thread.join(timeout=6)
    def summary(self):
        swaps=[]
        for row in self.samples:
            m=re.search(r'used = ([\d.]+)([MG])',row['swap'])
            if m:swaps.append(float(m[1])*(1024 if m[2]=='G' else 1))
        return {'sample_count':len(self.samples),'pressure':dict(Counter(r['pressure'] for r in self.samples)),
            'swap_MiB':{'first':swaps[0],'last':swaps[-1],'max':max(swaps)} if swaps else None,
            'problem':self.problem,'mlx_peak_bytes':mx.get_peak_memory()}


def environment_contract():
    files=[CONFIG,ROOT/'configs/evaluation-balanced.json',TEMPLATE,Path(__file__),
           ROOT/'src/tooltune/evaluate.py',ROOT/'src/tooltune/metrics.py']
    return {'files':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'data_contract_sha256':sha(ROOT/'data/manifests/v2-balanced/contract.json'),
        'model_manifest_sha256':sha(ROOT/'artifacts/model_manifest.json'),
        'packages':{p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy','transformers','jsonschema')},
        'python':sys.version}


def eval_contract(records,adapter=None):
    return {'environment':environment_contract(),'sample_ids':[r['sample_id'] for r in records],
        'records_sha256':digest(records),'adapter':adapter,
        'decode':json.loads((ROOT/'configs/evaluation-balanced.json').read_text())['decode'],
        'metric_protocol':'strict-native-tool-call-v1; JSON key order only; all truncation is error'}


def fixed_loss(model,examples,tokenizer,resources):
    state=np.random.get_state()
    try:
        value=evaluate(model,training_dataset(examples,tokenizer),1,-1,512,
                       loss=assistant_loss,progress_callback=resources.check)
        if not math.isfinite(value):raise RuntimeError('non-finite evaluation loss')
        return value
    finally:np.random.set_state(state)


def load_base():
    path=(ROOT/'artifacts/model_path.txt').read_text().strip()
    model,_=load(path,trust_remote_code=False)
    assert not any(k.endswith(('.lora_a','.lora_b')) for k,_ in tree_flatten(model.parameters()))
    return model,load_tokenizer(path)


def new_lora(model,config):
    mx.random.seed(config['seed']);np.random.seed(config['seed'])
    model.freeze();linear_to_lora_layers(model,config['num_layers'],config['lora_parameters'])
    trainable=dict(tree_flatten(model.trainable_parameters()))
    assert trainable and all(k.endswith(('.lora_a','.lora_b')) for k in trainable)
    assert all(32<=int(k.split('.')[2])<=35 for k in trainable)
    return {k:list(v.shape) for k,v in trainable.items()}


def train_run(model,tokenizer,examples,config,updates,path,resources,on_checkpoint=None):
    path.mkdir(parents=True,exist_ok=False)
    trainable=new_lora(model,config);before=snapshot(model)
    write(path/'trainable.json',trainable);write(path/'parameters_before.json',before)
    # The native iterator visits cached items only when a minibatch is yielded.
    dataset=training_dataset(examples,tokenizer)
    class Traced(type(dataset)):
        def __getitem__(self,index):
            result=super().__getitem__(index)
            with (path/'exposures.jsonl').open('a') as log:
                log.write(json.dumps({'sample_id':examples[index]['sample_id'],
                    'input_tokens':len(result[0]),'supervised_tokens':len(result[0])-result[1]})+'\n')
            return result
    dataset.__class__=Traced
    class Callback:
        def on_train_loss_report(self,info):
            resources.check()
            with (path/'metrics.jsonl').open('a') as log:log.write(json.dumps(info)+'\n')
            if not math.isfinite(info['train_loss']):raise RuntimeError('non-finite training loss')
            update=info['iteration']//config['grad_accumulation_steps']
            if on_checkpoint and update in config['checkpoint_updates']:
                on_checkpoint(update,model)
                model.train()
        def on_val_loss_report(self,info):raise AssertionError('validation is explicitly scheduled after optimizer update')
    microsteps=updates*config['grad_accumulation_steps']
    args=TrainingArgs(batch_size=1,iters=microsteps,max_seq_length=512,grad_checkpoint=True,
        grad_accumulation_steps=config['grad_accumulation_steps'],steps_per_report=4,
        steps_per_save=100000,adapter_file=str(path/'adapters.safetensors'))
    tic=time.perf_counter()
    with patch('mlx_lm.tuner.trainer.maybe_set_recommended_wired_limit',return_value=None):
        train(model,optim.Adam(config['learning_rate']),dataset,args=args,
              loss=assistant_loss,training_callback=Callback())
    after=snapshot(model);write(path/'parameters_after.json',after)
    changed=[k for k in before if before[k]!=after[k]]
    assert changed and all(k in trainable for k in changed)
    assert all(before[k]==after[k] for k in before if k not in trainable)
    trace=rows(path/'exposures.jsonl');assert len(trace)==microsteps
    summary={'optimizer_updates':updates,'micro_steps':microsteps,
        'effective_batch':4,'unique_examples_exposed':len({r['sample_id'] for r in trace}),
        'supervised_tokens_exposed':sum(r['supervised_tokens'] for r in trace),
        'input_tokens_exposed':sum(r['input_tokens'] for r in trace),
        'trainable_parameters':sum(math.prod(v) for v in trainable.values()),
        'frozen_base_unchanged':True,'changed_adapter_tensors':changed,
        'wall_seconds_including_checkpoints':time.perf_counter()-tic,'mlx_peak_bytes':mx.get_peak_memory()}
    write(path/'training_summary.json',summary)
    return summary


def preflight(config):
    path=RUN/'preflight';path.mkdir(parents=True,exist_ok=False)
    examples=rows(DATA/'train.records.jsonl')[:config['diagnostic_examples']]
    # Resource probe uses scored Train-side records, not formal Validation or Test.
    train_ids=json.loads((ROOT/'data/derived/v1/smoke/manifest.json').read_text())['check_sample_ids']
    by_id={r['sample_id']:r for r in rows(DATA/'train.records.jsonl')}
    candidates=[by_id[s] for s in train_ids]
    probe=[]
    for action in ('call','no_call'):probe.extend([r for r in candidates if r['expected_action']==action][:5])
    with Resources(path/'system_memory.jsonl',10) as resource:
        model,tokenizer=load_base()
        generated=run_evaluation(model,tokenizer,probe,path/'train-side-probe',eval_contract(probe),resource.check)
        before=fixed_loss(model,examples,tokenizer,resource)
        summary=train_run(model,tokenizer,examples,config,config['diagnostic_updates'],
                          path/'training',resource)
        after=fixed_loss(model,examples,tokenizer,resource)
        assert after<before, 'fixed training-side diagnostic loss did not decrease'
        metrics=rows(path/'training/metrics.jsonl')
        steady_rate=sum(m['iterations_per_second'] for m in metrics[1:])/len(metrics[1:])
        per_prediction=generated['resource']['generation_seconds']/len(probe)
        budget={'formal_optimizer_updates':config['optimizer_updates'],'formal_microsteps':800,
            'training_seconds_point_estimate':800/steady_rate,
            'three_validation_generation_seconds_point_estimate':600*per_prediction,
            'note':'short resource probe; compilation, sequence lengths and thermal throttling may change elapsed time',
            'additional_disk_budget_MiB':512,'formal_wall_limit_minutes':config['formal_wall_limit_minutes']}
        result={'status':'passed','train_side_only':True,'examples':len(examples),
            'fixed_loss_before':before,'fixed_loss_after':after,'training':summary,
            'resources':resource.summary(),'budget':budget}
        write(path/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


def baseline(config):
    valid=rows(DATA/'valid.records.jsonl');path=RUN/'baseline'
    if not path.exists():path.mkdir(parents=True)
    with Resources(path/'system_memory.jsonl',60) as resource:
        model,tokenizer=load_base()
        value=fixed_loss(model,valid,tokenizer,resource)
        summary=run_evaluation(model,tokenizer,valid,path/'validation',eval_contract(valid),resource.check)
        write(path/'result.json',{'summary':summary,'validation_loss':value,'resources':resource.summary(),
                                 'base_has_no_adapter':True})
        print('BASELINE',json.dumps(summary),flush=True)


def formal(config):
    pre=json.loads((RUN/'preflight/summary.json').read_text());assert pre['status']=='passed'
    assert (RUN/'baseline/result.json').is_file()
    if (RUN/'formal').exists():raise FileExistsError('formal training cannot silently restart or resume optimizer state')
    valid=rows(DATA/'valid.records.jsonl');examples=rows(DATA/'train.records.jsonl')
    ADAPTERS.mkdir(parents=True,exist_ok=False)
    candidates=[]
    with Resources(RUN/'formal-system-memory.jsonl',config['formal_wall_limit_minutes']) as resource:
        model,tokenizer=load_base()
        def checkpoint(update,model):
            dest=ADAPTERS/f'update-{update:04d}';dest.mkdir()
            write(dest/'adapter_config.json',{'fine_tune_type':'lora','num_layers':config['num_layers'],
                                             'lora_parameters':config['lora_parameters']})
            mx.save_safetensors(str(dest/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
            adapter={'updates':update,'path':str(dest.relative_to(ROOT)),'sha256':sha(dest/'adapters.safetensors')}
            value=fixed_loss(model,valid,tokenizer,resource)
            result=run_evaluation(model,tokenizer,valid,RUN/f'checkpoint-{update:04d}',
                eval_contract(valid,adapter),resource.check)
            candidates.append(dict(adapter,summary=result,validation_loss=value))
            write(RUN/'candidates.json',candidates)
            print('CHECKPOINT',update,'strict',result['strict_decision_match'],'loss',value,flush=True)
        training=train_run(model,tokenizer,examples,config,config['optimizer_updates'],
                           RUN/'formal',resource,checkpoint)
        chosen=select_checkpoint(candidates)
        write(RUN/'selection.json',{'selected':chosen,'candidates':candidates,
            'rule':'Validation strict match; reviewed unavailable-tools false-call rate if defined; earlier update',
            'test_used':False,'training':training,'resources':resource.summary()})
        print('SELECTED',chosen['path'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','baseline','train'])
    args=parser.parse_args();config=yaml.safe_load(CONFIG.read_text())
    RUN.mkdir(parents=True,exist_ok=True)
    contract=environment_contract();frozen=RUN/'environment_contract.json'
    if frozen.exists():assert json.loads(frozen.read_text())==contract,'T3 source/environment changed; use a new experiment ID'
    else:write(frozen,contract)
    try:
        {'preflight':preflight,'baseline':baseline,'train':formal}[args.stage](config)
    except BaseException as exc:
        write(RUN/f'{args.stage}-failure-{time.time_ns()}.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':main()
