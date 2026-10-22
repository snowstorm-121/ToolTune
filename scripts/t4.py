"""Freeze the T4 contract, then run each Test arm in a fresh process."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tooltune.data import digest
from tooltune.evaluate import run_evaluation
from tooltune.formatting import load_tokenizer
from t3 import Resources, rows, sha, write

RUN=ROOT/'artifacts/t4-test-v1'
DATA=ROOT/'data/derived/v2-balanced/test.records.jsonl'
FREEZE=RUN/'frozen_contract.json'


def contract():
    t3_run=ROOT/'artifacts/t3-r8-v1'
    prior=json.loads((t3_run/'environment_contract.json').read_text())
    for name,expected in prior['files'].items():
        if sha(ROOT/name)!=expected:raise ValueError('T3 environment file changed: '+name)
    if sha(ROOT/'data/manifests/v2-balanced/contract.json')!=prior['data_contract_sha256']:
        raise ValueError('T2 data contract changed')
    if sha(ROOT/'artifacts/model_manifest.json')!=prior['model_manifest_sha256']:
        raise ValueError('model manifest changed')
    packages={p:importlib.metadata.version(p) for p in prior['packages']}
    if packages!=prior['packages'] or sys.version!=prior['python']:
        raise ValueError('T3 runtime changed')
    selected=json.loads((t3_run/'selection.json').read_text())['selected']
    if selected['updates']!=200:raise ValueError('selected update changed')
    adapter=ROOT/selected['path']
    if sha(adapter/'adapters.safetensors')!=selected['sha256']:
        raise ValueError('selected adapter changed')
    adapter_manifest=json.loads((adapter/'experiment_manifest.json').read_text())
    if adapter_manifest['weights_sha256']!=selected['sha256'] or not adapter_manifest['selected']:
        raise ValueError('selected adapter manifest mismatch')
    manifest=json.loads((ROOT/'artifacts/model_manifest.json').read_text())
    model_path=Path((ROOT/'artifacts/model_path.txt').read_text().strip())
    for item in manifest['files']:
        if sha(model_path/item['name'])!=item['sha256']:
            raise ValueError('base model file changed: '+item['name'])
    data_contract=json.loads((ROOT/'data/manifests/v2-balanced/contract.json').read_text())
    if sha(DATA)!=data_contract['files']['data/derived/v2-balanced/test.records.jsonl']:
        raise ValueError('Test records changed')
    records=rows(DATA);ids=[r['sample_id'] for r in records]
    if len(ids)!=190 or len(set(ids))!=190 or Counter(r['expected_action'] for r in records)!={'call':95,'no_call':95}:
        raise ValueError('Test counts changed')
    if ids!=json.loads((ROOT/'data/manifests/v2-balanced/test_ids.json').read_text()):
        raise ValueError('Test IDs changed')
    config=json.loads((ROOT/'configs/evaluation-balanced.json').read_text())
    return {'experiment_id':'t4-test-v1','base_revision':manifest['revision'],
        'base_model_manifest_sha256':sha(ROOT/'artifacts/model_manifest.json'),
        'selected_adapter':selected['path'],'selected_adapter_sha256':selected['sha256'],
        't3_selection_sha256':sha(t3_run/'selection.json'),
        't3_environment_sha256':sha(t3_run/'environment_contract.json'),
        'source_sha256':{name:sha(ROOT/name) for name in
            ('scripts/t4.py','src/tooltune/evaluate.py','src/tooltune/metrics.py',
             'src/tooltune/data.py','src/tooltune/formatting.py','configs/chat_template.jinja',
             'configs/evaluation-balanced.json','uv.lock')},
        'test_records_sha256':sha(DATA),'test_ids_sha256':sha(ROOT/'data/manifests/v2-balanced/test_ids.json'),
        'test_record_digest':digest(records),'test_ids':ids,'test_counts':{'call':95,'no_call':95},
        'decode':config['decode'],'metric_protocol':'strict-native-tool-call-v1; JSON key order only; all truncation is error',
        'runtime':{'python':sys.version,'packages':packages},'selection_basis':'T3 Validation only'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','baseline','adapter'])
    stage=parser.parse_args().stage
    current=contract()
    if stage=='freeze':
        RUN.mkdir(parents=True,exist_ok=False)
        write(FREEZE,current)
        print(json.dumps({'status':'frozen','experiment_id':current['experiment_id'],
            'test_count':len(current['test_ids']),'selected_adapter':current['selected_adapter']},indent=2))
        return
    if json.loads(FREEZE.read_text())!=current:raise ValueError('T4 frozen contract mismatch')
    path=RUN/('baseline' if stage=='baseline' else 'adapter')
    records=rows(DATA)
    adapter=ROOT/current['selected_adapter'] if stage=='adapter' else None
    base_path=(ROOT/'artifacts/model_path.txt').read_text().strip()
    with Resources(path.parent/(path.name+'-system-memory.jsonl'),45) as resource:
        model,_=load(base_path,adapter_path=str(adapter) if adapter else None,trust_remote_code=False)
        keys={name for name,_ in tree_flatten(model.parameters()) if name.endswith(('.lora_a','.lora_b'))}
        if stage=='baseline' and keys:raise ValueError('baseline contains adapter tensors')
        if stage=='adapter' and len(keys)!=16:raise ValueError('selected adapter did not load completely')
        tokenizer=load_tokenizer(base_path)
        eval_contract={'frozen_contract_sha256':sha(FREEZE),'arm':stage,
            'adapter_sha256':current['selected_adapter_sha256'] if adapter else None,
            'sample_ids':current['test_ids'],'records_sha256':current['test_record_digest'],
            'decode':current['decode'],'metric_protocol':current['metric_protocol']}
        result=run_evaluation(model,tokenizer,records,path,eval_contract,resource.check)
    write(path/'run_result.json',{'summary':result,'resources':resource.summary(),
        'adapter_tensors_loaded':len(keys),'independent_process':True})
    print(json.dumps({'arm':stage,'strict':result['strict_decision_match'],
        'resource':result['resource']},indent=2))


if __name__=='__main__':main()
