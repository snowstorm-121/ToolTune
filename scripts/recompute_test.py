"""Independently rescore both first-pass frozen Test prediction logs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tooltune.data import digest
from tooltune.evaluate import generation_input, read_completed, recompute

RUN=ROOT/'artifacts/t4-test-v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    frozen=json.loads((RUN/'frozen_contract.json').read_text())
    for name,expected in frozen['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('frozen source changed: '+name)
    data=ROOT/'data/derived/v2-balanced/test.records.jsonl'
    if sha(data)!=frozen['test_records_sha256']:raise ValueError('Test records changed')
    records=[json.loads(line) for line in data.read_text().splitlines()]
    if [e['sample_id'] for e in records]!=frozen['test_ids'] or digest(records)!=frozen['test_record_digest']:
        raise ValueError('Test IDs or records changed')
    by_id={e['sample_id']:e for e in records}
    scored={};raw={};summaries={}
    for arm in ('baseline','adapter'):
        path=RUN/arm;contract=json.loads((path/'contract.json').read_text())
        if contract!={'frozen_contract_sha256':sha(RUN/'frozen_contract.json'),'arm':arm,
            'adapter_sha256':frozen['selected_adapter_sha256'] if arm=='adapter' else None,
            'sample_ids':frozen['test_ids'],'records_sha256':frozen['test_record_digest'],
            'decode':frozen['decode'],'metric_protocol':frozen['metric_protocol']}:
            raise ValueError('arm contract mismatch: '+arm)
        predictions=read_completed(path,contract,frozen['test_ids'])
        scores,summary=recompute(records,predictions)
        saved=json.loads((path/'summary.json').read_text())
        if any(summary[k]!=value for k,value in saved.items() if k!='resource'):
            raise ValueError('saved summary mismatch: '+arm)
        if scores!=[json.loads(line) for line in (path/'scores.jsonl').read_text().splitlines()]:
            raise ValueError('saved scores mismatch: '+arm)
        for sid,p in predictions.items():
            if p['input']!=generation_input(by_id[sid]):raise ValueError('prompt metadata mismatch: '+sid)
            if arm=='adapter' and p['prompt_ids']!=raw['baseline'][sid]['prompt_ids']:
                raise ValueError('arm prompt token mismatch: '+sid)
        raw[arm]=predictions;scored[arm]={s['sample_id']:s for s in scores};summaries[arm]=saved
    transitions=Counter();pairs=[]
    for e in records:
        sid=e['sample_id'];before=scored['baseline'][sid];after=scored['adapter'][sid]
        change='both_correct' if before['strict_match'] and after['strict_match'] else \
            'improved' if after['strict_match'] else 'regressed' if before['strict_match'] else 'both_wrong'
        transitions[change]+=1
        pairs.append({'sample_id':sid,'source_record_id':e['source_record_id'],'group_id':e['group_id'],
            'expected_action':e['expected_action'],'expected_tool':e['expected_tool'],
            'transition':change,'baseline_strict':before['strict_match'],'adapter_strict':after['strict_match'],
            'baseline_failure':before['failure_reason'],'adapter_failure':after['failure_reason']})
    (RUN/'paired_transitions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pairs))
    report={'status':'passed','experiment_id':frozen['experiment_id'],
        'first_pass_predictions_per_arm':len(records),'all_metrics_recomputed_match':True,
        'all_prompt_tokens_identical':True,'transitions':dict(transitions),
        'results':summaries,'test_was_used_for_selection':False}
    (RUN/'recomputed_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
