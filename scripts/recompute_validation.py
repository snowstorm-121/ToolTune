"""Recompute T3 metrics from first raw predictions without loading MLX/model weights."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tooltune.data import digest
from tooltune.evaluate import generation_input, read_completed, recompute
from tooltune.metrics import select_checkpoint


def main():
    run=ROOT/'artifacts/t3-r8-v1'
    records=[json.loads(l) for l in (ROOT/'data/derived/v2-balanced/valid.records.jsonl').read_text().splitlines()]
    by_id={r['sample_id']:r for r in records}
    environment=json.loads((run/'environment_contract.json').read_text())
    selection=json.loads((run/'selection.json').read_text())
    assert select_checkpoint(selection['candidates'])==selection['selected']
    results={};all_scores={};all_raw={}
    for label,path in [('B0',run/'baseline/validation'),('U100',run/'checkpoint-0100'),('U200',run/'checkpoint-0200')]:
        contract=json.loads((path/'contract.json').read_text())
        assert contract['environment']==environment and contract['records_sha256']==digest(records)
        raw=read_completed(path,contract,list(by_id))
        scores,summary=recompute(records,raw)
        saved=json.loads((path/'summary.json').read_text())
        assert all(saved[k]==v for k,v in summary.items())
        assert scores==[json.loads(l) for l in (path/'scores.jsonl').read_text().splitlines()]
        for sid,p in raw.items():
            assert p['input']==generation_input(by_id[sid])
            if label!='B0':assert p['prompt_ids']==all_raw['B0'][sid]['prompt_ids']
        results[label]=summary;all_raw[label]=raw;all_scores[label]={s['sample_id']:s for s in scores}
    selected_label='U'+str(selection['selected']['updates'])
    transitions={};cases=[];seen=set()
    for kind in ('all','call','no_call'):
        counts={'both_correct':0,'improved':0,'regressed':0,'both_wrong':0}
        for sid,e in by_id.items():
            if kind!='all' and e['expected_action']!=kind:continue
            before=all_scores['B0'][sid]['strict_match'];after=all_scores[selected_label][sid]['strict_match']
            key='both_correct' if before and after else 'improved' if after else 'regressed' if before else 'both_wrong'
            counts[key]+=1
        transitions[kind]=counts
    for label in (selected_label,'B0'):
        for sid,s in all_scores[label].items():
            reason=s['failure_reason']
            if not reason or reason in seen:continue
            seen.add(reason);e=by_id[sid]
            cases.append({'run':label,'sample_id':sid,'source_record_id':e['source_record_id'],
                'reason':reason,'last_user':next(m['content'] for m in reversed(e['messages']) if m['role']=='user'),
                'expected_action':e['expected_action'],'expected_tool':e['expected_tool'],
                'expected_arguments':e['expected_arguments'],'prediction':s['prediction'],
                'output':all_raw[label][sid]['output'],'stop_reason':all_raw[label][sid]['stop_reason']})
    report={'status':'passed','all_metrics_recomputed_match':True,'all_prompts_identical':True,
        'selected':selected_label,'validation_only':True,'test_used':False,
        'results':results,'paired_strict_transitions':transitions,'failure_cases':cases}
    (run/'recomputed_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('failure_cases','results')},indent=2))


if __name__=='__main__':main()
