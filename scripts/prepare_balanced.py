"""Approved T2 amendment: balanced held-out decisions; no model inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tooltune.data import canonical,decision_examples,normalize_query,parse_conversation,query_shingles,tool_families
from tooltune.formatting import format_example,load_tokenizer
from tooltune.sampling import balanced_sample
from tooltune.train import training_dataset
from tooltune.validation import verify_contract

ROOT=Path(__file__).resolve().parents[1]
VERSION='v2-balanced'
OUT=ROOT/'data/derived'/VERSION
MANIFESTS=ROOT/'data/manifests'/VERSION


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def file_hash(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def rows_bytes(rows):
    return ''.join(canonical(r)+'\n' for r in rows).encode()


def audit(selected):
    result={};queries={}
    for split,rows in selected.items():
        queries[split]={normalize_query(m['content']) for e in rows for m in e['messages'] if m['role']=='user'}
    for i,a in enumerate(selected):
        for b in list(selected)[i+1:]:
            for name,func in [('sources',lambda r:{r['source_record_id']}),
                ('groups',lambda r:{r['group_id']}),('families',tool_families)]:
                left=set().union(*(func(r) for r in selected[a]))
                right=set().union(*(func(r) for r in selected[b]))
                result[f'{a}:{b}:{name}_overlap']=len(left&right)
                assert not left&right
            result[f'{a}:{b}:normalized_query_overlap']=len(queries[a]&queries[b])
            assert not queries[a]&queries[b]
            other={q:query_shingles(q) for q in queries[b] if query_shingles(q)}
            postings=defaultdict(set)
            for q,sh in other.items():
                for s in sh:postings[s].add(q)
            count=0
            for q in queries[a]:
                sh=query_shingles(q)
                if not sh:continue
                for candidate in set().union(*(postings[s] for s in sh)):
                    target=other[candidate]
                    if len(sh&target)/len(sh|target)>=.85:count+=1
            result[f'{a}:{b}:near_duplicate_pairs']=count
            assert count==0
    return result


def pool_from_source(split,raw,assignments,tokenizer):
    pool=[]
    for assignment in assignments:
        source_id=assignment['source_record_id']
        row=raw[int(source_id.rsplit(':',1)[1])]
        c=parse_conversation(row,source_id)
        c.update(group_id=assignment['group_id'],split=split)
        for e in decision_examples(c):
            try:f=format_example(e,tokenizer)
            except ValueError as exc:
                if not str(exc).startswith('long sample:'):raise
                continue
            e['input_tokens']=len(f['input_ids']);e['supervised_tokens']=len(f['target_ids'])
            pool.append(e)
    return pool


def verify():
    verify_contract(ROOT)  # Parent data/model/code must remain intact.
    contract=json.loads((MANIFESTS/'contract.json').read_text())
    for name,sha in contract['files'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError(f'contract mismatch: {name}')
    selected={s:read_rows(OUT/f'{s}.records.jsonl') for s in ('train','valid','test')}
    summary=json.loads((OUT/'summary.json').read_text())
    for split,rows in selected.items():
        ids=json.loads((MANIFESTS/f'{split}_ids.json').read_text())
        assert ids==[r['sample_id'] for r in rows]
        payload=[{'tools':r['tools'],'messages':r['messages']} for r in rows]
        assert rows_bytes(payload)==(OUT/f'{split}.jsonl').read_bytes()
        assert dict(Counter(r['expected_action'] for r in rows))==summary['splits'][split]['actions']
    assert Counter(r['expected_action'] for r in selected['valid'])=={'call':100,'no_call':100}
    assert Counter(r['expected_action'] for r in selected['test'])=={'call':95,'no_call':95}
    assert (OUT/'train.jsonl').read_bytes()==(ROOT/'data/derived/v1/train.jsonl').read_bytes()
    assert audit(selected)==json.loads((OUT/'leakage_audit.json').read_text())
    return {'status':'passed','version':VERSION,'verified_files':len(contract['files']),
            'counts':{s:len(rows) for s,rows in selected.items()},'train_unchanged':True,
            'model_evaluation_run':False}


def prepare():
    start=time.perf_counter();verify_contract(ROOT)
    if OUT.exists() or MANIFESTS.exists():raise FileExistsError('refusing to overwrite balanced version')
    source=json.loads((ROOT/'data/manifests/v1/source_splits.json').read_text())
    old_summary=json.loads((ROOT/'data/derived/v1/summary.json').read_text())
    config=json.loads((ROOT/'configs/evaluation-balanced.json').read_text())
    raw=json.loads((ROOT/'data/raw/glaive-v2/glaive-function-calling-v2.json').read_text())
    tokenizer=load_tokenizer((ROOT/'artifacts/model_path.txt').read_text().strip())
    selected={'train':read_rows(ROOT/'data/derived/v1/train.records.jsonl')}
    pools={};reproduction={}
    for split,n in [('valid',100),('test',95)]:
        pool=pool_from_source(split,raw,source['splits'][split],tokenizer)
        assert dict(Counter(r['expected_action'] for r in pool))==old_summary[split]['eligible_actions']
        selected[split]=balanced_sample(pool,n)
        # Independently reparse reversed source order; reproduce IDs AND full rendered payload bytes.
        repeated=pool_from_source(split,raw,list(reversed(source['splits'][split])),tokenizer)
        again=balanced_sample(repeated,n)
        assert rows_bytes(again)==rows_bytes(selected[split])
        pools[split]=dict(Counter(r['expected_action'] for r in pool))
        reproduction[split]={'reparsed_reversed_sources':True,'full_records_byte_identical':True}
        print(split,'selected',len(again),'actions',dict(Counter(r['expected_action'] for r in again)),flush=True)
    del raw
    leakage=audit(selected)
    OUT.mkdir(parents=True);MANIFESTS.mkdir(parents=True)
    summary={'version':VERSION,'parent_version':'v1','distribution':'balanced diagnostic; not population representative',
        'user_approved':'2026-10-22: Validation 100+100; Test 95+95; no split reassignment or fabricated labels',
        'candidate_pools':pools,'splits':{},'reproduction':reproduction,
        'train_unchanged':True,'model_evaluation_run':False,'seconds':time.perf_counter()-start}
    for split,rows in selected.items():
        if split=='train':
            for suffix in ('jsonl','records.jsonl'):
                (OUT/f'train.{suffix}').symlink_to(f'../v1/train.{suffix}')
        else:
            (OUT/f'{split}.records.jsonl').write_bytes(rows_bytes(rows))
            (OUT/f'{split}.jsonl').write_bytes(rows_bytes([{'tools':e['tools'],'messages':e['messages']} for e in rows]))
        write_json(MANIFESTS/f'{split}_ids.json',[e['sample_id'] for e in rows])
        dataset=training_dataset(rows,tokenizer)
        for i,e in enumerate(rows):
            f=format_example(e,tokenizer)
            tokens,offset=dataset[i]
            assert tokens==f['input_ids'] and offset==len(f['prompt_ids'])
        summary['splits'][split]={'decisions':len(rows),'actions':dict(Counter(e['expected_action'] for e in rows)),
            'source_conversations':len({e['source_record_id'] for e in rows}),
            'groups':len({e['group_id'] for e in rows}),
            'available_tool_counts':dict(Counter(len(e['tools']) for e in rows)),
            'supervised_tokens':sum(e['supervised_tokens'] for e in rows),
            'total_tokens':sum(e['input_tokens'] for e in rows)}
    (MANIFESTS/'source_splits.json').symlink_to('../v1/source_splits.json')
    write_json(OUT/'summary.json',summary);write_json(OUT/'leakage_audit.json',leakage)
    contract={'version':VERSION,'parent_contract_sha256':file_hash(ROOT/'data/manifests/v1/contract.json'),
        'files':dict(json.loads((ROOT/'data/manifests/v1/contract.json').read_text())['files'])}
    new_files=list(OUT.iterdir())+list(MANIFESTS.iterdir())+[
        ROOT/'data/manifests/v1/contract.json',ROOT/'configs/evaluation-balanced.json',
        ROOT/'src/tooltune/sampling.py',ROOT/'tests/test_sampling.py',Path(__file__).resolve()]
    for p in new_files:contract['files'][str(p.relative_to(ROOT))]=file_hash(p)
    write_json(MANIFESTS/'contract.json',contract)
    print(json.dumps(verify(),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    if args.validate:print(json.dumps(verify(),indent=2))
    else:prepare()
