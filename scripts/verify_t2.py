"""Reparse/regroup source and byte-reproduce frozen selections without model inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tooltune.data import DataError,canonical,decision_examples,digest,parse_conversation,split_groups
from tooltune.formatting import format_example,load_tokenizer

ROOT=Path(__file__).resolve().parents[1]


def main():
    start=time.perf_counter();out=ROOT/'data/derived/v1';manifests=ROOT/'data/manifests/v1'
    contract=json.loads((manifests/'contract.json').read_text())
    changes={}
    for name,sha in contract['files'].items():
        with (ROOT/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=sha:changes[name]={'previous_sha256':sha,'current_sha256':actual}
    # Only the explicitly recorded finite-float parser hardening is permitted here.
    if set(changes)-{'src/tooltune/data.py'}:raise ValueError(f'unexpected changes: {changes}')
    raw=json.loads((ROOT/'data/raw/glaive-v2/glaive-function-calling-v2.json').read_text())
    conversations=[];rejected=Counter()
    for i,r in enumerate(raw):
        try:conversations.append(parse_conversation(r,f'glaive-v2:{i:06d}'))
        except DataError as exc:rejected[str(exc)]+=1
    del raw
    scan=json.loads((out/'scan.json').read_text())
    assert len(conversations)==scan['accepted_conversations']
    assert dict(rejected)==scan['rejected_conversations']
    splits=split_groups(list(reversed(conversations)),42)
    del conversations
    frozen=json.loads((manifests/'source_splits.json').read_text())
    reproduced={s:[{'source_record_id':r['source_record_id'],'group_id':r['group_id']} for r in rs]
                for s,rs in splits.items()}
    assert reproduced==frozen['splits']
    print('all source splits/groups reproduced with reversed input',flush=True)
    tokenizer=load_tokenizer((ROOT/'artifacts/model_path.txt').read_text().strip())
    verified={}
    for split,rs in splits.items():
        original=[json.loads(l) for l in (out/f'{split}.records.jsonl').read_text().splitlines()]
        cutoff=max(digest([42,e['sample_id']]) for e in original)
        selected=[]
        for r in rs:
            for e in decision_examples(r):
                if digest([42,e['sample_id']])>cutoff:continue
                try:f=format_example(e,tokenizer)
                except ValueError:continue
                e['input_tokens']=len(f['input_ids']);e['supervised_tokens']=len(f['target_ids'])
                selected.append(e)
        selected.sort(key=lambda e:digest([42,e['sample_id']]))
        assert selected==original
        for suffix,rows in [('records.jsonl',selected),('jsonl',[{'tools':e['tools'],'messages':e['messages']} for e in selected])]:
            payload=''.join(canonical(e)+'\n' for e in rows).encode()
            path=out/f'{split}.{suffix}'
            assert payload==path.read_bytes()
            verified[str(path.relative_to(ROOT))]=hashlib.sha256(payload).hexdigest()
        print(split,'reproduced',len(selected),'samples byte-for-byte',flush=True)
    report={'status':'passed','source_order':'reversed on second run','source_parse_counts_match':True,
        'all_source_group_assignments_match':True,'selected_files_reproduced':verified,
        'parser_hardening':changes,'seconds':time.perf_counter()-start}
    (ROOT/'artifacts/t2-reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    # All prior data bytes are unchanged; finalize the code contract before any model evaluation.
    paths=list((ROOT/'src/tooltune').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    for p in paths:contract['files'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (manifests/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    print('T2 reproduction passed',report['seconds'],'seconds',flush=True)


if __name__=='__main__':main()
