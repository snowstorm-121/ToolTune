"""T2: strict source parsing, connected groups, frozen manifests and native datasets."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tooltune.data import (DataError, canonical, decision_examples, digest,
    normalize_query, parse_conversation, query_shingles, split_groups, tool_families)
from tooltune.formatting import TEMPLATE, format_example, load_tokenizer
from tooltune.train import training_dataset

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def write_jsonl(path, rows):
    with path.open('w') as f:
        for row in rows: f.write(canonical(row)+'\n')


def main():
    start = time.perf_counter()
    out = ROOT/'data/derived/v1'
    manifest_dir = ROOT/'data/manifests/v1'
    out.mkdir(parents=True, exist_ok=False)
    manifest_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'configs/evaluation.json').read_text())
    raw_manifest = json.loads((ROOT/'data/manifests/raw.json').read_text())
    raw_path = ROOT/raw_manifest['path']
    with raw_path.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest() == raw_manifest['sha256']
    records = json.loads(raw_path.read_text())
    accepted = []; rejected = Counter();tool_counts=Counter();role_counts=Counter()
    with (out/'quarantine_sources.jsonl').open('w') as log:
        for i,record in enumerate(records):
            source_id = f'glaive-v2:{i:06d}'
            try:
                parsed = parse_conversation(record,source_id)
                accepted.append(parsed);tool_counts[len(parsed['tools'])]+=1
                role_counts.update(m['role'] for m in parsed['messages'])
            except DataError as exc:
                rejected[str(exc)]+=1
                log.write(canonical({'source_record_id':source_id,'source_sha256':digest(record),
                                     'reason':str(exc)})+'\n')
            if i%10000==0:print('parsed',i,'accepted',len(accepted),flush=True)
    scan = {'raw_records':len(records),'accepted_conversations':len(accepted),
        'rejected_conversations':dict(rejected),'available_tool_count':dict(tool_counts),
        'accepted_message_roles':dict(role_counts)}
    del records
    write_json(out/'scan.json',scan)
    print('scan',json.dumps(scan),flush=True)
    splits = split_groups(accepted, config['seed'])
    del accepted
    group_sizes=Counter(r['group_id'] for rs in splits.values() for r in rs)
    split_manifest = {'protocol':'v1','seed':config['seed'],'raw':raw_manifest,
        'grouping':'tool signature families + complete conversation + normalized user queries + word-trigram Jaccard>=.85 on >=8-word queries; transitive union',
        'query_normalization':'casefold, quoted strings and numbers to placeholders, Unicode word tokens',
        'group_count':len(group_sizes),'largest_group_sizes':sorted(group_sizes.values(),reverse=True)[:20],
        'splits':{s:[{'source_record_id':r['source_record_id'],'group_id':r['group_id']}
                     for r in rs] for s,rs in splits.items()}}
    # Freeze all source/group assignments before decision export and token filtering.
    write_json(manifest_dir/'source_splits.json',split_manifest)
    print('groups',len(group_sizes),'largest',split_manifest['largest_group_sizes'],
          'split_sources',{s:len(rs) for s,rs in splits.items()},flush=True)
    tokenizer=load_tokenizer((ROOT/'artifacts/model_path.txt').read_text().strip())
    summary = dict(scan,groups=len(group_sizes),largest_group_sizes=split_manifest['largest_group_sizes'])
    selected={};masks=[]
    with (out/'quarantine_decisions.jsonl').open('w') as log:
        for split,conversations in splits.items():
            candidates=[];filtered=Counter();actions=Counter();lengths=Counter();total=0
            for c in conversations:
                for e in decision_examples(c):
                    total+=1
                    try:
                        f=format_example(e,tokenizer,config['max_seq_length'])
                    except ValueError as exc:
                        reason=str(exc).split(':')[0];filtered[reason]+=1
                        log.write(canonical({'sample_id':e['sample_id'],'reason':str(exc)})+'\n')
                        continue
                    e['input_tokens']=len(f['input_ids']);e['supervised_tokens']=len(f['target_ids'])
                    actions[e['expected_action']]+=1;lengths[len(f['input_ids'])//64*64]+=1
                    candidates.append(e)
            candidates.sort(key=lambda e:digest([config['seed'],e['sample_id']]))
            chosen=candidates[:config['sample_caps'][split]]
            selected[split]=chosen
            summary[split]={'source_conversations':len(conversations),'decisions_before_length_filter':total,
                'decision_filters':dict(filtered),'eligible_decisions':len(candidates),
                'eligible_actions':dict(actions),'length_histogram_bins64':dict(lengths),
                'selected_decisions':len(chosen),
                'selected_source_conversations':len({r['source_record_id'] for r in chosen}),
                'selected_actions':dict(Counter(r['expected_action'] for r in chosen)),
                'selected_tool_counts':dict(Counter(len(r['tools']) for r in chosen)),
                'selected_supervised_tokens':sum(r['supervised_tokens'] for r in chosen),
                'selected_total_tokens':sum(r['input_tokens'] for r in chosen)}
            write_json(manifest_dir/f'{split}_ids.json',[e['sample_id'] for e in chosen])
            write_jsonl(out/f'{split}.records.jsonl',chosen)
            # Only model inputs/targets go to upstream data loader: no expected_* fields.
            write_jsonl(out/f'{split}.jsonl',({'tools':e['tools'],'messages':e['messages']} for e in chosen))
            if chosen:
                dataset=training_dataset(chosen,tokenizer)
                for i,e in enumerate(chosen):
                    f=format_example(e,tokenizer)
                    tokens,offset=dataset[i]
                    assert tokens==f['input_ids'] and offset==len(f['prompt_ids'])
            print(split,summary[split],flush=True)
    # Independent cross-split family/group/exact-template audit over all source rows.
    audits={}
    for i,a in enumerate(splits):
        for b in list(splits)[i+1:]:
            for kind,func in [('groups',lambda r:{r['group_id']}),('families',tool_families),
                ('queries',lambda r:{normalize_query(m['content']) for m in r['messages'] if m['role']=='user'})]:
                left=set().union(*(func(r) for r in splits[a])) if splits[a] else set()
                right=set().union(*(func(r) for r in splits[b])) if splits[b] else set()
                audits[f'{a}:{b}:{kind}_overlap']=len(left&right)
                assert not left&right
    # Exhaustive selected-query cross-split trigram overlap scan via inverted index.
    indexes={}
    for split,examples in selected.items():
        queries={normalize_query(m['content']) for e in examples for m in e['messages'] if m['role']=='user'}
        indexes[split]={q:query_shingles(q) for q in queries if query_shingles(q)}
    near=[]
    for i,a in enumerate(indexes):
        for b in list(indexes)[i+1:]:
            postings=defaultdict(set)
            for q,sh in indexes[b].items():
                for s in sh:postings[s].add(q)
            for q,sh in indexes[a].items():
                candidates=set().union(*(postings[s] for s in sh))
                for other in candidates:
                    other_sh=indexes[b][other]
                    if len(sh&other_sh)/len(sh|other_sh)>=.85:
                        near.append({'splits':[a,b],'query_hashes':[digest(q),digest(other)]})
    audits['selected_cross_split_near_duplicate_pairs']=len(near)
    write_json(out/'leakage_audit.json',dict(audits,pairs=near))
    assert not near
    # Readable examples from train only; do not inspect Test predictions or choose on Test.
    train=selected['train']
    cases={
        'call':next((r for r in train if r['expected_action']=='call'),None),
        'ordinary_answer':next((r for r in train if not r['tools']),None),
        'tools_present_noncall':next((r for r in train if r['tools'] and r['expected_action']=='no_call' and len(r['messages'])==3),None),
        'tool_history':next((r for r in train if any(m['role']=='tool' for m in r['messages'][:-1])),None)}
    for case,e in cases.items():
        if e:
            f=format_example(e,tokenizer)
            masks.append({'case':case,'sample_id':e['sample_id'],**f,
                'prompt':tokenizer.decode(f['prompt_ids']),'target':tokenizer.decode(f['target_ids']),
                'tokens':[{'id':t,'piece':tokenizer.decode([t]),'loss_mask':m}
                          for t,m in zip(f['input_ids'],f['loss_mask'])]})
    write_json(out/'mask_examples.json',masks)
    summary['mask_example_cases']={k:v['sample_id'] if v else None for k,v in cases.items()}
    summary['explicit_clarification_gold_labels']=0
    summary['unavailable_tools_reviewed_labels']=0
    summary['leakage_audit']=audits
    summary['seconds']=time.perf_counter()-start
    write_json(out/'summary.json',summary)
    contract={'raw_sha256':raw_manifest['sha256'],'template_sha256':hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        'files':{},'model_manifest_sha256':hashlib.sha256((ROOT/'artifacts/model_manifest.json').read_bytes()).hexdigest()}
    for p in sorted(list(out.glob('*.jsonl'))+list(manifest_dir.glob('*.json'))+
                    [ROOT/'src/tooltune/data.py',ROOT/'src/tooltune/formatting.py',ROOT/'configs/evaluation.json',ROOT/'uv.lock']):
        contract['files'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(manifest_dir/'contract.json',contract)
    print('T2 prepared',summary['seconds'],'seconds',flush=True)


if __name__=='__main__':main()
