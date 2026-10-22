import hashlib
import json
from pathlib import Path


def verify_contract(root):
    root = Path(root)
    contract = json.loads((root/'data/manifests/v1/contract.json').read_text())
    for name, expected in contract['files'].items():
        path = root/name
        if not path.is_file():
            raise ValueError(f'contract mismatch: {name}')
        with path.open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual != expected:raise ValueError(f'contract mismatch: {name}')
    return len(contract['files'])


def validate_records(root):
    root = Path(root)
    counts = {}; seen_ids = set(); source_split = {}; group_split = {}
    for split in ('train', 'valid', 'test'):
        data = root/'data/derived/v1'
        records = [json.loads(l) for l in (data/f'{split}.records.jsonl').read_text().splitlines()]
        payloads = [json.loads(l) for l in (data/f'{split}.jsonl').read_text().splitlines()]
        ids = json.loads((root/f'data/manifests/v1/{split}_ids.json').read_text())
        if [r['sample_id'] for r in records] != ids or len(payloads) != len(records):
            raise ValueError('sample IDs or payload count mismatch')
        for e,p in zip(records,payloads):
            if p != {'tools':e['tools'], 'messages':e['messages']}:
                raise ValueError('training payload contains unexpected metadata or target changes')
            if e['split'] != split or e['sample_id'] in seen_ids:
                raise ValueError('duplicate sample or wrong split')
            seen_ids.add(e['sample_id'])
            for key,lookup in [('source_record_id',source_split),('group_id',group_split)]:
                if e[key] in lookup and lookup[e[key]] != split: raise ValueError('split overlap')
                lookup[e[key]] = split
        counts[split] = len(records)
    return counts
