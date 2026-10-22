import hashlib
import json

import pytest

from tooltune.validation import verify_contract
from tooltune.validation import validate_records


def test_frozen_contract_rejects_changed_or_missing_file(tmp_path):
    manifest=tmp_path/'data/manifests/v1';manifest.mkdir(parents=True)
    data=tmp_path/'frozen.jsonl';data.write_text('original\n')
    sha=hashlib.sha256(data.read_bytes()).hexdigest()
    (manifest/'contract.json').write_text(json.dumps({'files':{'frozen.jsonl':sha}}))
    assert verify_contract(tmp_path)==1
    data.write_text('changed\n')
    with pytest.raises(ValueError,match='contract mismatch'):verify_contract(tmp_path)
    data.unlink()
    with pytest.raises(ValueError,match='contract mismatch'):verify_contract(tmp_path)


def test_generation_metadata_is_not_allowed_in_training_payload(tmp_path):
    out=tmp_path/'data/derived/v1';out.mkdir(parents=True)
    manifests=tmp_path/'data/manifests/v1';manifests.mkdir(parents=True)
    for split in ['train','valid','test']:
        e={'sample_id':split,'source_record_id':split,'group_id':split,'split':split,
           'tools':[],'messages':[{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello'}]}
        (out/f'{split}.records.jsonl').write_text(json.dumps(e)+'\n')
        (out/f'{split}.jsonl').write_text(json.dumps({'tools':[],'messages':e['messages']})+'\n')
        (manifests/f'{split}_ids.json').write_text(json.dumps([split]))
    assert validate_records(tmp_path)=={'train':1,'valid':1,'test':1}
    path=out/'test.jsonl';payload=json.loads(path.read_text())
    payload['expected_arguments']={'secret':'gold'};path.write_text(json.dumps(payload)+'\n')
    with pytest.raises(ValueError,match='payload'):validate_records(tmp_path)
