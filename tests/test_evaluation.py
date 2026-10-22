import copy
import json

import pytest

from tooltune.evaluate import generation_input, read_completed, recompute


def test_generation_inputs_exclude_target_and_all_scoring_metadata():
    e={'tools':[],'messages':[{'role':'user','content':'Hi'},
        {'role':'assistant','content':'GOLD_SECRET'}], 'expected_arguments':{'secret':'a'}}
    first=generation_input(e);other=copy.deepcopy(e)
    other['messages'][-1]['content']='OTHER_GOLD';other['expected_arguments']={'different':2}
    assert first==generation_input(other)
    assert first=={'tools':[],'messages':[{'role':'user','content':'Hi'}]}


def test_resume_preserves_first_output_and_refuses_mismatched_contract(tmp_path):
    c={'model':'base','adapter':None};p=tmp_path/'run';p.mkdir()
    (p/'contract.json').write_text(json.dumps(c))
    (p/'predictions.jsonl').write_text(json.dumps({'sample_id':'s','output':'first'})+'\n')
    completed=read_completed(p,c,['s','t'])
    assert completed['s']['output']=='first' and 't' not in completed
    with pytest.raises(ValueError,match='contract'):read_completed(p,{'model':'other'},['s'])
    with pytest.raises(ValueError,match='unknown'):read_completed(p,c,['t'])


def test_duplicate_prediction_ids_rejected(tmp_path):
    (tmp_path/'contract.json').write_text('{}')
    line=json.dumps({'sample_id':'s','output':'first'})+'\n'
    (tmp_path/'predictions.jsonl').write_text(line+line)
    with pytest.raises(ValueError,match='duplicate'):read_completed(tmp_path,{},['s'])


def test_recompute_retains_failed_rows_and_requires_complete_ids():
    e={'sample_id':'s','source_record_id':'source','group_id':'g','tools':[],
       'expected_action':'no_call','expected_tool':None,'expected_arguments':None}
    scores,summary=recompute([e],{'s':{'output':'unfinished','stop_reason':'length'}})
    assert scores[0]['failure_reason']=='output_truncated'
    assert summary['samples']==1 and summary['strict_decision_match']['numerator']==0
    with pytest.raises(ValueError,match='incomplete'):recompute([e],{})
