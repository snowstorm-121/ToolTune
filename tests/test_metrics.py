import pytest

from tooltune.metrics import aggregate, parse_prediction, score_prediction, select_checkpoint


def gold(action='call'):
    return {'expected_action':action,'expected_tool':'add' if action=='call' else None,
        'expected_arguments':{'a':1,'b':2} if action=='call' else None,
        'tools':[{'type':'function','function':{'name':'add','parameters':{'type':'object',
        'properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],
        'additionalProperties':False}}}]}


def test_argument_key_order_is_not_an_error():
    p=parse_prediction('<tool_call>\n{"name":"add","arguments":{"b":2,"a":1}}\n</tool_call>')
    assert score_prediction(gold(),p)['strict_match'] is True


def test_missing_call_and_truncation_remain_in_denominator():
    p=parse_prediction('partial','length');r=score_prediction(gold(),p)
    assert not r['strict_match'] and r['call_denominator']==1
    s=aggregate([r])
    assert s['strict_decision_match']=={'numerator':0,'denominator':1,'value':0.0}
    assert s['argument_strict_match']['denominator']==1


@pytest.mark.parametrize('text',[
    '<tool_call>{"name":"add","arguments":"{\\"a\\":1}"}</tool_call>',
    '<tool_call>{"name":"add","arguments":{"a":1,"a":2}}</tool_call>',
    '<tool_call>{"name":"add","arguments":{}}</tool_call> trailing',
    '<tool_call>{"name":"add","arguments":{}}</tool_call><tool_call>{}</tool_call>',
    '<tool_call>{broken',
    '{"name":"add","arguments":{}}',
    '```json\n<tool_call>{"name":"add","arguments":{}}</tool_call>\n```',
    '',
])
def test_malformed_empty_or_mixed_output_is_not_repaired(text):
    assert parse_prediction(text)['action']=='error'


def test_ordinary_text_does_not_fabricate_calls():
    p=parse_prediction('The add function can add numbers.')
    assert p['action']=='no_call'
    assert score_prediction(gold('no_call'),p)['strict_match']


@pytest.mark.parametrize('tool,args,reason',[
    ('unknown',{'a':1,'b':2},'wrong_tool'),
    ('add',{'a':'1','b':2},'arguments_schema'),
    ('add',{'a':True,'b':2},'arguments_schema'),
    ('add',{'a':1},'arguments_schema'),
    ('add',{'a':1,'b':2,'c':3},'arguments_schema'),
    ('add',{'a':1,'b':3},'arguments_mismatch'),
    ('add',{'a':1.0,'b':2},'arguments_mismatch'),
])
def test_tool_schema_and_semantic_errors_are_distinct(tool,args,reason):
    row=score_prediction(gold(),{'action':'call','tool':tool,'arguments':args})
    assert not row['strict_match'] and row['failure_reason']==reason


def test_array_order_and_string_case_are_preserved():
    e={'expected_action':'call','expected_tool':'f','expected_arguments':{'items':['A','B']},
       'tools':[{'function':{'name':'f','parameters':{'type':'object'}}}]}
    assert not score_prediction(e,{'action':'call','tool':'f','arguments':{'items':['B','A']}})['strict_match']
    assert not score_prediction(e,{'action':'call','tool':'f','arguments':{'items':['a','B']}})['strict_match']


def test_zero_denominators_are_na_and_false_call_is_an_error():
    r=score_prediction(gold('no_call'),{'action':'call','tool':'add','arguments':{'a':1,'b':2}})
    s=aggregate([r])
    assert r['failure_reason']=='unexpected_call'
    assert s['call_recall']['value'] is None
    assert s['unavailable_tools_false_call_rate']['value'] is None


def test_checkpoint_tie_selects_earlier_without_test_results():
    summary=aggregate([score_prediction(gold(),{'action':'call','tool':'add','arguments':{'a':1,'b':2}})])
    chosen=select_checkpoint([{'updates':200,'summary':summary,'path':'later'},
                              {'updates':100,'summary':summary,'path':'earlier'}])
    assert chosen['path']=='earlier'
