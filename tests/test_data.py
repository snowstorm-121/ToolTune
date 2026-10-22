import copy
import json

import pytest

from tooltune.data import DataError, parse_record, split_groups


def raw(chat=None, tools=None):
    if tools is None:
        tools = [{'name': 'lookup', 'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string'}, 'options': {'type': 'object',
            'properties': {'n': {'type': 'integer'}}, 'required': ['n'],
            'additionalProperties': False}}, 'required': ['query'],
            'additionalProperties': False}}]
    return {'system': 'SYSTEM: Tools available -\n'+ '\n\n'.join(json.dumps(t) for t in tools),
        'chat': chat or ('USER: Look up a record\n\nASSISTANT: <functioncall> '+
        '{"name":"lookup","arguments":\'{"query":"O\'Brien", "options":{"n":2}}\'}'+
        ' <|endoftext|>\n\nFUNCTION RESPONSE: {"value":2}\n\n'+
        'ASSISTANT: Found it. <|endoftext|>')}


def test_nested_json_single_quoted_wrapper_and_apostrophe():
    out = parse_record(raw(), 'source-1')
    assert len(out) == 2
    assert out[0]['expected_arguments'] == {'query': "O'Brien", 'options': {'n': 2}}
    assert out[0]['expected_action'] == 'call'
    assert out[1]['messages'][-2]['role'] == 'tool'
    assert all(x['source_record_id'] == 'source-1' for x in out)


def test_plain_tool_name_mention_is_not_a_call():
    out = parse_record(raw('USER: Hello\n\nASSISTANT: You can use lookup. <|endoftext|>'), 's')
    assert out[0]['expected_action'] == 'no_call'
    assert out[0]['expected_tool'] is None


@pytest.mark.parametrize('payload,reason', [
    ('{"name":"unknown","arguments":{}}', 'unknown_tool'),
    ('{"name":"lookup","arguments":{}}', 'arguments_schema'),
    ('{"name":"lookup","arguments":{"query":2}}', 'arguments_schema'),
    ('{"name":"lookup","arguments":{"query":"x","extra":1}}', 'arguments_schema'),
    ('{"name":"lookup","arguments":{"query":"x","options":{"n":"2"}}}', 'arguments_schema'),
    ('{"name":"lookup","arguments":{"query":"x","query":"y"}}', 'invalid_call'),
    ('{"name":"lookup","arguments": "not JSON"}', 'invalid_call'),
    ('{"name":"lookup","arguments":{"query":"x","n":1e999}}', 'invalid_call'),
    ('{"name":"lookup","arguments":{}} {"name":"lookup","arguments":{}}', 'invalid_call'),
])
def test_invalid_calls_are_quarantined(payload, reason):
    chat = 'USER: Request\n\nASSISTANT: <functioncall> '+payload+ ' <|endoftext|>\n\n'+\
           'FUNCTION RESPONSE: {}\n\nASSISTANT: Done. <|endoftext|>'
    with pytest.raises(DataError, match=reason): parse_record(raw(chat), 's')


@pytest.mark.parametrize('chat', [
    'ASSISTANT: Hello <|endoftext|>',
    'USER: Request\n\nASSISTANT: Truncated',
    'USER: Request\n\nFUNCTION RESPONSE: {}',
    'USER: First\n\nUSER: Second\n\nASSISTANT: hi <|endoftext|>',
    'USER: Request\n\nASSISTANT: <functioncall> {"name":"lookup","arguments":{"query":"x"}} <|endoftext|>',
    'USER: Request\n\nASSISTANT: I will call <functioncall> {} <|endoftext|>',
])
def test_broken_roles_incomplete_and_mixed_records_rejected(chat):
    with pytest.raises(DataError): parse_record(raw(chat), 's')


def test_duplicate_tool_names_rejected():
    r = raw(); tool = {'name':'lookup','parameters':{'type':'object'}}
    with pytest.raises(DataError, match='duplicate_tool'):
        parse_record(raw(tools=[tool,tool]), 's')


def test_no_tools_chat_supported():
    r = {'system':'SYSTEM: You have no external functions.',
         'chat':'USER: Hello\n\nASSISTANT: Hi <|endoftext|>'}
    assert parse_record(r, 's')[0]['tools'] == []


def grouped_record(id, name, user, extra=None):
    tools=[{'type':'function','function':{'name':name,'description':'prose',
        'parameters':{'type':'object','properties':{'x':{'type':'string'}},'required':['x']}}}]
    if extra: tools.append(copy.deepcopy(extra))
    return {'source_record_id':id, 'tools':tools, 'messages':[{'role':'user','content':user}]}


def test_group_connectivity_family_duplicates_and_stable_order():
    a=grouped_record('a','first','Please look up the customer record for account number 123 today')
    b=grouped_record('b','second','Please look up the customer record for account number 456 today')
    c=grouped_record('c','second','A completely different query')
    d=grouped_record('d','third','Another entirely different request')
    one=split_groups([a,b,c,d]);two=split_groups([d,c,b,a])
    def membership(s):return {r['source_record_id']:(key,r['group_id']) for key,rs in s.items() for r in rs}
    assert membership(one)==membership(two)
    from tooltune.data import canonical
    assert canonical(one)==canonical(two)
    m=membership(one)
    assert m['a']==m['b']==m['c']
    assert m['d'][1]!=m['a'][1]


def test_multitool_family_overlap_connects_records():
    a=grouped_record('a','one','a')
    b=grouped_record('b','two','b',extra=a['tools'][0])
    result=split_groups([a,b])
    groups={r['group_id'] for rs in result.values() for r in rs}
    assert len(groups)==1


def test_near_duplicate_with_one_extra_word_is_grouped():
    a=grouped_record('a','one','Please look up the detailed customer account information for my latest recent transaction today')
    b=grouped_record('b','two','Please look up the detailed customer account information for my latest recent transaction today thanks')
    result=split_groups([a,b])
    assert len({r['group_id'] for rs in result.values() for r in rs})==1


def test_duplicate_source_ids_rejected():
    a=grouped_record('a','one','a')
    with pytest.raises(ValueError,match='duplicate source'):split_groups([a,a])


def test_schema_properties_named_title_or_description_keep_types():
    from tooltune.data import tool_families
    a=grouped_record('a','one','a');b=copy.deepcopy(a)
    a['tools'][0]['function']['parameters']['properties']={'title':{'type':'string'}}
    b['tools'][0]['function']['parameters']['properties']={'title':{'type':'integer'}}
    assert tool_families(a)!=tool_families(b)


def test_tool_order_and_required_order_do_not_change_family():
    from tooltune.data import tool_families
    a=grouped_record('a','one','a');b=copy.deepcopy(a)
    a['tools'][0]['function']['parameters']['required']=['x','title']
    b['tools'][0]['function']['parameters']['required']=['title','x']
    b['tools'][0]['function']['description']='Different wording'
    assert tool_families(a)==tool_families(b)
