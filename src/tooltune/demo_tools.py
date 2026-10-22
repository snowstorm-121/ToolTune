"""Two local, deterministic read-only tools for execution demonstrations."""
from jsonschema import Draft7Validator

TOOLS=[
    {'type':'function','function':{'name':'sum_integers',
        'description':'Return the sum of a list of integers.',
        'parameters':{'type':'object','properties':{'values':{'type':'array','items':{'type':'integer'}}},
            'required':['values'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'text_length',
        'description':'Return the number of Unicode code points in a string.',
        'parameters':{'type':'object','properties':{'text':{'type':'string'}},
            'required':['text'],'additionalProperties':False}}},
]


def execute(call):
    if call['action']!='call':raise ValueError('no complete structured tool call')
    tool=next((t['function'] for t in TOOLS if t['function']['name']==call['tool']),None)
    if tool is None:raise ValueError('tool not available in local demo')
    arguments=call['arguments']
    Draft7Validator(tool['parameters']).validate(arguments)
    if call['tool']=='sum_integers':return {'sum':sum(arguments['values'])}
    return {'length':len(arguments['text'])}
