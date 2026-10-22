from collections import Counter

from jsonschema.exceptions import ValidationError

from .data import _validator, canonical, strict_json


def parse_prediction(text, stop_reason='eos'):
    def error(reason):return {'action':'error','reason':reason}
    if stop_reason!='eos':return error('output_truncated' if stop_reason=='length' else stop_reason)
    text=text.strip()
    if not text:return error('empty_output')
    if '<tool_call' in text or '</tool_call' in text:
        if not (text.startswith('<tool_call>') and text.endswith('</tool_call>')) \
                or text.count('<tool_call>')!=1 or text.count('</tool_call>')!=1:
            return error('invalid_call_envelope')
        try:
            call=strict_json(text[len('<tool_call>'):-len('</tool_call>')].strip())
            if not isinstance(call,dict) or set(call)!={'name','arguments'} \
                    or not isinstance(call['name'],str) or not call['name'] \
                    or not isinstance(call['arguments'],dict):
                return error('invalid_call_values')
            return {'action':'call','tool':call['name'],'arguments':call['arguments']}
        except (ValueError,TypeError):return error('invalid_call_json')
    if text.startswith('{'):
        try:
            obj=strict_json(text)
            if isinstance(obj,dict) and 'name' in obj and 'arguments' in obj:
                return error('missing_call_envelope')
        except ValueError:pass
    return {'action':'no_call'}


def score_prediction(gold, prediction):
    expected=gold['expected_action']
    if expected not in ('call','no_call'):raise ValueError('unsupported gold action')
    action=prediction['action'];call=action=='call';gold_call=expected=='call'
    tools={t['function']['name']:t['function'] for t in gold['tools']}
    function=tools.get(prediction.get('tool')) if call else None
    schema_valid=False
    if function:
        try:
            _validator(canonical(function['parameters'])).validate(prediction['arguments'])
            schema_valid=True
        except ValidationError:pass
    tool_correct=bool(gold_call and call and function and prediction['tool']==gold['expected_tool'])
    arguments_correct=bool(tool_correct and canonical(prediction['arguments'])==canonical(gold['expected_arguments']))
    strict=bool((gold_call and tool_correct and arguments_correct and schema_valid)
                or (not gold_call and action=='no_call'))
    reason=None
    if not strict:
        if action=='error':reason=prediction['reason']
        elif not gold_call:reason='unexpected_call'
        elif not call:reason='missing_call'
        elif not tool_correct:reason='wrong_tool'
        elif not schema_valid:reason='arguments_schema'
        else:reason='arguments_mismatch'
    reviewed=bool(gold.get('unavailable_tools_reviewed',False) and not gold_call)
    return {'strict_match':strict,'call_denominator':int(gold_call),
        'call_recalled':bool(gold_call and call),'tool_correct':tool_correct,
        'schema_valid':schema_valid,'arguments_correct':arguments_correct,
        'generated_call':call,'prediction_action':action,'failure_reason':reason,
        'tool_count':len(tools),'unavailable_denominator':int(reviewed),
        'unavailable_false_call':bool(reviewed and call)}


def aggregate(rows):
    def rate(n,d):return {'numerator':int(n),'denominator':int(d),'value':n/d if d else None}
    n=len(rows);calls=sum(r['call_denominator'] for r in rows)
    generated=sum(r['generated_call'] for r in rows)
    result={'samples':n,'strict_decision_match':rate(sum(r['strict_match'] for r in rows),n),
        'call_recall':rate(sum(r['call_recalled'] for r in rows),calls),
        'tool_name_accuracy':rate(sum(r['tool_correct'] for r in rows),calls),
        'argument_schema_validity':rate(sum(r['schema_valid'] and r['call_denominator'] for r in rows),calls),
        'generated_call_schema_validity':rate(sum(r['schema_valid'] for r in rows),generated),
        'argument_strict_match':rate(sum(r['arguments_correct'] for r in rows),calls),
        'unavailable_tools_false_call_rate':rate(sum(r['unavailable_false_call'] for r in rows),
                                                sum(r['unavailable_denominator'] for r in rows)),
        'clarification':rate(0,0),
        'prediction_actions':dict(Counter(r['prediction_action'] for r in rows)),
        'failures':dict(Counter(r['failure_reason'] for r in rows if r['failure_reason'])),
        'tool_name_accuracy_by_tool_count':{}}
    for count in sorted({r['tool_count'] for r in rows if r['call_denominator']}):
        bucket=[r for r in rows if r['call_denominator'] and r['tool_count']==count]
        result['tool_name_accuracy_by_tool_count'][str(count)]=rate(sum(r['tool_correct'] for r in bucket),len(bucket))
    return result


def select_checkpoint(candidates):
    if not candidates:raise ValueError('no checkpoints')
    denominators={c['summary']['strict_decision_match']['denominator'] for c in candidates}
    if len(denominators)!=1 or 0 in denominators:raise ValueError('incompatible Validation denominators')
    def key(c):
        s=c['summary'];false=s['unavailable_tools_false_call_rate']['value']
        return (-s['strict_decision_match']['value'], false if false is not None else float('inf'),c['updates'])
    return min(candidates,key=key)
