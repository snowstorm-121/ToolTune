from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
import re

from jsonschema import Draft7Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError


class DataError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _unique_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result: raise ValueError('duplicate JSON key')
        result[k] = v
    return result


def _nonfinite(value):
    raise ValueError('non-finite JSON number')


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value): raise ValueError('non-finite JSON number')
    return value


DECODER = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_nonfinite,
                          parse_float=_finite_float)


def strict_json(text):
    obj, end = DECODER.raw_decode(text.lstrip())
    if text.lstrip()[end:].strip(): raise ValueError('trailing JSON data')
    return obj


def parse_call(text):
    """Parse Glaive's outer object, including its single-quoted JSON args envelope.

    This is a small grammar, not Python evaluation or quote replacement. JSONDecoder
    consumes the complete nested JSON before the closing envelope quote is checked.
    """
    text = text.strip()
    i = 0
    def ws():
        nonlocal i
        while i < len(text) and text[i].isspace(): i += 1
    def take(char):
        nonlocal i
        ws()
        if i >= len(text) or text[i] != char: raise ValueError('invalid call delimiter')
        i += 1
    take('{')
    values = {}
    while True:
        ws()
        key, i = DECODER.raw_decode(text, i)
        if key not in ('name', 'arguments') or key in values: raise ValueError('invalid call keys')
        take(':');ws()
        if key == 'arguments' and i < len(text) and text[i] == "'":
            i += 1;ws()
            value, i = DECODER.raw_decode(text, i)
            take("'")
        else:
            value, i = DECODER.raw_decode(text, i)
        if key == 'arguments' and isinstance(value, str): value = strict_json(value)
        values[key] = value
        ws()
        if i < len(text) and text[i] == ',': i += 1;continue
        take('}');ws()
        if i != len(text): raise ValueError('multiple or mixed calls')
        break
    if set(values) != {'name', 'arguments'} or not isinstance(values['name'], str) \
            or not isinstance(values['arguments'], dict):
        raise ValueError('invalid call values')
    return values


@lru_cache(maxsize=8192)
def _validator(schema_text):
    schema = strict_json(schema_text)
    # This project never fetches remote JSON Schema references.
    if '"$ref"' in schema_text: raise SchemaError('references are outside v1 scope')
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema, format_checker=FormatChecker())


def parse_conversation(record, source_record_id):
    try:
        system, chat = record['system'], record['chat']
        if not isinstance(system, str) or not isinstance(chat, str): raise DataError('invalid_fields')
        start = system.find('{')
        tools = []
        if start >= 0:
            tail = system[start:]
            while tail.strip():
                tail = tail.lstrip()
                f, end = DECODER.raw_decode(tail)
                if not isinstance(f, dict) or not isinstance(f.get('name'), str):
                    raise DataError('invalid_tool')
                schema = f.get('parameters')
                if not isinstance(schema, dict) or schema.get('type') != 'object':
                    raise DataError('invalid_schema')
                _validator(canonical(schema))
                tools.append({'type': 'function', 'function': f})
                tail = tail[end:]
        names = [t['function']['name'] for t in tools]
        if len(names) != len(set(names)): raise DataError('duplicate_tool')
        lookup = {t['function']['name']: t['function'] for t in tools}
        prefix = system[:start] if start >= 0 else system
        prefix = prefix.removeprefix('SYSTEM:').strip().rstrip('-').rstrip()
        messages = [{'role': 'system', 'content': prefix}]
        markers = list(re.finditer(r'(?m)^(USER|ASSISTANT|FUNCTION RESPONSE):\s*', chat))
        if not markers or chat[:markers[0].start()].strip(): raise DataError('broken_roles')
        expected = 'USER'
        pending_call = False
        for j, marker in enumerate(markers):
            role = marker.group(1)
            if role != expected: raise DataError('broken_roles')
            end = markers[j+1].start() if j+1 < len(markers) else len(chat)
            text = chat[marker.end():end].strip()
            if not text: raise DataError('empty_message')
            if role == 'ASSISTANT':
                if not text.endswith('<|endoftext|>'): raise DataError('incomplete_assistant')
                text = text[:-len('<|endoftext|>')].rstrip()
                if '<|endoftext|>' in text: raise DataError('broken_roles')
                if '<functioncall>' in text:
                    if not text.startswith('<functioncall>'): raise DataError('mixed_action')
                    try: call = parse_call(text[len('<functioncall>'):])
                    except (ValueError, TypeError, IndexError) as exc:
                        raise DataError('invalid_call') from exc
                    name, arguments = call['name'], call['arguments']
                    if name not in lookup: raise DataError('unknown_tool')
                    try: _validator(canonical(lookup[name]['parameters'])).validate(arguments)
                    except ValidationError as exc: raise DataError('arguments_schema') from exc
                    messages.append({'role': 'assistant', 'content': '', 'tool_calls': [
                        {'type': 'function', 'function': call}]})
                    pending_call = True;expected = 'FUNCTION RESPONSE'
                else:
                    if not text: raise DataError('empty_target')
                    messages.append({'role': 'assistant', 'content': text})
                    pending_call = False;expected = 'USER'
            elif role == 'FUNCTION RESPONSE':
                if not pending_call: raise DataError('orphan_tool_response')
                strict_json(text)
                messages.append({'role': 'tool', 'content': text})
                pending_call = False;expected = 'ASSISTANT'
            else:
                if '<|endoftext|>' in text: raise DataError('broken_roles')
                messages.append({'role': 'user', 'content': text})
                expected = 'ASSISTANT'
        if expected != 'USER' or messages[-1]['role'] != 'assistant':
            raise DataError('incomplete_conversation')
        return {'source_record_id': source_record_id, 'tools': tools, 'messages': messages}
    except DataError: raise
    except SchemaError as exc: raise DataError('invalid_schema') from exc
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise DataError('invalid_json_or_fields') from exc


def decision_examples(conversation):
    out = []
    for i, message in enumerate(conversation['messages']):
        if message['role'] != 'assistant': continue
        call = message.get('tool_calls', [])
        function = call[0]['function'] if call else None
        out.append({'sample_id': f"{conversation['source_record_id']}:a{i}",
            'source_record_id': conversation['source_record_id'], 'turn_index': i,
            'group_id': conversation.get('group_id'), 'split': conversation.get('split'),
            'tools': conversation['tools'], 'messages': conversation['messages'][:i+1],
            'expected_action': 'call' if function else 'no_call',
            'expected_tool': function['name'] if function else None,
            'expected_arguments': function['arguments'] if function else None,
            'filter_reason': None})
    return out


def parse_record(record, source_record_id):
    return decision_examples(parse_conversation(record, source_record_id))


def structural(value):
    if isinstance(value, dict):
        return {k: ({name: structural(schema) for name,schema in v.items()}
                    if k in ('properties', 'patternProperties', 'definitions', '$defs')
                    else sorted([structural(x) for x in v], key=canonical)
                    if k in ('required', 'enum') and isinstance(v, list) else structural(v))
                for k,v in value.items() if k not in ('description', 'title', 'examples')}
    if isinstance(value, list): return [structural(x) for x in value]
    return value


def tool_families(record):
    return {digest(structural(t['function'])) for t in record['tools']}


def normalize_query(text):
    text = text.casefold()
    text = re.sub(r'"[^"\n]*"', ' quotedvalue ', text)
    text = re.sub(r'\b\d+(?:[.:/-]\d+)*\b', ' numbervalue ', text)
    return ' '.join(re.findall(r'\w+', text))


def query_shingles(text):
    words = text.split()
    return {' '.join(words[i:i+3]) for i in range(len(words)-2)} if len(words) >= 8 else set()


def split_groups(records, seed=42):
    """Union families, exact queries, normalized templates and candidate near duplicates.

    Near duplicates: Jaccard >=.85 on word trigrams for queries with >=8 words.
    Rare-trigram prefix filtering retrieves every candidate meeting that threshold;
    it is not semantic deduplication. Selected examples are independently scanned.
    """
    rows = sorted(records, key=lambda r: r['source_record_id'])
    ids = [r['source_record_id'] for r in rows]
    if len(ids) != len(set(ids)): raise ValueError('duplicate source IDs')
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]];i = parents[i]
        return i
    def union(i,j):
        a,b = find(i),find(j)
        if a != b: parents[max(a,b)] = min(a,b)
    owners = {}
    queries = {}
    for i,r in enumerate(rows):
        keys = [('family', f) for f in tool_families(r)]
        keys.append(('conversation', digest({'tools': sorted(r['tools'],key=canonical), 'messages':r['messages']})))
        for message in r['messages']:
            if message['role'] != 'user': continue
            query = normalize_query(message['content'])
            keys.append(('query',query))
            if query in queries: union(i, queries[query][0])
            else: queries[query] = (i, query_shingles(query))
        for key in keys:
            if key in owners: union(i,owners[key])
            else: owners[key] = i
    freq = Counter(s for _,shingles in queries.values() for s in shingles)
    postings = defaultdict(list)
    for query,(i,shingles) in sorted(queries.items()):
        if not shingles: continue
        candidates = set()
        prefix_length = len(shingles) - math.ceil(.85 * len(shingles)) + 1
        for s in sorted(shingles,key=lambda s:(freq[s],s))[:prefix_length]:
            candidates.update(postings[s])
        for other in candidates:
            j,other_shingles = queries[other]
            if len(shingles & other_shingles) / len(shingles | other_shingles) >= .85: union(i,j)
        # Index all trigrams so rare-query lookup can find earlier candidates.
        for s in shingles: postings[s].append(query)
    members = defaultdict(list)
    for i,r in enumerate(rows): members[find(i)].append(r)
    output = {'train': [], 'valid': [], 'test': []}
    for group in members.values():
        group_id = digest([r['source_record_id'] for r in group])
        bucket = int(digest([seed,group_id])[:16],16) / 2**64
        split = 'train' if bucket < .8 else ('valid' if bucket < .9 else 'test')
        for row in group: output[split].append(dict(row, group_id=group_id, split=split))
    for split in output: output[split].sort(key=lambda r:r['source_record_id'])
    return output
