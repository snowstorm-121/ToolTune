from .data import digest


def balanced_sample(records, calls, seed=42):
    if len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('duplicate sample IDs')
    if any(r['expected_action'] not in ('call', 'no_call') for r in records):
        raise ValueError('unsupported action')
    selected=[]
    for action in ('call', 'no_call'):
        pool=sorted((r for r in records if r['expected_action']==action),
                    key=lambda r:digest([seed,r['sample_id']]))
        if len(pool)<calls:raise ValueError(f'insufficient {action}: {len(pool)} < {calls}')
        selected.extend(pool[:calls])
    return sorted(selected,key=lambda r:digest([seed,r['sample_id']]))
