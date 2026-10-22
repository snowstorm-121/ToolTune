from collections import Counter

import pytest

from tooltune.sampling import balanced_sample


def examples():
    return [{'sample_id':str(i), 'expected_action':'call' if i<5 else 'no_call',
             'group_id':f'group-{i}', 'source_record_id':str(i), 'split':'valid'}
            for i in range(12)]


def test_balance_is_deterministic_and_keeps_source_group_metadata():
    rows=examples();chosen=balanced_sample(rows,3)
    assert Counter(r['expected_action'] for r in chosen)=={'call':3,'no_call':3}
    assert chosen==balanced_sample(list(reversed(rows)),3)
    assert all(r in rows for r in chosen)
    assert len({r['sample_id'] for r in chosen})==6


def test_insufficient_real_labels_raise_instead_of_fabricating():
    with pytest.raises(ValueError,match='insufficient'):balanced_sample(examples(),6)


def test_duplicate_ids_or_unknown_actions_rejected():
    rows=examples()
    with pytest.raises(ValueError,match='duplicate'):balanced_sample(rows+[rows[0]],3)
    rows[0]=dict(rows[0],expected_action='clarify')
    with pytest.raises(ValueError,match='action'):balanced_sample(rows,3)
