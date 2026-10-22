import copy
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tooltune.formatting import format_example, load_tokenizer
from tooltune.train import training_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL = next((ROOT / '.cache/huggingface').glob('models--*/snapshots/*'))


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer(MODEL)


def example():
    return {'tools': [], 'messages': [
        {'role': 'system', 'content': 'Answer briefly.'},
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi!'},
    ]}


def test_official_instruct_template_has_no_thinking(tokenizer):
    text = tokenizer.apply_chat_template(example()['messages'], tokenize=False)
    assert '<think>' not in text


def test_prefix_target_and_end_tokens(tokenizer):
    out = format_example(example(), tokenizer)
    prompt = tokenizer.apply_chat_template(example()['messages'][:-1],
        tools=[], add_generation_prompt=True, return_dict=False)
    assert out['prompt_ids'] == prompt
    assert out['input_ids'] == out['prompt_ids'] + out['target_ids']
    assert out['loss_mask'] == [0] * len(prompt) + [1] * len(out['target_ids'])
    assert tokenizer.eos_token_id in out['target_ids']
    assert '<think>' not in tokenizer.decode(out['target_ids'])


def test_history_tool_response_is_not_supervised(tokenizer):
    e = example()
    e['messages'][1:1] = [
        {'role': 'user', 'content': 'Earlier request'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function',
         'function': {'name': 'lookup', 'arguments': {'key': 'x'}}}]},
        {'role': 'tool', 'content': 'HISTORY_SENTINEL'},
        {'role': 'assistant', 'content': 'Earlier answer'},
    ]
    out = format_example(e, tokenizer)
    assert 'HISTORY_SENTINEL' in tokenizer.decode(out['prompt_ids'])
    assert 'HISTORY_SENTINEL' not in tokenizer.decode(out['target_ids'])


@pytest.mark.parametrize('change', ['empty', 'role', 'long'])
def test_reject_invalid_target_and_length(tokenizer, change):
    e = copy.deepcopy(example())
    if change == 'empty': e['messages'][-1]['content'] = ''
    if change == 'role': e['messages'][-1]['role'] = 'user'
    if change == 'long': e['messages'][1]['content'] = 'long ' * 1000
    with pytest.raises(ValueError):
        format_example(e, tokenizer, max_seq_length=512)


def test_reject_prefix_misalignment(tokenizer):
    class Misaligned:
        eos_token_id = tokenizer.eos_token_id
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if kwargs.get('add_generation_prompt') else [9, 8, 7]
    with pytest.raises(ValueError, match='prefix'):
        format_example(example(), Misaligned())


def test_training_dataset_matches_upstream_batch_contract(tokenizer):
    from mlx_lm.tuner.trainer import iterate_batches
    data = training_dataset([example()], tokenizer)
    batch, lengths = next(iterate_batches(data, 1, 512))
    formatted = format_example(example(), tokenizer)
    assert lengths.tolist() == [[len(formatted['prompt_ids']), len(formatted['input_ids'])]]
    assert batch[0, :len(formatted['input_ids'])].tolist() == formatted['input_ids']
