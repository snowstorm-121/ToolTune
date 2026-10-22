from pathlib import Path

from transformers import AutoTokenizer

TEMPLATE = Path(__file__).resolve().parents[2] / 'configs/chat_template.jinja'


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                              trust_remote_code=False)
    # Official Instruct-2507 template; retain community snapshot without mutation.
    tokenizer.chat_template = TEMPLATE.read_text()
    return tokenizer


def format_example(record, tokenizer, max_seq_length=512):
    messages = record['messages']
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('target must be the last assistant message')
    target = messages[-1]
    if not target.get('content', '').strip() and not target.get('tool_calls'):
        raise ValueError('empty target')
    tools = record.get('tools') or None
    prompt = tokenizer.apply_chat_template(messages[:-1], tools=tools,
        add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(messages, tools=tools, return_dict=False)
    if full[:len(prompt)] != prompt:
        raise ValueError('template prefix misalignment')
    if len(full) <= len(prompt):
        raise ValueError('zero supervised tokens')
    if len(full) > max_seq_length:
        raise ValueError(f'long sample: {len(full)} > {max_seq_length}')
    return {'prompt_ids': prompt, 'target_ids': full[len(prompt):],
            'input_ids': full, 'loss_mask': [0] * len(prompt) + [1] * (len(full)-len(prompt))}
