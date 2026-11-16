"""Additional pre-execution checks; historical experiment contracts stay intact."""
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_files(root, files):
    for name, expected in files.items():
        if sha(Path(root) / name) != expected:
            raise ValueError('asset changed: ' + name)


def freeze(path, snapshot):
    with Path(path).open('x') as stream:
        json.dump(snapshot, stream, indent=2)
        stream.write('\n')


def verify(path, snapshot):
    if json.loads(Path(path).read_text()) != snapshot:
        raise ValueError('input guard changed; preserve this run and investigate')


def snapshot(root):
    root = Path(root)
    frozen = json.loads((root/'reports/evidence/t4/frozen-contract.json').read_text())
    prior = json.loads((root/'reports/evidence/t3/environment-contract.json').read_text())
    check_files(root, frozen['source_sha256'])
    check_files(root, prior['files'])
    check_files(root, {
        'data/manifests/v2-balanced/contract.json': prior['data_contract_sha256'],
        'artifacts/model_manifest.json': prior['model_manifest_sha256'],
    })
    data = json.loads((root/'data/manifests/v2-balanced/contract.json').read_text())
    check_files(root, data['files'])
    model = json.loads((root/'artifacts/model_manifest.json').read_text())
    model_path = Path((root/'artifacts/model_path.txt').read_text().strip())
    check_files(model_path, {item['name']: item['sha256'] for item in model['files']})
    adapter = root/frozen['selected_adapter']
    check_files(adapter, {'adapters.safetensors': frozen['selected_adapter_sha256']})
    config = json.loads((adapter/'adapter_config.json').read_text())
    expected = {'fine_tune_type': 'lora', 'num_layers': 4,
                'lora_parameters': {'rank': 8, 'scale': 16.0, 'dropout': 0.0,
                                    'keys': ['self_attn.q_proj', 'self_attn.v_proj']}}
    if config != expected:
        raise ValueError('adapter configuration differs from documented U200 training')
    sources = sorted((root/'src/tooltune').glob('*.py')) + sorted((root/'scripts').glob('*.py'))
    return {'version': 1, 'scope': 'existing t3-r8-v1 / t4-test-v1 assets',
            'model_path': str(model_path.resolve()),
            'adapter_config_sha256': sha(adapter/'adapter_config.json'),
            'data_contract_sha256': sha(root/'data/manifests/v2-balanced/contract.json'),
            'model_manifest_sha256': sha(root/'artifacts/model_manifest.json'),
            'files': {str(path.relative_to(root)): sha(path) for path in sources}}
