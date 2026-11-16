import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tooltune.input_guard import check_files, freeze, verify


class InputGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.asset = self.root / 'weights.bin'
        self.asset.write_bytes(b'original')
        self.files = {'weights.bin': hashlib.sha256(b'original').hexdigest()}
        self.guard = self.root / 'guard.json'

    def test_asset_changed_with_manifest_unchanged(self):
        check_files(self.root, self.files)
        self.asset.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            check_files(self.root, self.files)

    def test_missing_asset(self):
        self.asset.unlink()
        with self.assertRaises(FileNotFoundError):
            check_files(self.root, self.files)

    def test_config_drift_rejected(self):
        frozen = {'adapter_config_sha256': 'scale16', 'files': self.files}
        freeze(self.guard, frozen)
        verify(self.guard, frozen)
        with self.assertRaises(ValueError):
            verify(self.guard, dict(frozen, adapter_config_sha256='scale0'))

    def test_freeze_never_overwrites(self):
        freeze(self.guard, {'version': 1})
        with self.assertRaises(FileExistsError):
            freeze(self.guard, {'version': 2})
        self.assertEqual(json.loads(self.guard.read_text()), {'version': 1})

    def test_missing_guard_does_not_auto_accept(self):
        with self.assertRaises(FileNotFoundError):
            verify(self.guard, {})


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        from tooltune.input_guard import sha
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        def write(name, value):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return path
        self.write = write
        asset = write('model/weights', {'weights': 1})
        adapter = write('adapter/adapters.safetensors', {'weights': 2})
        self.config = {'fine_tune_type': 'lora', 'num_layers': 4,
                       'lora_parameters': {'rank': 8, 'scale': 16.0, 'dropout': 0.0,
                                           'keys': ['self_attn.q_proj', 'self_attn.v_proj']}}
        write('adapter/adapter_config.json', self.config)
        data = write('data/train.json', {'example': 1})
        contract = write('data/manifests/v2-balanced/contract.json',
                         {'files': {'data/train.json': sha(data)}})
        model = write('artifacts/model_manifest.json',
                      {'files': [{'name': 'weights', 'sha256': sha(asset)}]})
        (self.root/'artifacts/model_path.txt').write_text(str(self.root/'model'))
        write('reports/evidence/t4/frozen-contract.json',
              {'source_sha256': {}, 'selected_adapter': 'adapter',
               'selected_adapter_sha256': sha(adapter)})
        write('reports/evidence/t3/environment-contract.json',
              {'files': {}, 'data_contract_sha256': sha(contract),
               'model_manifest_sha256': sha(model)})
        write('src/tooltune/train.py', 'original source')

    def test_real_adapter_scale_change_rejected(self):
        from tooltune.input_guard import snapshot
        snapshot(self.root)
        self.config['lora_parameters']['scale'] = 0
        self.write('adapter/adapter_config.json', self.config)
        with self.assertRaisesRegex(ValueError, 'adapter configuration'):
            snapshot(self.root)

    def test_actual_assets_checked(self):
        from tooltune.input_guard import snapshot
        for name in ('data/train.json', 'model/weights'):
            path = self.root/name
            original = path.read_bytes()
            with self.subTest(path=name):
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'asset changed'):
                    snapshot(self.root)
                path.write_bytes(original)

    def test_training_source_drift_rejected(self):
        from tooltune.input_guard import snapshot
        guard = self.root/'guard.json'
        freeze(guard, snapshot(self.root))
        self.write('src/tooltune/train.py', 'modified source')
        with self.assertRaises(ValueError):
            verify(guard, snapshot(self.root))
