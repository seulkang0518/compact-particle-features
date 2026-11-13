"""Adapter routing tests; real MAE-640 weights are not bundled in this suite."""
import json
import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from compact_features.teacher import MAETeacher


@pytest.mark.parametrize('checkpoint_format', ['jax', 'torch'])
def test_local_artifact_uses_conversion_before_strict_load(tmp_path, monkeypatch, checkpoint_format):
    repo = tmp_path / 'source'
    for name in ('models/mae_model.py', 'models/hf.py', 'utils/param_convert.py'):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# local fixture\n')
    artifact = tmp_path / 'artifact'
    artifact.mkdir()
    (artifact / 'metadata.json').write_text(json.dumps({
        'model_config': {'in_channels': 4, 'base_channels': 2, 'input_patch_size': 1}}))
    (artifact / ('ema_params.msgpack' if checkpoint_format == 'jax' else 'ema_params.pt')).write_bytes(b'fixture')

    class FixtureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Conv2d(4, 2, 1, bias=False)
            self.decoder = nn.Linear(2, 2)

    template = FixtureModel()
    expected = {k: torch.full_like(v, .25) for k, v in template.state_dict().items()}
    raw = {'nested_jax_fixture': True} if checkpoint_format == 'jax' else expected
    calls = []

    def convert(tree, target):
        assert tree is raw
        assert set(target) == set(expected)
        calls.append('convert')
        return expected

    def load_or_convert(model, tree, converter):
        calls.append('load_or_convert')
        return converter(tree, model.state_dict()) if checkpoint_format == 'jax' else tree

    for name in ('models', 'utils'):
        module = ModuleType(name)
        module.__path__ = [str(repo / name)]
        monkeypatch.setitem(sys.modules, name, module)
    modules = {
        'models.mae_model': {'_mae_from_metadata': lambda meta: FixtureModel()},
        'models.hf': {'load_jax_ema_params': lambda path: raw,
                      '_load_torch_or_convert': load_or_convert},
        'utils.param_convert': {'convert_mae_tree_to_state_dict': convert},
    }
    for name, exports in modules.items():
        module = ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    old_path = list(sys.path)
    monkeypatch.setattr(sys, 'path', old_path.copy())
    teacher = MAETeacher(repo, artifact)
    assert calls == (['load_or_convert', 'convert'] if checkpoint_format == 'jax' else ['load_or_convert'])
    torch.testing.assert_close(teacher.encoder.weight, expected['encoder.weight'])
    assert not hasattr(teacher, 'decoder')
    assert all(not p.requires_grad for p in teacher.parameters())
