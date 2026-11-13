"""Frozen teacher map adapters. Real mode reuses a trusted existing MAE checkout."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
from .data import STAGES, file_hash


class SyntheticTeacher(nn.Module):
    """Fixed random convolutions for wiring checks, never quality experiments."""
    def __init__(self, width=4, seed=812):
        super().__init__()
        if width < 1:
            raise ValueError('Positive synthetic width required')
        self.channels = dict(zip(STAGES, [width, width, width*2, width*4, width*8]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers, previous = [], 4
            for i, channels in enumerate(self.channels.values()):
                layers.append(nn.Conv2d(previous, channels, 3, stride=2 if i >= 2 else 1, padding=1))
                previous = channels
            self.layers = nn.ModuleList(layers)
        self.requires_grad_(False)
        self.eval()
        self.provenance = {'type': 'synthetic_fixed_random_conv', 'seed': seed,
                           'warning': 'Pipeline validation only; not a pretrained image representation'}

    def forward(self, x):
        output = {}
        for stage, layer in zip(STAGES, self.layers):
            x = torch.nn.functional.silu(layer(x))
            output[stage] = x
        return output


class MAETeacher(nn.Module):
    def __init__(self, source_repo, teacher_root, teacher_name='mae_latent_640'):
        super().__init__()
        repo, root = Path(source_repo).resolve(), Path(teacher_root).resolve()
        candidates = [root, root/'models'/'mae'/'jax'/teacher_name, root/teacher_name]
        candidates = [p for p in candidates if (p/'metadata.json').is_file()]
        if len(candidates) != 1:
            raise ValueError('Provide the exact local teacher artifact directory or its HF cache root; no download is performed')
        artifact = candidates[0]
        metadata = json.loads((artifact/'metadata.json').read_text())
        model_config = metadata.get('model_config', {})
        if int(model_config.get('input_patch_size', 1)) != 1:
            raise ValueError('v1 supports teacher input_patch_size=1 only')
        if int(model_config.get('in_channels', 3)) != 4:
            raise ValueError('Teacher must be the four-channel latent MAE, not an RGB MAE')
        if not (repo/'models'/'mae_model.py').is_file():
            raise FileNotFoundError('Source checkout must contain models/mae_model.py')
        os.environ['DRIFT_COMPILE'] = '0'
        sys.path.insert(0, str(repo))
        from models.mae_model import build_feature_model_and_params
        model, state = build_feature_model_and_params(path=str(artifact))
        self.encoder = model.encoder
        # The reconstruction decoder and classifier are not moved to CUDA.
        del state, model
        gc.collect()
        self.requires_grad_(False)
        self.eval()
        width = int(model_config.get('base_channels', 64))
        self.channels = dict(zip(STAGES, [width, width, width*2, width*4, width*8]))
        try:
            revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).splitlines()
        except (subprocess.CalledProcessError, FileNotFoundError):
            revision, dirty = 'unavailable', []
        files = [artifact/'metadata.json'] + [p for name in ('ema_params.pt', 'ema_params.msgpack')
                                                  if (p := artifact/name).is_file()]
        self.provenance = {'type': 'frozen_latent_mae_encoder', 'teacher_name': teacher_name,
            'source_revision': revision, 'source_dirty_files': dirty,
            'source_files_sha256': {str(p.relative_to(repo)): file_hash(p) for p in
                [repo/'models'/'mae_model.py', repo/'models'/'hf.py', repo/'utils'/'param_convert.py']},
            'artifact_files_sha256': {p.name: file_hash(p) for p in files},
            'model_config': model_config, 'decoder_and_classifier_on_gpu': False}

    def forward(self, x):
        dtype = next(self.encoder.parameters()).dtype
        maps = self.encoder(x.permute(0, 2, 3, 1).to(dtype=dtype), train=False)
        if set(maps) != set(STAGES):
            raise ValueError('Teacher must produce exactly conv1 and layer1..layer4 maps')
        return maps
