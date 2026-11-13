"""Portable latent packets and deterministic same-class comparison episodes."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

STAGES = ('conv1', 'layer1', 'layer2', 'layer3', 'layer4')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity_split(identity, seed=42):
    """Identity, not checkpoint or augmentation, determines the held-out split."""
    value = int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest()[:16], 16) / 2**64
    return 'train' if value < .7 else 'val' if value < .85 else 'audit'


def check_capacity(output, estimated_bytes, max_gb):
    """Bound materialized arrays and leave working room; never delete old outputs."""
    if not np.isfinite(max_gb) or max_gb <= 0:
        raise ValueError('Positive --max-*-gb required')
    required = int(estimated_bytes * 1.15) + 16 * 1024**2
    if required > max_gb * 1024**3:
        raise ValueError(f'Estimated output with margin {required / 1024**3:.2f} GiB exceeds cap {max_gb:.2f} GiB')
    parent = Path(output).resolve().parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < required + 128 * 1024**2:
        raise OSError('Insufficient free space for output and safety margin')
    return required


def verify_latents(array, labels, name):
    if array.ndim != 4 or tuple(array.shape[1:]) != (32, 32, 4):
        raise ValueError(f'{name}: expected already-scaled BHWC [N,32,32,4], got {array.shape}')
    if labels.shape != (array.shape[0],) or labels.dtype.kind not in 'iu':
        raise ValueError(f'{name}: labels must be one integer per latent')
    if array.shape[0] == 0 or np.any((labels < 0) | (labels >= 1000)):
        raise ValueError(f'{name}: empty data or labels outside ImageNet indices 0..999')
    if array.dtype.kind != 'f':
        raise ValueError(f'{name}: floating point latent array required')


def collect_cached_features(maps, positions):
    """Match model.collect_features: [position, image, channel], population std."""
    output = {}
    for stage in STAGES:
        value = maps[stage].float()
        if value.ndim != 4:
            raise ValueError('Teacher maps must have BCHW shape')
        flat = value.flatten(2)
        index = positions[stage].to(value.device)
        output[f'{stage}/spatial'] = flat.index_select(2, index).permute(2, 0, 1).contiguous()
        output[f'{stage}/mean'] = flat.mean(-1).unsqueeze(0)
        output[f'{stage}/std'] = (flat.var(-1, unbiased=False).clamp_min(0) + 1e-6).sqrt().unsqueeze(0)
    return output


def spatial_position_pools(sizes, seed=42):
    """Hold out local spatial indices; global mean/std still summarize all pixels."""
    rng = np.random.default_rng(seed + 82031)
    pools = {}
    for stage in STAGES:
        count = int(np.prod(sizes[stage]))
        if count < 2:
            raise ValueError('Spatial auditing needs at least two positions in every teacher stage')
        permutation = rng.permutation(count)
        halfway = count // 2
        pools[stage] = {'development': permutation[:halfway].tolist(),
                        'audit': permutation[halfway:].tolist()}
    return pools


class Packet:
    def __init__(self, root, verify=True):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('version') != 1 or not self.manifest.get('complete'):
            raise ValueError('Packet is incomplete or has an unsupported version')
        if verify:
            for name, expected in self.manifest['files_sha256'].items():
                if file_hash(self.root / name) != expected:
                    raise ValueError(f'Packet identity changed: {name}')
        self.inputs = np.load(self.root / 'inputs.npy', mmap_mode='r', allow_pickle=False)
        self.records = json.loads((self.root / 'records.json').read_text())
        if self.inputs.shape != (len(self.records), 4, 32, 32):
            raise ValueError('Packet array and record dimensions differ')
        self.pools = {split: {} for split in ('train', 'val', 'audit')}
        for index, record in enumerate(self.records):
            pool = self.pools[record['split']].setdefault(int(record['class_id']), {'real': [], 'generated': []})
            pool[record['kind']].append(index)

    def eligible(self, split, queries, references):
        return sorted(c for c, pool in self.pools[split].items()
                      if len(pool['generated']) >= queries and len(pool['real']) >= references)

    def episode(self, split, queries, references, seed):
        classes = self.eligible(split, queries, references)
        if not classes:
            raise ValueError(f'No {split} class has {queries} generated and {references} real examples; prepare a larger packet')
        rng = np.random.default_rng(seed)
        class_id = int(rng.choice(classes))
        pool = self.pools[split][class_id]
        selected = np.concatenate([rng.choice(pool['generated'], queries, replace=False),
                                   rng.choice(pool['real'], references, replace=False)])
        inputs = torch.from_numpy(np.array(self.inputs[selected], dtype=np.float32, copy=True))
        return inputs, class_id, [self.records[int(i)] for i in selected]
