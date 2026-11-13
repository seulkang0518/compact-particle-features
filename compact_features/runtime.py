"""Trusted local cache/checkpoint loading and atomic research-run outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .model import CompactEncoder, ReconstructionModel


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_digest():
    """Fingerprint executable pilot code, independent of absolute checkout path."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted(list((root / "compact_features").glob("*.py")) +
                   list(root.glob("*.py")))
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode() + b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def read_index(cache):
    cache = Path(cache).resolve()
    path = cache / "index.json"
    index = json.loads(path.read_text())
    if index.get("version") != 1 or index.get("complete") is not True:
        raise ValueError("Incomplete or unsupported cache")
    if not index.get("episodes"):
        raise ValueError("Cache has no episodes")
    for item in index["episodes"]:
        ep = (cache / item["file"]).resolve()
        if not ep.is_relative_to(cache) or not ep.is_file():
            raise ValueError(f"Missing or unsafe episode path: {item['file']}")
    return cache, index, digest(path)


def load_episode(cache, item, device):
    path = Path(cache) / item["file"]
    if item.get("sha256") and digest(path) != item["sha256"]:
        raise ValueError(f"Episode checksum mismatch: {path}")
    # Only load caches produced locally or by a trusted collaborator.
    ep = torch.load(path, map_location="cpu", weights_only=False)
    result = dict(ep)
    result["inputs"] = ep["inputs"].to(device=device, dtype=torch.float32)
    result["positions"] = {k: v.to(device) for k, v in ep["positions"].items()}
    result["teacher_features"] = {
        k: v.to(device=device, dtype=torch.float32) for k, v in ep["teacher_features"].items()
    }
    if not 1 < ep["queries"] < len(ep["inputs"]):
        raise ValueError("Each episode needs at least two queries and one reference")
    return result


def build_model(arm, config, teacher_channels):
    opts = config["model"]
    if arm == "reconstruction":
        return ReconstructionModel(width=opts["width"], blocks=opts["blocks"])
    if arm not in {"feature", "geometry"}:
        raise ValueError(f"Unknown arm {arm}")
    return CompactEncoder(width=opts["width"], blocks=opts["blocks"],
                          teacher_channels=teacher_channels)


def encoder_of(model, arm):
    model = model.module if hasattr(model, "module") else model
    return model.encoder if arm == "reconstruction" else model


def save_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
