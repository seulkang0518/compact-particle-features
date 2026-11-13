"""Measure a frozen STUDENT encoder's forward plus generic input-backward cost."""
from __future__ import annotations

import argparse
import time

import torch

from compact_features.model import collect_features
from compact_features.runtime import (build_model, encoder_of, load_episode, read_index, write_json)


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("Invalid timing repetitions")
    cache, index, sha = read_index(args.cache)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt["cache_sha256"] != sha:
        raise ValueError("Cache mismatch")
    torch.set_num_threads(ckpt["config"]["train"].get("threads", 4))
    model = build_model(ckpt["arm"], ckpt["config"], ckpt["teacher_channels"])
    model.load_state_dict(ckpt["model"])
    encoder = encoder_of(model, ckpt["arm"])
    del model  # reconstruction decoder must not reside on the measured device
    device = torch.device(args.device)
    encoder = encoder.eval().requires_grad_(False).to(device)
    item = next(e for e in index["episodes"] if e["split"] == "val")
    ep = load_episode(cache, item, device)
    # Teacher targets are unnecessary for this timing and should not occupy GPU.
    ep.pop("teacher_features")
    def update():
        x = ep["inputs"].detach().requires_grad_(True)
        features = collect_features(encoder(x), ep["positions"], target_sizes=index["feature_sizes"])
        loss = sum(f.square().mean() for f in features.values()) / len(features)
        torch.autograd.grad(loss, x)
    for _ in range(args.warmup):
        update()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.repeats):
        update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {"arm": ckpt["arm"], "synthetic": index.get("synthetic", False),
              "device": str(device), "torch": torch.__version__,
              "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
              "samples_per_episode": len(ep["inputs"]), "queries": ep["queries"],
              "encoder_parameters_including_retained_heads": sum(p.numel() for p in encoder.parameters()),
              "mean_forward_input_backward_ms": 1000 * (time.perf_counter() - start) / args.repeats,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "warmup": args.warmup, "repeats": args.repeats,
              "scope": "Student only, FP32, generic feature-energy backward, cached sampled positions; no teacher speedup ratio or end-to-end claim"}
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    run()
