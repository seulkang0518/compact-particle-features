"""Compare cached teacher/student geometry on validation or held-out audit episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compact_features.diagnostics import diagnose_features
from compact_features.model import collect_features
from compact_features.runtime import (build_model, digest, encoder_of, load_episode,
                                      read_index, write_json)


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "audit"], default="val")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--max-episodes", type=int, default=0, help="0 means all episodes in this split")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cache, index, cache_sha = read_index(args.cache)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if cache_sha != ckpt["cache_sha256"]:
        raise ValueError("Checkpoint/cache manifest mismatch")
    config, arm = ckpt["config"], ckpt["arm"]
    torch.set_num_threads(config["train"].get("threads", 4))
    model = build_model(arm, config, ckpt["teacher_channels"])
    model.load_state_dict(ckpt["model"])
    encoder = encoder_of(model, arm)
    del model
    device = torch.device(args.device)
    encoder = encoder.to(device).eval().requires_grad_(False)
    items = [e for e in index["episodes"] if e["split"] == args.split]
    if args.max_episodes:
        if args.max_episodes < 0:
            raise ValueError("max-episodes must be nonnegative")
        items = items[:args.max_episodes]
    if not items:
        raise ValueError("No episodes in selected split")
    rows = []
    with torch.no_grad():
        for item in items:
            ep = load_episode(cache, item, device)
            student = collect_features(encoder(ep["inputs"]), ep["positions"],
                                       target_sizes=index["feature_sizes"])
            diagnostics = diagnose_features(student, ep["teacher_features"], ep["queries"],
                                             config["train"]["temperatures"])
            rows.append({"file": item["file"], "class_id": item["class_id"], **diagnostics})
    mean_keys = rows[0]["mean"].keys()
    summary = {key: sum(r["mean"][key] for r in rows) / len(rows) for key in mean_keys}
    result = {"arm": arm, "step": ckpt["step"], "split": args.split,
              "synthetic": index.get("synthetic", False), "episodes": len(rows),
              "checkpoint_sha256": digest(args.checkpoint), "cache_sha256": cache_sha,
              "mean": summary, "per_episode": rows,
              "scope": "Cached feature geometry only; not actual Riesz-field input gradients or FID"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_episode"}, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    run()
