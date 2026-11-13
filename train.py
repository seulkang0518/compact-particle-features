"""Train a compact encoder from a fixed episode cache (one GPU or torchrun)."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from compact_features.model import collect_features, masked_reconstruction_loss
from compact_features.objectives import feature_mse, geometry_loss
from compact_features.runtime import (build_model, encoder_of, load_episode, read_index,
                                      save_checkpoint, source_digest, write_json)


def objective(model, arm, ep, sizes, config):
    if arm == "reconstruction":
        return masked_reconstruction_loss(
            model, ep["inputs"], mask_ratio=config["mask_ratio"],
            patch_size=config["mask_patch_size"])
    features = collect_features(model(ep["inputs"]), ep["positions"], target_sizes=sizes)
    if arm == "feature":
        return feature_mse(features, ep["teacher_features"])
    return geometry_loss(features, ep["teacher_features"], ep["queries"],
                         config["temperatures"], config["distance_weight"])[0]


def episode_schedule(count, global_episodes, step, seed):
    """Stateless global schedule; rank count only changes the partition."""
    generator = torch.Generator().manual_seed(seed + 1000003 * step)
    return torch.randint(count, (global_episodes,), generator=generator).tolist()


def learning_rate(step, horizon, config):
    warmup = min(config["warmup_steps"], horizon)
    if warmup and step < warmup:
        return config["learning_rate"] * (step + 1) / warmup
    fraction = (step - warmup) / max(1, horizon - warmup - 1)
    return config["learning_rate"] * 0.5 * (1 + math.cos(math.pi * min(1, fraction)))


def validate(model, arm, items, cache, index, config, device):
    model.eval()
    losses = []
    # Fixed validation mask randomness, isolated from training randomness.
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        for i, item in enumerate(items[:config["validation_episodes"]]):
            torch.manual_seed(900000 + i)
            ep = load_episode(cache, item, device)
            loss = objective(model, arm, ep, index["feature_sizes"], config)
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--config", default="configs/pilot.yaml")
    p.add_argument("--arm", choices=["reconstruction", "feature", "geometry"], required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--steps", type=int, help="Override the total optimization/schedule horizon")
    p.add_argument("--stop-after", type=int, help="Stop early without changing the schedule; for resumability checks")
    p.add_argument("--resume", help="Trusted local checkpoint from this arm/cache/config")
    args = p.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    tc = config["train"]
    if args.steps is not None:
        tc["steps"] = args.steps
    horizon = int(tc["steps"])
    if horizon < 1 or tc["global_episodes"] < 1:
        raise ValueError("steps and global_episodes must be positive")
    if min(tc["learning_rate"], tc["max_grad_norm"]) <= 0 or tc["warmup_steps"] < 0:
        raise ValueError("Invalid optimizer configuration")
    if min(tc["log_every"], tc["eval_every"], tc["save_every"], tc["validation_episodes"]) < 1:
        raise ValueError("Logging, evaluation, save intervals and validation count must be positive")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if tc["global_episodes"] % world:
        raise ValueError("global_episodes must be divisible by WORLD_SIZE")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install a driver-compatible PyTorch or use --device cpu")
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(tc.get("threads", 4))
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    run_lock = None
    try:
        cache, index, cache_sha = read_index(args.cache)
        code_sha = source_digest()
        train_items = [e for e in index["episodes"] if e["split"] == "train"]
        val_items = [e for e in index["episodes"] if e["split"] == "val"]
        if not train_items or not val_items:
            raise ValueError("Both train and val episodes are required")
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            run_lock = (workdir / ".run.lock").open("a")
            try:
                fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another training process owns this workdir; choose a new directory") from error
        if rank == 0 and (workdir / "latest.pt").exists() and not args.resume:
            raise ValueError("Workdir already contains a checkpoint; use --resume or a new workdir")
        torch.manual_seed(tc["seed"])
        model = build_model(args.arm, config, index["teacher_channels"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
        completed, best = 0, float("inf")
        if args.resume:
            payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            if payload["arm"] != args.arm or payload["cache_sha256"] != cache_sha or payload["config"] != config:
                raise ValueError("Resume requires the same arm, cache, and full configuration/schedule")
            if payload.get("source_sha256") != code_sha:
                raise ValueError("Resume code fingerprint differs; resume with the original source revision")
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            completed, best = payload["step"], payload["best_validation_loss"]
            if completed > horizon:
                raise ValueError("Checkpoint is beyond the requested horizon")
        raw_model = model
        if world > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
        if rank == 0:
            manifest = {"arm": args.arm, "config": config, "cache": str(cache),
                        "cache_sha256": cache_sha, "synthetic": index.get("synthetic", False),
                        "source_sha256": code_sha,
                        "device": str(device), "world_size": world,
                        "parameters": sum(t.numel() for t in raw_model.parameters()),
                        "retained_encoder_parameters": sum(t.numel() for t in encoder_of(raw_model, args.arm).parameters()),
                        "torch": torch.__version__, "completed_steps": completed,
                        "checkpoint_selection": "best own validation objective; final latest.pt also retained"}
            write_json(workdir / "manifest.json", manifest)
            print(json.dumps(manifest), flush=True)
        local_episodes = tc["global_episodes"] // world
        start = time.perf_counter()
        last = min(horizon, args.stop_after) if args.stop_after is not None else horizon
        if last < completed:
            raise ValueError("stop-after precedes the resume step")
        for step in range(completed, last):
            model.train()
            lr = learning_rate(step, horizon, tc)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            schedule = episode_schedule(len(train_items), tc["global_episodes"], step, tc["seed"])
            loss_sum = torch.zeros((), device=device)
            for k in range(local_episodes):
                offset = rank * local_episodes + k
                # Masks depend on global position, not number of ranks.
                torch.manual_seed(tc["seed"] + step * tc["global_episodes"] + offset)
                ep = load_episode(cache, train_items[schedule[offset]], device)
                sync = model.no_sync() if world > 1 and k + 1 < local_episodes else contextlib.nullcontext()
                with sync:
                    loss = objective(model, args.arm, ep, index["feature_sizes"], tc)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
                    (loss / local_episodes).backward()
                loss_sum += loss.detach() / local_episodes
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            if world > 1:
                dist.all_reduce(loss_sum)
                loss_sum /= world
            current = step + 1
            if rank == 0 and (current % tc["log_every"] == 0 or current == last):
                row = {"step": current, "loss": float(loss_sum), "gradient_norm_before_clipping": float(norm),
                       "learning_rate": lr, "elapsed_seconds_this_launch": time.perf_counter() - start}
                print(json.dumps(row), flush=True)
                with (workdir / "train.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
            need_eval = current % tc["eval_every"] == 0 or current == last
            need_save = current % tc["save_every"] == 0 or current == last or need_eval
            if world > 1 and need_save:
                dist.barrier()
            if rank == 0 and need_save:
                improved = False
                if need_eval:
                    val_loss = validate(raw_model, args.arm, val_items, cache, index, tc, device)
                    if not math.isfinite(val_loss):
                        raise FloatingPointError("Nonfinite validation objective")
                    improved = val_loss < best
                    best = min(best, val_loss)
                    row = {"step": current, "validation_loss": val_loss, "best_validation_loss": best}
                    print(json.dumps(row), flush=True)
                    with (workdir / "train.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                payload = {"version": 1, "arm": args.arm, "step": current, "config": config,
                           "source_sha256": code_sha, "training_world_size": world,
                           "cache_sha256": cache_sha, "synthetic": index.get("synthetic", False),
                           "teacher_channels": index["teacher_channels"], "model": raw_model.state_dict(),
                           "optimizer": optimizer.state_dict(), "best_validation_loss": best}
                save_checkpoint(workdir / "latest.pt", payload)
                if improved:
                    save_checkpoint(workdir / "best.pt", payload)
            if world > 1 and need_save:
                dist.barrier()
        if rank == 0:
            print(f"Finished at step {last}. Checkpoint: {workdir / 'latest.pt'}", flush=True)
    finally:
        if run_lock is not None:
            run_lock.close()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run()
