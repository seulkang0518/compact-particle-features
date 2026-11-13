"""Held-out geometry diagnostics, not a test of generator-gradient fidelity."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch


EPS = 1e-12


def _distances(features: torch.Tensor, queries: int):
    # Channel counts may differ between encoders. All comparisons below use
    # within-encoder distances, standardized by that encoder's own positive scale.
    features = features.detach().to(device="cpu", dtype=torch.float64)
    positive = torch.cdist(features[:, :queries], features[:, queries:])
    negative = torch.cdist(features[:, :queries], features[:, :queries])
    valid_negative = ~torch.eye(queries, dtype=torch.bool)
    scale = positive.mean(dim=(1, 2), keepdim=True).clamp_min(EPS)
    return features, positive / scale, negative / scale, valid_negative, scale


def _overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    left = torch.argsort(a, dim=-1, stable=True)[..., :k]
    right = torch.argsort(b, dim=-1, stable=True)[..., :k]
    return float((left.unsqueeze(-1) == right.unsqueeze(-2)).any(-1).double().mean())


def _effective_neighbors(distance: torch.Tensor, temperature: float,
                         valid: torch.Tensor | None = None) -> float:
    logits = -distance / temperature
    if valid is not None:
        logits = logits.masked_fill(~valid, -torch.inf)
    probability = torch.softmax(logits, dim=-1)
    entropy = -(probability * probability.clamp_min(EPS).log()).sum(-1)
    return float(entropy.exp().mean())


def _spread(features: torch.Tensor, queries: int, scale: torch.Tensor):
    generated, real = features[:, :queries], features[:, queries:]
    gv = generated.var(dim=1, unbiased=False).sum(-1)
    rv = real.var(dim=1, unbiased=False).sum(-1)
    within = (gv + rv) / 2
    energy = (generated.square().sum(-1).mean(1) + real.square().sum(-1).mean(1)) / 2
    total_variance = features.var(dim=1, unbiased=False).sum(-1)
    all_energy = features.square().sum(-1).mean(1)
    normalization = scale.squeeze(-1).squeeze(-1).square().clamp_min(EPS)
    return {
        "within_group_variance_fraction": float((within / energy.clamp_min(EPS)).mean()),
        "generated_spread_normalized": float((gv / normalization).mean()),
        "real_spread_normalized": float((rv / normalization).mean()),
        # Fractions over positions. These numerical checks cannot certify
        # semantics, nor do they imply diverse generated images.
        "collapsed_position_fraction": float((total_variance <= EPS * all_energy.clamp_min(1.)).double().mean()),
        "collapsed_generated_position_fraction": float((gv <= EPS * all_energy.clamp_min(1.)).double().mean()),
    }


def diagnose_features(student: Mapping[str, torch.Tensor],
                      teacher: Mapping[str, torch.Tensor], queries: int,
                      temperatures: Sequence[float]) -> dict:
    """Compare cached branch geometry; return JSON-compatible finite scalars.

    Each branch is [positions, samples, channels], with generated samples first.
    Generated self-pairs are excluded from neighborhoods and negative distances.
    Effective neighbors = exp(Shannon entropy) of Laplace-style affinities at tau.
    NN overlap uses the combined generated/real bank and k=min(5, samples-1).
    Zero-distance ties use stable index ordering; overlap for collapsed features
    must therefore be interpreted together with the collapse diagnostics.
    """
    if not student or set(student) != set(teacher):
        raise ValueError("Student and teacher must contain the same nonempty branch names")
    if isinstance(queries, bool) or not isinstance(queries, int) or queries < 2:
        raise ValueError("Need at least two generated queries")
    taus = [float(value) for value in temperatures]
    if not taus or any(not math.isfinite(tau) or tau <= 0 for tau in taus):
        raise ValueError("Need finite positive temperatures")
    if len(set(taus)) != len(taus):
        raise ValueError("Temperatures must be distinct")
    branches = {}
    for name in sorted(student):
        s, t = student[name], teacher[name]
        if (not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor)
                or s.ndim != 3 or t.ndim != 3 or s.shape[:2] != t.shape[:2]
                or min(s.shape) < 1 or min(t.shape) < 1 or queries >= s.shape[1]):
            raise ValueError(f"Invalid or mismatched [P,N,C] branch {name}")
        if not s.is_floating_point() or not t.is_floating_point():
            raise ValueError(f"Feature branch {name} must be floating point")
        if not torch.isfinite(s).all() or not torch.isfinite(t).all():
            raise ValueError(f"Nonfinite feature branch {name}")
        sf, sp, sn, mask, ss = _distances(s, queries)
        tf, tp, tn, _, ts = _distances(t, queries)
        snegative, tnegative = sn[:, mask], tn[:, mask]
        positive_mse = (sp - tp).square().mean()
        negative_mse = (snegative - tnegative).square().mean()
        # Equal weight for the positive and negative populations, irrespective
        # of their different counts (and equal weight for positions/branches).
        result = {
            "normalized_distance_rmse": float(((positive_mse + negative_mse) / 2).sqrt()),
            "positive_normalized_distance_rmse": float(positive_mse.sqrt()),
            "negative_normalized_distance_rmse": float(negative_mse.sqrt()),
        }
        combined_s = torch.cat([sn.masked_fill(~mask, torch.inf), sp], dim=-1)
        combined_t = torch.cat([tn.masked_fill(~mask, torch.inf), tp], dim=-1)
        k = min(5, s.shape[1] - 1)
        result["neighbor_overlap_at_k"] = _overlap(combined_s, combined_t, k)
        result["neighbor_k"] = float(k)
        sspread, tspread = _spread(sf, queries, ss), _spread(tf, queries, ts)
        for prefix, values in (("student", sspread), ("teacher", tspread)):
            result.update({f"{prefix}_{key}": value for key, value in values.items()})
        for key in ("within_group_variance_fraction", "generated_spread_normalized", "real_spread_normalized"):
            result[f"student_over_teacher_{key}"] = sspread[key] / max(tspread[key], EPS)
        for tau in taus:
            suffix = f"tau_{tau:g}"
            for prefix, pos, neg in (("student", sp, sn), ("teacher", tp, tn)):
                result[f"{prefix}_positive_effective_neighbors_{suffix}"] = _effective_neighbors(pos, tau)
                result[f"{prefix}_negative_effective_neighbors_{suffix}"] = _effective_neighbors(neg, tau, mask)
        if any(not math.isfinite(value) for value in result.values()):
            raise ValueError(f"Nonfinite geometry diagnostic for {name}")
        branches[name] = result
    mean = {key: sum(branch[key] for branch in branches.values()) / len(branches)
            for key in next(iter(branches.values()))}
    return {"branches": branches, "mean": mean}
