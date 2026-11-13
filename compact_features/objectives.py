"""Teacher-guided feature/geometry objectives; no raw-MMD maximization here."""
import torch
from torch.nn import functional as F


def _branches(student, teacher, *, equal_channels=False):
    if not student or set(student) != set(teacher):
        raise ValueError('Student and teacher feature branches must match')
    for key in student:
        s, t = student[key], teacher[key]
        if s.ndim != 3 or t.ndim != 3 or s.shape[:2] != t.shape[:2]:
            raise ValueError(f'Expected matching [position,sample,channel] branches: {key}')
        if equal_channels and s.shape != t.shape:
            raise ValueError(f'Feature MSE needs retained channel adapters: {key}')
        t = t.detach().to(device=s.device, dtype=torch.float32)
        s = s.float()
        if not torch.isfinite(s).all() or not torch.isfinite(t).all():
            raise ValueError(f'Nonfinite features: {key}')
        yield key, s, t


def feature_mse(student, teacher):
    """Mean branch MSE under a shared, detached TEACHER branch RMS scale.

    The RMS is over all sampled positions/samples/channels in that branch. The
    same scalar divides student and teacher values; independently normalizing
    each would erase amplitude differences. Retained adapters are required.
    """
    losses = []
    for _, s, t in _branches(student, teacher, equal_channels=True):
        scale = (t.square().mean() + 1e-6).sqrt()
        losses.append(((s - t) / scale).square().mean())
    return torch.stack(losses).mean()


def _normalized_distances(features, queries):
    n = features.shape[1]
    if not isinstance(queries, int) or queries < 2 or queries >= n:
        raise ValueError('Need at least two generated queries and one real reference')
    q, r = features[:, :queries], features[:, queries:]
    positive = torch.cdist(q, r, p=2)
    scale = positive.mean((1, 2), keepdim=True)
    if not torch.isfinite(scale).all() or torch.any(scale.detach() <= 1e-8):
        raise ValueError('Degenerate query-reference distance scale')
    all_negative = torch.cdist(q, q, p=2)
    other = ~torch.eye(queries, dtype=torch.bool, device=q.device)
    negative = all_negative[:, other].reshape(features.shape[0], queries, queries - 1)
    # Do not detach student scale: its effect is part of encoder pretraining.
    return positive / scale, negative / scale, scale


def _kl_and_neighbors(student_distance, teacher_distance, temperature):
    log_student = (-student_distance / temperature).log_softmax(-1)
    log_teacher = (-teacher_distance / temperature).log_softmax(-1)
    kl = F.kl_div(log_student, log_teacher, reduction='none', log_target=True).sum(-1).mean()
    logp = log_student.detach()
    effective = (-(logp.exp() * logp).sum(-1)).exp().mean()
    return kl, effective


def geometry_loss(student, teacher, queries, temperatures=(.2,.05,.02), distance_weight=.1):
    """KL(teacher affinity || student affinity), plus normalized-distance Huber.

    Each branch averages its positions, query rows, positive/negative terms and
    temperatures; the outer average gives every branch equal weight. Negative
    self-pairs are removed before softmax, including the two-query edge case.
    This trains affinities, not a proof of matching stopped-field input gradients.
    Returns a scalar loss plus detached scalar-tensor diagnostics.
    """
    temperatures = tuple(float(value) for value in temperatures)
    if not temperatures or any(not torch.isfinite(torch.tensor(t)) or t <= 0 for t in temperatures):
        raise ValueError('Temperatures must be positive and finite')
    if not 0 <= distance_weight < float('inf'):
        raise ValueError('distance_weight must be nonnegative and finite')
    rows = []
    for _, s, t in _branches(student, teacher):
        sp, sn, ss = _normalized_distances(s, queries)
        tp, tn, ts = _normalized_distances(t, queries)
        positives, negatives, pn, nn = [], [], [], []
        for tau in temperatures:
            pos, pos_neighbors = _kl_and_neighbors(sp, tp, tau)
            neg, neg_neighbors = _kl_and_neighbors(sn, tn, tau)
            positives.append(pos); negatives.append(neg)
            pn.append(pos_neighbors); nn.append(neg_neighbors)
        kl = .5 * (torch.stack(positives).mean() + torch.stack(negatives).mean())
        distance = .5 * (F.smooth_l1_loss(sp, tp) + F.smooth_l1_loss(sn, tn))
        rows.append((kl, distance, torch.stack(pn).mean(), torch.stack(nn).mean(), ss.mean(), ts.mean()))
    values = torch.stack([torch.stack(row) for row in rows]).mean(0)
    loss = values[0] + distance_weight * values[1]
    names = ('affinity_kl', 'distance_error', 'effective_neighbors_positive',
             'effective_neighbors_negative', 'student_distance_scale', 'teacher_distance_scale')
    return loss, {name: value.detach() for name, value in zip(names, values)}
