"""HDSF's original pooled-L1 simplex solver and half-day partition.

Ported from PowerData_Prediction/attainment_fusion.py (83f044c).
Optimization stays pooled L1; the new experiment selects its mix using MAPE.
"""
import numpy as np


def weighted_breakpoint(residual, direction):
    keep = np.isfinite(residual) & np.isfinite(direction) & (direction != 0)
    if not keep.any():
        return 0.
    points, mass = -residual[keep]/direction[keep], np.abs(direction[keep])
    finite = np.isfinite(points)
    points, mass = points[finite], mass[finite]
    if not len(points):
        return 0.
    order = np.argsort(points)
    points, mass = points[order], mass[order]
    cumulative = np.cumsum(mass)
    return float(points[min(np.searchsorted(cumulative, cumulative[-1]/2), len(points)-1)])


def fit_simplex(members, truth, valid=None, *, iterations=25, max_points=20000):
    members, truth = np.asarray(members, float), np.asarray(truth, float)
    valid = np.ones_like(truth, bool) if valid is None else np.asarray(valid, bool)
    keep = valid & np.isfinite(members).all(axis=1) & np.isfinite(truth)
    indices = np.flatnonzero(keep)
    if not len(indices):
        raise ValueError("Convex fitting requires valid development observations.")
    if len(indices) > max_points:
        indices = np.random.default_rng(0).choice(indices, size=max_points, replace=False)
    P, y = members[indices], truth[indices]
    count = members.shape[1]
    weights = np.full(count, 1/count)
    current = P @ weights
    for _ in range(iterations):
        moved = 0.
        for i in range(count):
            for j in range(count):
                if i == j or weights[j] <= 1e-12:
                    continue
                direction = P[:, i]-P[:, j]
                step = float(np.clip(weighted_breakpoint(y-current, direction), -max(weights[j], 0.), max(weights[i], 0.)))
                if not np.isfinite(step) or abs(step) <= 1e-12:
                    continue
                weights[i] = max(weights[i]-step, 0.)
                weights[j] = max(weights[j]+step, 0.)
                current = P @ weights
                moved += abs(step)
        if moved < 1e-10:
            break
    weights = np.clip(weights, 0., None)
    return weights/weights.sum()


def segment_ids(horizon, segment_points=48):
    count = max(1, round(horizon/segment_points))
    return np.minimum(np.arange(horizon)*count//horizon, count-1)


def fit_segments(members, truth, segment_points=48):
    """members [origin, lead, member], truth [origin, lead]."""
    ids = segment_ids(truth.shape[1], segment_points)
    return np.stack([fit_simplex(members[:, ids == i].reshape(-1, members.shape[-1]),
                                truth[:, ids == i].reshape(-1)) for i in range(ids.max()+1)])


def apply_segments(members, weights):
    ids = np.minimum(np.arange(members.shape[1])*len(weights)//members.shape[1], len(weights)-1)
    return np.einsum("olm,lm->ol", members, weights[ids])


def purged_folds(origins, horizon, folds=5, mode="blocked"):
    """Purge training origins whose target intervals overlap the held-out block."""
    if len(origins) < 2:
        raise ValueError("HDSF selection needs >=2 valid validation origins.")
    all_indices = np.arange(len(origins))
    blocks = [[i] for i in all_indices] if mode == "loo" else np.array_split(all_indices, min(folds, len(origins)))
    duration = horizon * 15 * 60 * 1_000_000_000
    times = origins.as_unit("ns").asi8
    result = []
    for block in blocks:
        block = np.asarray(block, dtype=int)
        keep = np.ones(len(origins), bool)
        for held in block:
            keep &= (times+duration <= times[held]) | (times >= times[held]+duration)
        train = all_indices[keep]
        if len(train):
            result.append((train, block))
    if not result:
        raise ValueError("No independent validation folds after purging overlapping horizons; use more data.")
    return result
