"""Grid-foreground metrics (AP, fixed operating point FN/FP), upstream native traffic
losses, and episode-level paired bootstrap. Everything works on numpy arrays.

Names are deliberately `grid_fg_*`: this is InterFuser traffic-grid foreground support
detection, not 3D object detection mAP.
"""
import numpy as np
import torch


def grid_fg_ap(scores, labels):
    """Average precision over all valid cells (score = raw first traffic channel).
    Returns None when there is no positive cell (never a fake 1.0)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(labels[order])
    fp = np.cumsum(~labels[order])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # standard step-wise AP: sum of precision at each positive
    ap = float((precision * labels[order]).sum() / n_pos)
    return ap


def calibrate_threshold(scores, labels, target_precision=0.90):
    """Highest-recall threshold whose precision >= target on the calibration split.
    Returns None (operating_point_unavailable) if no threshold with >=1 prediction reaches it."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    if labels.sum() == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(labels[order])
    fp = np.cumsum(~labels[order])
    precision = tp / (tp + fp)
    ok = np.where(precision >= target_precision)[0]
    if len(ok) == 0:
        return None
    k = int(ok[-1])  # most predictions (highest recall) still meeting the target
    return float(scores[order][k])


def confusion(scores, labels, threshold):
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    pred = scores >= threshold
    tp, fp = int((pred & labels).sum()), int((pred & ~labels).sum())
    fn, tn = int((~pred & labels).sum()), int((~pred & ~labels).sum())

    def ratio(a, b):
        return None if b == 0 else a / b
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "fnr": ratio(fn, tp + fn), "fpr": ratio(fp, fp + tn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn)}


def native_traffic_losses(output, target):
    """Upstream MVTL1Loss decomposed: (score loss, geometry loss, speed loss) with L1, plus the
    combined value train.py would use (0.5*(0.5*score)+0.5*geometry ... see train.py:837)."""
    output = torch.as_tensor(output, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    l1 = torch.nn.L1Loss()
    m1 = target[..., 0].ge(0.01)
    m0 = target[..., 0].le(0.01)
    score1 = l1(output[..., 0][m1], target[..., 0][m1]) if m1.any() else torch.tensor(0.0)
    score0 = l1(output[..., 0][m0], target[..., 0][m0]) if m0.any() else torch.tensor(0.0)
    score = 0.5 * score0 + 0.5 * score1
    geometry = l1(output[m1][:, 1:6], target[m1][:, 1:6]) if m1.any() else torch.tensor(0.0)
    speed = l1(output[m1][:, 6], target[m1][:, 6]) if m1.any() else torch.tensor(0.0)
    combined = 0.5 * score + 0.5 * geometry
    return {"score": float(score), "geometry": float(geometry), "speed": float(speed), "traffic_combined": float(combined)}


def paired_bootstrap(per_group_a, per_group_b, repeats=2000, seed=0, stat=np.mean):
    """CI of stat(B) - stat(A) resampling *groups* (episodes) with replacement, paired."""
    a, b = np.asarray(per_group_a, dtype=np.float64), np.asarray(per_group_b, dtype=np.float64)
    assert a.shape == b.shape
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(repeats)
    for r in range(repeats):
        idx = rng.integers(0, n, size=n)
        diffs[r] = stat(b[idx]) - stat(a[idx])
    return {"delta": float(stat(b) - stat(a)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "groups": n, "repeats": repeats}
