"""Optional object-level evaluation: decode the 20x20x7 traffic grid into object centres and
match them one-to-one to GT actors (class-agnostic centre AP at a distance threshold).

Grid semantics (upstream det_utils.generate_det_data): cell (i, j) has centre
(x = j - 9.5, y = 17.5 - i) in the ego frame (x lateral right +, y forward +); channel 1/2 are
(actor - centre) offsets divided by 3.5. The GT encode->decode round trip is unit-tested in
tests/test_objects.py before any AP is reported.
"""
import numpy as np

from rrf import metrics

OFFSET_SCALE = 3.5


def cell_centres():
    i, j = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    return (j - 9.5).ravel(), (17.5 - i).ravel()  # x, y for the 400 flattened cells


def decode_centres(grid, score_threshold, nms_radius=2.0):
    """grid (400,7) -> list of (score, x, y). Peak extraction: cells above threshold, greedy
    suppression of lower-scored detections within nms_radius metres of an accepted one."""
    grid = np.asarray(grid).reshape(400, 7)
    cx, cy = cell_centres()
    x = cx + grid[:, 1] * OFFSET_SCALE
    y = cy + grid[:, 2] * OFFSET_SCALE
    order = np.argsort(-grid[:, 0], kind="stable")
    kept = []
    for k in order:
        s = float(grid[k, 0])
        if s < score_threshold:
            break
        if all((x[k] - kx) ** 2 + (y[k] - ky) ** 2 >= nms_radius ** 2 for _, kx, ky in kept):
            kept.append((s, float(x[k]), float(y[k])))
    return kept


def gt_centres(grid):
    """GT actor centres from the soft target: every foreground cell decodes to its actor's centre,
    so identical decoded centres (within 1e-3 m) collapse to one object."""
    grid = np.asarray(grid).reshape(400, 7)
    cx, cy = cell_centres()
    objs = []
    for k in np.where(grid[:, 0] > 0.01)[0]:
        x, y = cx[k] + grid[k, 1] * OFFSET_SCALE, cy[k] + grid[k, 2] * OFFSET_SCALE
        if all(abs(x - ox) > 1e-3 or abs(y - oy) > 1e-3 for ox, oy in objs):
            objs.append((float(x), float(y)))
    return objs


def match(dets, gts, radius):
    """Greedy one-to-one matching by score; returns list of (score, is_tp) and n_gt."""
    used = [False] * len(gts)
    out = []
    for s, x, y in sorted(dets, key=lambda d: -d[0]):
        best, bd = -1, radius
        for g, (gx, gy) in enumerate(gts):
            if used[g]:
                continue
            d = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5
            if d <= bd:
                best, bd = g, d
        if best >= 0:
            used[best] = True
        out.append((s, best >= 0))
    return out, len(gts)


def center_ap(pred_grids, gt_grids, radius=1.0, score_threshold=0.05, nms_radius=2.0):
    """class_agnostic_center_AP@radius over a set of frames; None if no GT object."""
    scored, n_gt = [], 0
    for p, g in zip(pred_grids, gt_grids):
        m, n = match(decode_centres(p, score_threshold, nms_radius), gt_centres(g), radius)
        scored += m
        n_gt += n
    if n_gt == 0:
        return None, {"n_gt": 0, "n_det": len(scored)}
    scored.sort(key=lambda t: -t[0])
    tp = np.cumsum([t[1] for t in scored]) if scored else np.array([])
    fp = np.cumsum([not t[1] for t in scored]) if scored else np.array([])
    precision = tp / np.maximum(tp + fp, 1) if len(scored) else np.array([])
    ap = float(sum(precision[i] for i, t in enumerate(scored) if t[1]) / n_gt) if len(scored) else 0.0
    return ap, {"n_gt": n_gt, "n_det": len(scored), "recall_at_all_dets": float(tp[-1] / n_gt) if len(scored) else 0.0}
