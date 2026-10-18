"""Deterministic raw-sensor corruption engine.

Camera corruptions act on raw uint8 RGB images (before the upstream resize/crop/normalize),
LiDAR corruptions on the raw sensor-frame point cloud (N,4) before the BEV histogram.
Inputs are never mutated; every call returns a new sample plus an audit dict with the
realized parameters. Severity 0 or family "clean" is the identity.
"""
import hashlib
import math

import cv2
import numpy as np

# severity 1..5 -> parameter. All values are initial experimental settings, not measured
# failure distributions.
CAMERA_LEVELS = {
    "gaussian_noise": [0.01, 0.03, 0.05, 0.08, 0.12],        # sigma in [0,1] RGB
    "gaussian_blur": [0.5, 1.0, 1.5, 2.0, 3.0],              # sigma px at reference width 800
    "darkening": [0.8, 0.6, 0.4, 0.25, 0.1],                  # intensity multiplier
    "rectangular_occlusion": [0.05, 0.1, 0.2, 0.35, 0.5],    # area ratio
}
LIDAR_LEVELS = {
    "point_dropout": [0.1, 0.3, 0.5, 0.7, 0.9],              # removal rate
    "xyz_jitter": [0.01, 0.03, 0.05, 0.1, 0.2],               # sigma metres per axis
    "azimuth_sector_dropout": [10.0, 20.0, 40.0, 60.0, 90.0],  # removed sector width deg
}
FULL_LOSS = {"silent_blackout", "silent_empty"}
CAMERA_FAMILIES = set(CAMERA_LEVELS) | {"silent_blackout"}
LIDAR_FAMILIES = set(LIDAR_LEVELS) | {"silent_empty"}
BLUR_REFERENCE_WIDTH = 800
CAMERA_VIEWS = ("rgb_front", "rgb_left", "rgb_right")
# Sensor-frame forward axis of the LiDAR (mounted with yaw=-90 in upstream). [to verify]
# in collected data: sensor +y should point along the ego heading. See docs/DECISIONS.md.
LIDAR_FORWARD_AXIS = np.array([0.0, 1.0])
# Perception ROI of the 20x20 traffic grid in the (lateral, forward) ego frame, metres.
ROI_LATERAL = (-10.0, 10.0)
ROI_FORWARD = (-2.0, 18.0)


def corruption_seed(experiment_seed, sample_id, family, severity, repeat=0):
    """Stable 63-bit seed from SHA256 (never Python hash())."""
    key = f"{experiment_seed}|{sample_id}|{family}|{severity}|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") >> 1


def _rng(experiment_seed, sample_id, family, severity, repeat, stream=""):
    return np.random.default_rng(corruption_seed(experiment_seed, sample_id + stream, family, severity, repeat))


# ------------------------------------------------------------------ camera
def corrupt_image(img, family, severity, rng):
    """uint8 HxWx3 RGB -> (new uint8 image, audit). Identity for clean/severity 0."""
    img = np.asarray(img)
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, img.shape
    if family == "clean" or severity == 0:
        return img.copy(), {"family": "clean"}
    h, w = img.shape[:2]
    if family == "silent_blackout":
        return np.zeros_like(img), {"family": family}
    level = CAMERA_LEVELS[family][severity - 1]
    if family == "gaussian_noise":
        x = img.astype(np.float32) / 255.0
        x = np.clip(x + rng.normal(0.0, level, size=x.shape).astype(np.float32), 0.0, 1.0)
        return np.round(x * 255.0).astype(np.uint8), {"family": family, "sigma": level}
    if family == "gaussian_blur":
        sigma = level * w / BLUR_REFERENCE_WIDTH
        ksize = 2 * int(math.ceil(3.0 * sigma)) + 1
        out = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return out, {"family": family, "sigma_ref": level, "sigma_applied": sigma, "kernel": ksize}
    if family == "darkening":
        out = np.round(img.astype(np.float32) * level).astype(np.uint8)
        return out, {"family": family, "multiplier": level}
    if family == "rectangular_occlusion":
        area = level * h * w
        aspect = float(rng.uniform(0.5, 2.0))  # width / height
        rh = int(round(min(h, math.sqrt(area / aspect))))
        rw = int(round(min(w, area / max(rh, 1))))
        y0 = int(rng.integers(0, h - rh + 1))
        x0 = int(rng.integers(0, w - rw + 1))
        fill = int(rng.integers(0, 256))
        out = img.copy()
        out[y0:y0 + rh, x0:x0 + rw] = fill
        return out, {"family": family, "area_ratio": level, "rect": [x0, y0, rw, rh], "fill": fill,
                     "realized_area_ratio": rw * rh / (h * w)}
    raise KeyError(family)


# ------------------------------------------------------------------ lidar
def corrupt_lidar(points, family, severity, rng, dropout_rng=None):
    """(N,4) float32 sensor-frame cloud -> (new cloud, audit). Shape/dtype preserved, N may
    become 0. `dropout_rng` (severity-independent) makes dropout subsets nested."""
    pts = np.asarray(points, dtype=np.float32)
    assert pts.ndim == 2 and pts.shape[1] >= 3, pts.shape
    if family == "clean" or severity == 0:
        return pts.copy(), {"family": "clean", "n_in": len(pts), "n_out": len(pts)}
    if family == "silent_empty":
        return pts[:0].copy(), {"family": family, "n_in": len(pts), "n_out": 0}
    level = LIDAR_LEVELS[family][severity - 1]
    if family == "point_dropout":
        u = (dropout_rng or rng).random(len(pts))
        keep = u >= level  # nested: higher rate removes a superset
        out = pts[keep].copy()
        return out, {"family": family, "rate": level, "n_in": len(pts), "n_out": len(out),
                     "realized_rate": 1.0 - len(out) / max(len(pts), 1)}
    if family == "xyz_jitter":
        out = pts.copy()
        out[:, :3] += rng.normal(0.0, level, size=(len(pts), 3)).astype(np.float32)
        return out, {"family": family, "sigma_m": level, "n_in": len(pts), "n_out": len(out)}
    if family == "azimuth_sector_dropout":
        center = float(rng.uniform(-180.0, 180.0))
        fwd = LIDAR_FORWARD_AXIS
        # angle of each point relative to the forward axis, degrees in (-180, 180]
        ang = np.degrees(np.arctan2(pts[:, 0] * fwd[1] - pts[:, 1] * fwd[0], pts[:, 0] * fwd[0] + pts[:, 1] * fwd[1]))
        d = (ang - center + 180.0) % 360.0 - 180.0
        remove = np.abs(d) <= level / 2.0
        out = pts[~remove].copy()
        # ROI overlap statistic in the (lateral, forward) ego frame
        forward = pts[:, 0] * fwd[0] + pts[:, 1] * fwd[1]
        lateral = pts[:, 0] * fwd[1] - pts[:, 1] * fwd[0]
        in_roi = (ROI_LATERAL[0] <= lateral) & (lateral <= ROI_LATERAL[1]) & (ROI_FORWARD[0] <= forward) & (forward <= ROI_FORWARD[1])
        return out, {"family": family, "width_deg": level, "center_deg": center, "n_in": len(pts), "n_out": len(out),
                     "removed_in_roi": int((remove & in_roi).sum()), "roi_points": int(in_roi.sum())}
    raise KeyError(family)


# ------------------------------------------------------------------ sample level
def corrupt_sample(raw, spec, experiment_seed, repeat=0):
    """Apply a corruption spec to a raw sample dict without mutating it.

    spec = {"camera": (family, severity) | None, "lidar": (family, severity) | None}
    Camera scope is the whole rig: the same family/severity on front/left/right with
    independent noise streams per view. rgb_center is derived later from the corrupted
    front image by upstream preprocessing, so it can never bypass the corruption.
    """
    out = dict(raw)
    audit = {"sample_id": raw["sample_id"], "experiment_seed": experiment_seed, "repeat": repeat, "camera": None, "lidar": None}
    cam = spec.get("camera")
    if cam and cam[0] != "clean" and cam[1] != 0:
        family, sev = cam
        views = {}
        for view in CAMERA_VIEWS:
            rng = _rng(experiment_seed, raw["sample_id"], family, sev, repeat, stream="/" + view)
            out[view], views[view] = corrupt_image(raw[view], family, sev, rng)
        audit["camera"] = {"family": family, "severity": sev, "views": views}
    lid = spec.get("lidar")
    if lid and lid[0] != "clean" and lid[1] != 0:
        family, sev = lid
        rng = _rng(experiment_seed, raw["sample_id"], family, sev, repeat)
        drop_rng = _rng(experiment_seed, raw["sample_id"], family, 0, repeat) if family == "point_dropout" else None
        out["lidar"], a = corrupt_lidar(raw["lidar"], family, sev, rng, dropout_rng=drop_rng)
        audit["lidar"] = {"family": family, "severity": sev, **a}
    return out, audit


def condition_name(spec):
    parts = []
    for mod in ("camera", "lidar"):
        v = spec.get(mod)
        if v and v[0] != "clean" and v[1] != 0:
            parts.append(f"{mod}_{v[0]}_s{v[1]}")
    return "__".join(parts) if parts else "clean"
