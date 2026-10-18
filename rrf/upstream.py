"""Adapter around the vendored upstream InterFuser code (third_party/InterFuser).

Responsibilities: sys.path bootstrap, checkpoint loading, the original preprocessing
(RGB transforms + LiDAR histogram), a dynamically derived token layout, and the GT
traffic-grid generation. Nothing here changes upstream semantics.
"""
import copy
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import torch

from rrf import ROOT

UPSTREAM = ROOT / "third_party" / "InterFuser"
CARLA_API = ROOT / "carla" / "PythonAPI" / "carla"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OUTPUT_NAMES = ("traffic", "waypoints", "is_junction", "traffic_light_state", "stop_sign", "traffic_feature")
MODEL_INPUT_KEYS = ("rgb", "rgb_left", "rgb_right", "rgb_center", "lidar", "measurements", "target_point")

_bootstrapped = False


def bootstrap():
    """Make vendored `timm`, `team_code`, and CARLA `agents` importable. Idempotent."""
    global _bootstrapped
    if _bootstrapped:
        return
    for p in (UPSTREAM / "interfuser", UPSTREAM / "leaderboard", CARLA_API):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    if not hasattr(np, "int"):
        np.int = int  # upstream heatmap code uses the removed alias
    if "imgaug" not in sys.modules:  # only needed by upstream's optional augmenter
        imgaug = types.ModuleType("imgaug")
        imgaug.augmenters = types.ModuleType("imgaug.augmenters")
        sys.modules["imgaug"] = imgaug
        sys.modules["imgaug.augmenters"] = imgaug.augmenters
    import timm  # noqa: F401  (vendored)
    timm_file = Path(timm.__file__).resolve()
    if UPSTREAM.resolve() not in timm_file.parents:
        raise RuntimeError(f"non-vendored timm imported from {timm_file}")
    import timm.models.helpers as helpers
    helpers.load_pretrained = lambda *a, **k: None  # never download ImageNet weights
    _bootstrapped = True


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_model(model_name="interfuser_baseline"):
    bootstrap()
    from timm.models import create_model
    return create_model(model_name)


def load_checkpoint(model, ckpt_path, strict=True, allowed_missing_prefixes=()):
    """Load upstream weights. Missing keys are only tolerated under allowed prefixes."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    result = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in result.missing_keys if not k.startswith(tuple(allowed_missing_prefixes))]
    report = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "arch": ckpt.get("arch") if isinstance(ckpt, dict) else None,
        "state_dict_keys": len(state),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
    if strict and (bad_missing or result.unexpected_keys):
        raise RuntimeError(f"checkpoint load not clean: missing={bad_missing[:5]} unexpected={result.unexpected_keys[:5]}")
    return report


def load_upstream_model(ckpt_path, device="cpu", model_name="interfuser_baseline"):
    model = build_model(model_name)
    report = load_checkpoint(model, ckpt_path, strict=True)
    model.to(device).eval()
    return model, report


# ---------------------------------------------------------------- preprocessing
class _Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize(self.size)


def rgb_transform(input_size, need_scale=True):
    """Identical to leaderboard/team_code/interfuser_agent.create_carla_rgb_transform."""
    from torchvision import transforms
    tfl = []
    if need_scale:
        tfl.append(_Resize({128: (195, 146), 224: (341, 256)}[input_size]))
    tfl += [transforms.CenterCrop(input_size), transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(IMAGENET_MEAN), std=torch.tensor(IMAGENET_STD))]
    return transforms.Compose(tfl)


_TRANSFORMS = {}


def _tf(key):
    if not _TRANSFORMS:
        _TRANSFORMS.update(front=rgb_transform(224), side=rgb_transform(128), center=rgb_transform(128, need_scale=False))
    return _TRANSFORMS[key]


def lidar_histogram(lidar_xyz):
    """Upstream BEV histogram of a raw sensor-frame point cloud (N,3+), with the y flip the
    agent applies. transform_2d_points(x, r, t, r, t) is the identity, so it is skipped
    (verified in tests/test_upstream.py)."""
    bootstrap()
    from team_code.utils import lidar_to_histogram_features
    pts = np.asarray(lidar_xyz, dtype=np.float32)[:, :3].copy()
    pts[:, 1] *= -1
    return lidar_to_histogram_features(pts, crop=224)  # (3, 224, 224) float32


def command_one_hot(command, speed):
    cmd = int(command) - 1
    if cmd < 0 or cmd > 5:
        cmd = 3
    vec = [0.0] * 6
    vec[cmd] = 1.0
    vec.append(float(speed))
    return np.asarray(vec, dtype=np.float32)


def preprocess(raw):
    """Raw observation dict -> model inputs (CPU tensors, no batch dim).

    raw keys: rgb_front/rgb_left/rgb_right (H,W,3 uint8 RGB), lidar (N,4 float32 sensor frame),
    speed (m/s), command (RoadOption int), target_point (2,) in the ego/planner frame.
    GT and metadata keys are ignored, never forwarded.
    """
    from PIL import Image
    front = Image.fromarray(raw["rgb_front"])
    return {
        "rgb": _tf("front")(front),
        "rgb_left": _tf("side")(Image.fromarray(raw["rgb_left"])),
        "rgb_right": _tf("side")(Image.fromarray(raw["rgb_right"])),
        "rgb_center": _tf("center")(front),
        "lidar": torch.from_numpy(lidar_histogram(raw["lidar"])),
        "measurements": torch.from_numpy(command_one_hot(raw["command"], raw["speed"])),
        "target_point": torch.as_tensor(np.asarray(raw["target_point"], dtype=np.float32)),
    }


def collate(samples, device="cpu"):
    return {k: torch.stack([s[k] for s in samples]).to(device) for k in MODEL_INPUT_KEYS}


# ---------------------------------------------------------------- token layout
# view index (0 front,1 left,2 right,3 center,4 lidar) -> (modality, physical sensor)
VIEW_INFO = [("camera", "front"), ("camera", "left"), ("camera", "right"), ("camera", "front"), ("lidar", "lidar")]


@torch.no_grad()
def token_layout(model, inputs):
    """Derive the encoder token sequence layout from the actual patch-embed outputs.

    Returns a list of dicts (one per token, in sequence order) with keys
    view, modality, physical_sensor, is_global; the total must match the sequence
    length the encoder sees. Nothing is hard-coded to 151.
    """
    layout = []
    views = [
        (0, inputs["rgb"], model.rgb_patch_embed),
        (1, inputs["rgb_left"], model.rgb_patch_embed),
        (2, inputs["rgb_right"], model.rgb_patch_embed),
        (3, inputs["rgb_center"], model.rgb_patch_embed),
        (4, inputs["lidar"], model.lidar_patch_embed),
    ]
    for view, x, embed in views:
        feat, _ = embed(x[:1])
        n = int(feat.shape[-2] * feat.shape[-1])
        modality, sensor = VIEW_INFO[view]
        layout += [dict(view=view, modality=modality, physical_sensor=sensor, is_global=False)] * n
        layout.append(dict(view=view, modality=modality, physical_sensor=sensor, is_global=True))
    return layout


# ---------------------------------------------------------------- ground truth
def traffic_grid_target(measurements, actors_data):
    """Upstream 20x20x7 soft target, flattened to (400, 7) float32. Inputs are deep-copied
    because upstream mutates them."""
    bootstrap()
    from timm.data.heatmap_utils import generate_heatmap
    from timm.data.det_utils import generate_det_data
    heatmap = generate_heatmap(copy.deepcopy(measurements), copy.deepcopy(actors_data))
    det = generate_det_data(heatmap, copy.deepcopy(measurements), copy.deepcopy(actors_data))
    return det.reshape(400, 7).astype(np.float32)


def foreground_label(target, threshold=0.01):
    """Binary grid foreground support label from the soft target's first channel (MVTL1Loss
    uses >= 0.01 as foreground; we use > threshold so the 0.01 boundary is background)."""
    return (np.asarray(target)[..., 0] > threshold)


@torch.no_grad()
def predict(model, batch):
    """Raw upstream forward on an allowed-key batch. Returns dict of outputs."""
    filtered = {k: batch[k] for k in MODEL_INPUT_KEYS}
    outs = model(filtered)
    return dict(zip(OUTPUT_NAMES, outs))
