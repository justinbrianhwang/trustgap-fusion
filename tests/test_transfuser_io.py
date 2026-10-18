import gc
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

if "rrf_tf" not in Path(sys.executable).parent.name:
    pytest.skip("TransFuser tests require the rrf_tf interpreter", allow_module_level=True)
pytest.importorskip("timm")

import torch

from rrf.transfuser_io import (VARIANTS, bootstrap_tf, load_raw_tf, load_tf_model, predict_tf,
                               preprocess_tf, tf_targets)


def _synthetic_raw(seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform([-15, -30, -3, 0], [15, 5, 2, 1], size=(3000, 4)).astype(np.float32)
    return {f"rgb_{name}": rng.integers(0, 256, (480, 960, 3), dtype=np.uint8)
            for name in ("left", "front", "right")} | {
                "lidar": points, "speed": 3.0, "target_point": np.array([8.0, -1.0], np.float32)}


def _config(backbone="transfuser"):
    bootstrap_tf()
    from config import GlobalConfig
    config = GlobalConfig(setting="eval")
    config.backbone = backbone
    config.use_target_point_image = True
    return config


def test_preprocess_tf_shapes():
    batch = preprocess_tf(_synthetic_raw(), _config())
    assert tuple(batch["rgb"].shape) == (3, 160, 704)
    assert tuple(batch["lidar"].shape) == (2, 256, 256)
    assert tuple(batch["target_point"].shape) == (2,)
    assert tuple(batch["target_point_image"].shape) == (1, 256, 256)
    assert tuple(batch["velocity"].shape) == (1,)


def test_load_raw_tf_splits_lossless_camera_strip(monkeypatch):
    import rrf.transfuser_io as tfio
    strip = np.concatenate([np.broadcast_to(np.array(pixel, np.uint8), (480, 960, 3))
                            for pixel in ([1, 2, 3], [4, 5, 6], [7, 8, 9])], axis=1)
    monkeypatch.setattr(tfio.Image, "open", lambda _path: Image.fromarray(strip))
    monkeypatch.setattr(tfio.np, "load", lambda _path: np.ones((4, 4), np.float32))
    monkeypatch.setattr(tfio.Path, "read_text", lambda _path: json.dumps(
        {"speed": 2.5, "target_point": [7.0, -1.0], "compass": 0.2, "gps": [1.0, 2.0, 3.0]}))
    raw = load_raw_tf("unused", "ep/0000")
    assert [raw[f"rgb_{name}"][0, 0].tolist() for name in ("left", "front", "right")] == [
        [3, 2, 1], [6, 5, 4], [9, 8, 7]]
    assert raw["lidar"].dtype == np.float32 and raw["speed"] == 2.5


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoint_strict_load_and_forward_shapes(variant):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_tf_model(variant, device=device)
    batch = preprocess_tf(_synthetic_raw(), config)
    output = predict_tf(model, batch)
    assert tuple(output["waypoints"].shape) == (1, 4, 2)
    assert tuple(output["boxes"].shape) == (1, 100, 8)
    assert tuple(output["box_labels"].shape) == (1, 100)
    assert tuple(output["center_heatmap"].shape) == (1, 1, 64, 64)
    assert tuple(output["bev_logits"].shape) == (1, 3, 160, 160)
    del output, batch, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_tf_targets_known_actor_bev_cell():
    episode, frame = "ep0", "0000"
    root = Path(__file__).parent / "fixtures" / "tf_target"
    target = tf_targets(root, f"{episode}/{frame}", _config())
    assert target.shape == (1, 7)
    # actor 10 m ahead of the vehicle centre = 8.7 m ahead of the LiDAR (BEV origin, lidar_pos x = 1.3 m)
    assert np.allclose(target[0, :4], [144.0, 176.0 + 1.3 * 8.0, 16.0, 32.0])
    assert tuple((target[0, :2] / 4).astype(int)) == (36, 46)
    assert np.isclose(target[0, 4], 0.0) and np.isclose(target[0, 5], 3.0)
