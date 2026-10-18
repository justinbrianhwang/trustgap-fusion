import numpy as np
import pytest
import torch

from rrf import corruptions as C
from rrf import upstream as U


def _raw(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-30, 30, size=(3000, 4)).astype(np.float32)
    return {
        "sample_id": "ep0/000001",
        "rgb_front": rng.integers(0, 255, size=(600, 800, 3), dtype=np.uint8),
        "rgb_left": rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8),
        "rgb_right": rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8),
        "lidar": pts,
        "speed": 2.0, "command": 4, "target_point": np.array([0.0, -8.0], np.float32),
        # privileged keys that must never reach the model
        "corruption": "x", "severity": 5, "weather": "rain", "target": np.zeros((400, 7)),
    }


def test_transform_2d_points_with_equal_args_is_identity():
    U.bootstrap()
    from team_code.utils import transform_2d_points
    pts = np.random.default_rng(0).uniform(-50, 50, size=(500, 3))
    out = transform_2d_points(pts, np.pi / 2 - 0.3, -12.0, 7.5, np.pi / 2 - 0.3, -12.0, 7.5)
    assert np.allclose(out, pts, atol=1e-9)


def test_preprocess_shapes_and_only_allowed_keys():
    x = U.preprocess(_raw())
    assert set(x) == set(U.MODEL_INPUT_KEYS)
    assert tuple(x["rgb"].shape) == (3, 224, 224)
    assert tuple(x["rgb_left"].shape) == (3, 128, 128) == tuple(x["rgb_center"].shape)
    assert tuple(x["lidar"].shape) == (3, 224, 224)
    assert tuple(x["measurements"].shape) == (7,) and float(x["measurements"][3]) == 1.0
    assert float(x["lidar"].max()) <= 2.0 and float(x["lidar"].min()) >= 0.0


def test_front_corruption_reaches_center_crop():
    raw = _raw()
    corrupted, _ = C.corrupt_sample(raw, {"camera": ("silent_blackout", 1)}, 1)
    a, b = U.preprocess(raw), U.preprocess(corrupted)
    assert not torch.equal(a["rgb_center"], b["rgb_center"])
    assert not torch.equal(a["rgb"], b["rgb"])
    assert torch.equal(a["lidar"], b["lidar"])


def test_empty_lidar_gives_zero_histogram():
    h = U.lidar_histogram(np.zeros((0, 4), np.float32))
    assert h.shape == (3, 224, 224) and float(h.sum()) == 0.0


def test_foreground_label_threshold_boundary():
    t = np.zeros((400, 7), np.float32)
    t[0, 0] = 0.01
    t[1, 0] = 0.0101
    y = U.foreground_label(t)
    assert not y[0] and y[1] and y.sum() == 1


def test_traffic_grid_target_places_actor_ahead():
    # ego at origin facing +x (theta = pi/2 in upstream's compass convention); actor 10 m ahead
    meas = {"x": 0.0, "y": 0.0, "theta": np.pi / 2, "affected_light_id": -1,
            "is_vehicle_present": [], "is_bike_present": [], "is_junction_vehicle_present": [], "is_pedestrian_present": []}
    actors = {"7": {"loc": [10.0, 0.0, 0.0], "ori": [1.0, 0.0, 0.0], "box": [2.3, 1.0], "vel": [0.0, 0.0, 0.0], "tpe": 0}}
    t = U.traffic_grid_target(meas, actors).reshape(20, 20, 7)
    i, j = np.unravel_index(np.argmax(t[..., 0]), (20, 20))
    assert t[i, j, 0] > 0.5
    assert 6 <= i <= 8 and 9 <= j <= 10  # row 7 ~ 10.5 m ahead (y = 17.5 - i), centre column
    # actor 5 m to the right (CARLA +y when facing +x) lands in the right half
    actors["7"]["loc"] = [3.0, 5.0, 0.0]
    t = U.traffic_grid_target(meas, actors).reshape(20, 20, 7)
    i, j = np.unravel_index(np.argmax(t[..., 0]), (20, 20))
    assert j >= 13 and 13 <= i <= 15
