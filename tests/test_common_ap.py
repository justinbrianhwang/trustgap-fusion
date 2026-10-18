import json

import numpy as np
import pytest

from rrf.common_ap import (COMMON_REGION, actor_centres, common_center_ap,
                           evaluate_predictions, interfuser_detections,
                           transfuser_detections)


def test_transfuser_vehicle_frame_conversion_and_common_region():
    # right=2 m, vehicle-forward=10 m => LiDAR-forward=8.7 m.
    box = np.array([[18 * 8, (32 - 8.7) * 8, 1, 1, 0, 0, 0, 0.8],
                    [0, 0, 1, 1, 0, 0, 0, 0.9],
                    [16 * 8, (32 - 8.7) * 8, 1, 1, 0, 0, 0, 0.01]], np.float32)
    detections = transfuser_detections(box, count=3)
    assert len(detections) == 1
    assert detections[0] == pytest.approx((0.8, 2.0, 10.0))
    assert COMMON_REGION["forward_min_m"] == 1.3


def test_common_ap_known_answer_and_interfuser_decode():
    traffic = np.zeros((400, 7), np.float32)
    # Cell (8, 10) has centre (0.5, 9.5), inside the common region.
    traffic[8 * 20 + 10, 0] = 0.8
    detections = interfuser_detections(traffic)
    assert detections == [(np.float32(0.8), 0.5, 9.5)]
    predictions = [[(0.9, -9, 17), (0.8, 0.5, 9.5)]]
    ap, info = common_center_ap(predictions, [[(0.5, 9.5)]], radius=1.0)
    assert ap == 0.5 and info["n_gt"] == 1 and info["n_det"] == 2


def test_actor_centres_are_vehicle_only_but_walkers_are_counted(tmp_path):
    directory = tmp_path / "ep"
    (directory / "measurements").mkdir(parents=True)
    (directory / "actors_data").mkdir()
    (directory / "measurements" / "0000.json").write_text(json.dumps(
        {"x": 0, "y": 0, "theta": np.pi / 2, "yaw_deg": 0}))
    actors = {
        "ego": {"loc": [0, 0, 0], "tpe": 0},
        "car": {"loc": [10, 2, 0], "tpe": 0},
        "walker": {"loc": [8, -1, 0], "tpe": 1},
        "behind": {"loc": [-2, 0, 0], "tpe": 0},
    }
    (directory / "actors_data" / "0000.json").write_text(json.dumps(actors))
    vehicles, walkers = actor_centres(tmp_path, "ep/0000")
    assert vehicles == [(2.0, 10.0)]
    assert walkers == [(-1.0, 8.0)]

    traffic = np.zeros((1, 400, 7), np.float32)
    cell = 8 * 20 + 11  # centre (right=1.5, forward=9.5)
    traffic[0, cell, :3] = [1.0, (2.0 - 1.5) / 3.5, (10.0 - 9.5) / 3.5]
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    np.savez(pred_dir / "pred_clean.npz", sample_ids=np.array(["ep/0000"]), traffic=traffic)
    report = evaluate_predictions(pred_dir, "interfuser", tmp_path)
    assert report["conditions"]["clean"]["vehicle_center_ap_1m"] == 1.0
    assert report["walker_gt_instances_in_region"] == 1
    assert (pred_dir / "common_ap.json").exists()
