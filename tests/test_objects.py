import numpy as np

from rrf import objects, upstream


def _meas():
    return {"x": 0.0, "y": 0.0, "theta": np.pi / 2, "affected_light_id": -1,
            "is_vehicle_present": [], "is_bike_present": [], "is_junction_vehicle_present": [], "is_pedestrian_present": []}


def _actor(x, y, yaw_deg=0.0, box=(2.3, 1.0), vel=(3.0, 0.0, 0.0)):
    o = np.deg2rad(yaw_deg)
    return {"loc": [x, y, 0.0], "ori": [float(np.cos(o)), float(np.sin(o)), 0.0], "box": list(box), "vel": list(vel), "tpe": 0}


def test_gt_encode_decode_recovers_actor_centres():
    # ego at origin facing +x: actor 10 m ahead, 1.5 m right; second actor 6 m ahead 4 m left
    actors = {"1": _actor(10.0, 1.5), "2": _actor(6.0, -4.0, yaw_deg=90)}
    grid = upstream.traffic_grid_target(_meas(), actors)
    centres = sorted(objects.gt_centres(grid), key=lambda c: c[1])
    assert len(centres) == 2
    # ego frame: x lateral (right +), y forward
    assert abs(centres[0][0] - (-4.0)) < 0.05 and abs(centres[0][1] - 6.0) < 0.05
    assert abs(centres[1][0] - 1.5) < 0.05 and abs(centres[1][1] - 10.0) < 0.05


def test_perfect_prediction_gives_ap_one_and_offsets_matter():
    actors = {"1": _actor(10.0, 1.5), "2": _actor(6.0, -4.0)}
    grid = upstream.traffic_grid_target(_meas(), actors)
    ap, info = objects.center_ap([grid], [grid], radius=1.0)
    assert ap == 1.0 and info["n_gt"] == 2
    # zeroing the offsets moves decoded centres to cell centres: still within 1 m of the actor
    # for the nearest cell, so AP stays 1 at 1 m but drops at 0.3 m
    g2 = grid.copy()
    g2[:, 1:3] = 0.0
    assert objects.center_ap([g2], [grid], radius=1.0)[0] == 1.0
    assert objects.center_ap([g2], [grid], radius=0.3)[0] < 1.0


def test_no_gt_returns_none_and_false_positives_reduce_ap():
    empty = np.zeros((400, 7), np.float32)
    assert objects.center_ap([empty], [empty])[0] is None
    actors = {"1": _actor(10.0, 1.5)}
    grid = upstream.traffic_grid_target(_meas(), actors)
    fp = grid.copy()
    fp[0, 0] = 1.0  # most confident detection, far from the actor (row 0 = 17.5 m ahead, col 0 = -9.5 m)
    ap, info = objects.center_ap([fp], [grid], radius=1.0)
    assert info["n_det"] == 2 and ap < 1.0
