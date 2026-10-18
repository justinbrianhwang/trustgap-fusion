import numpy as np

from rrf import metrics as M


def test_perfect_prediction_has_ap_one():
    labels = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.0])
    assert M.grid_fg_ap(scores, labels) == 1.0


def test_ap_known_value():
    labels = np.array([1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7])  # precisions at positives: 1.0, 2/3
    assert abs(M.grid_fg_ap(scores, labels) - (1.0 + 2 / 3) / 2) < 1e-12


def test_no_positive_returns_none_not_one():
    assert M.grid_fg_ap(np.array([0.1, 0.2]), np.array([0, 0])) is None
    assert M.calibrate_threshold(np.array([0.1, 0.2]), np.array([0, 0])) is None


def test_all_background_and_all_foreground_predictors():
    labels = np.array([1, 1, 0, 0])
    bg = M.confusion(np.zeros(4), labels, threshold=0.5)
    assert bg["fn"] == 2 and bg["fp"] == 0 and bg["fnr"] == 1.0 and bg["precision"] is None
    fg = M.confusion(np.ones(4), labels, threshold=0.5)
    assert fg["fp"] == 2 and fg["fn"] == 0 and fg["fpr"] == 1.0 and fg["recall"] == 1.0


def test_calibration_picks_highest_recall_at_target_precision():
    labels = np.array([1, 1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    # top-2 precision 1.0; top-3 0.667; top-4 0.75 -> target 0.75 accepts top-4 (threshold 0.6)
    assert M.calibrate_threshold(scores, labels, 0.75) == 0.6
    assert M.calibrate_threshold(scores, labels, 0.9) == 0.8
    assert M.calibrate_threshold(scores, labels, 1.01) is None


def test_native_losses_empty_foreground_is_zero_not_crash():
    out = np.full((400, 7), 0.5, np.float32)
    tgt = np.zeros((400, 7), np.float32)
    l = M.native_traffic_losses(out, tgt)
    assert l["geometry"] == 0.0 and l["speed"] == 0.0 and abs(l["score"] - 0.25) < 1e-6


def test_bootstrap_is_group_level():
    a = np.array([0.5, 0.6, 0.7])
    b = a + 0.1
    r = M.paired_bootstrap(a, b, repeats=200, seed=1)
    assert abs(r["delta"] - 0.1) < 1e-9 and r["groups"] == 3
    assert r["ci95"][0] <= 0.1 <= r["ci95"][1]
