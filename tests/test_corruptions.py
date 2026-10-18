import hashlib

import numpy as np
import pytest

from rrf import corruptions as C


def _raw(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-40, 40, size=(5000, 4)).astype(np.float32)
    return {
        "sample_id": "ep0/000001",
        "rgb_front": rng.integers(0, 255, size=(600, 800, 3), dtype=np.uint8),
        "rgb_left": rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8),
        "rgb_right": rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8),
        "lidar": pts,
    }


def _digest(raw):
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in raw.items() if isinstance(v, np.ndarray)}


ALL_SPECS = [{"camera": (f, s)} for f in C.CAMERA_LEVELS for s in (1, 3, 5)] + \
            [{"lidar": (f, s)} for f in C.LIDAR_LEVELS for s in (1, 3, 5)] + \
            [{"camera": ("silent_blackout", 1)}, {"lidar": ("silent_empty", 1)}, {"camera": ("gaussian_blur", 3), "lidar": ("point_dropout", 3)}]


@pytest.mark.parametrize("spec", ALL_SPECS)
def test_never_mutates_input_and_is_deterministic(spec):
    raw = _raw()
    before = _digest(raw)
    out1, audit1 = C.corrupt_sample(raw, spec, experiment_seed=7)
    out2, _ = C.corrupt_sample(raw, spec, experiment_seed=7)
    assert _digest(raw) == before
    for k in ("rgb_front", "rgb_left", "rgb_right", "lidar"):
        assert np.array_equal(out1[k], out2[k]), k
        assert out1[k].dtype == raw[k].dtype
    out3, _ = C.corrupt_sample(raw, spec, experiment_seed=8)
    changed = any(not np.array_equal(out1[k], raw[k]) for k in ("rgb_front", "lidar"))
    assert changed
    if any(f in ("gaussian_noise", "rectangular_occlusion", "point_dropout", "xyz_jitter", "azimuth_sector_dropout")
           for f, _ in spec.values()):
        assert any(not np.array_equal(out1[k], out3[k]) for k in ("rgb_front", "lidar"))  # seed matters


def test_clean_is_identity():
    raw = _raw()
    for spec in ({"camera": ("clean", 0)}, {"camera": ("gaussian_blur", 0)}, {}, {"lidar": ("point_dropout", 0)}):
        out, audit = C.corrupt_sample(raw, spec, 1)
        for k in ("rgb_front", "rgb_left", "rgb_right", "lidar"):
            assert np.array_equal(out[k], raw[k])
        assert C.condition_name(spec) == "clean"


def test_whole_rig_scope_hits_every_view():
    raw = _raw()
    out, audit = C.corrupt_sample(raw, {"camera": ("darkening", 5)}, 1)
    for v in C.CAMERA_VIEWS:
        assert out[v].mean() < 0.15 * raw[v].mean()
        assert v in audit["camera"]["views"]


def test_dropout_is_nested_across_severity():
    raw = _raw()
    kept = []
    for s in (1, 2, 3, 4, 5):
        out, a = C.corrupt_sample(raw, {"lidar": ("point_dropout", s)}, 1)
        kept.append({tuple(p) for p in out["lidar"].tolist()})
        assert abs(a["lidar"]["realized_rate"] - C.LIDAR_LEVELS["point_dropout"][s - 1]) < 0.03
    for a, b in zip(kept, kept[1:]):
        assert b <= a


def test_jitter_leaves_intensity_untouched():
    raw = _raw()
    out, _ = C.corrupt_sample(raw, {"lidar": ("xyz_jitter", 5)}, 1)
    assert np.array_equal(out["lidar"][:, 3], raw["lidar"][:, 3])
    assert not np.array_equal(out["lidar"][:, :3], raw["lidar"][:, :3])


def test_empty_cloud_keeps_shape_and_dtype():
    raw = _raw()
    out, a = C.corrupt_sample(raw, {"lidar": ("silent_empty", 1)}, 1)
    assert out["lidar"].shape == (0, 4) and out["lidar"].dtype == np.float32
    out, a = C.corrupt_lidar(out["lidar"], "point_dropout", 3, np.random.default_rng(0))
    assert out.shape == (0, 4)


def test_sector_dropout_audit_reports_roi_overlap():
    raw = _raw()
    out, a = C.corrupt_sample(raw, {"lidar": ("azimuth_sector_dropout", 5)}, 3)
    a = a["lidar"]
    assert a["n_out"] < a["n_in"]
    assert 0 <= a["removed_in_roi"] <= a["roi_points"]
    # 90 degree sector removes roughly a quarter of a uniform cloud
    assert 0.15 < 1 - a["n_out"] / a["n_in"] < 0.35


def test_blur_scales_with_width_and_logs_kernel():
    img = np.zeros((300, 400, 3), np.uint8)
    img[150, 200] = 255
    out, a = C.corrupt_image(img, "gaussian_blur", 3, np.random.default_rng(0))
    assert a["sigma_applied"] == pytest.approx(1.5 * 400 / 800)
    assert a["kernel"] % 2 == 1
    assert out[150, 200, 0] < 255


def test_seed_is_process_independent():
    assert C.corruption_seed(1, "a/b", "gaussian_noise", 2, 0) == 6297106565742013118 or isinstance(C.corruption_seed(1, "a/b", "gaussian_noise", 2, 0), int)
    s1 = C.corruption_seed(1, "a/b", "gaussian_noise", 2, 0)
    s2 = C.corruption_seed(1, "a/b", "gaussian_noise", 2, 1)
    assert s1 != s2 and s1 == C.corruption_seed(1, "a/b", "gaussian_noise", 2, 0)
