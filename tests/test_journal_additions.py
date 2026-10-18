import copy
import json
from pathlib import Path

import numpy as np
import pytest

from rrf import corruptions as C
from rrf import data
from rrf.config import ConfigError, load_config, validate
from rrf.tools.select_checkpoint import load_test_ap, select_from_log
from rrf.train import sample_condition


def test_make_splits_by_role(monkeypatch):
    root = Path(__file__).parent / "fixtures" / "role_split"
    town1, town2 = "Town01", "Town10HD_Opt"
    base1, base2 = data._town_hash(town1) * 100, data._town_hash(town2) * 100
    train = f"{town1}_ep00_seed{base1 + 20}"
    test0 = f"{town1}_ep01_seed{base1 + 500}"
    test3 = f"{town2}_ep02_seed{base2 + 503}"
    select1 = f"{town1}_ep03_seed{base1 + 600}"
    calibrate1 = f"{town1}_ep04_seed{base1 + 601}"
    select2 = f"{town2}_ep05_seed{base2 + 600}"
    calibrate2 = f"{town2}_ep06_seed{base2 + 601}"
    excluded = f"{town2}_ep07_seed{base2 + 501}"

    def fake_gt(_root, sample_id, write_cache=True):
        assert write_cache is False
        target = np.zeros((400, 7), dtype=np.float32)
        target[:5, 0] = 1
        return target

    monkeypatch.setattr(data, "load_gt", fake_gt)
    monkeypatch.setattr(data, "sample_ids", lambda _root, episode_id: [f"{episode_id}/0000"])
    cfg = load_config("configs/pilot.yaml")["split"]
    out = root / "generated_split.json"
    written = {}
    original_write_text = Path.write_text

    def capture_write(path, text, *args, **kwargs):
        if path == out:
            written["text"] = text
            return len(text)
        return original_write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", capture_write)
    split = data.make_splits_by_role(root, cfg, out)

    assert set(split["train"]["probe_fit_groups"] + split["train"]["reliability_fit_groups"]) == {train}
    assert split["validation"]["model_select_groups"] == sorted([select1, select2])
    assert split["validation"]["calibration_groups"] == sorted([calibrate1, calibrate2])
    assert split["test"]["locked_groups"] == sorted([test0, test3])
    assert excluded in split["excluded_episodes"] and excluded not in split["test"]["locked_groups"]
    assert split["statistics"]["validation"]["episodes_per_town"] == {town1: 2, town2: 2}
    assert split["statistics"]["train"]["mean_foreground_cells_per_episode"] == 5.0
    assert split["statistics"]["train"]["mean_foreground_cells_by_episode"][train] == 5.0
    assert json.loads(written["text"])["test"] == split["test"]
    with pytest.raises(FileExistsError):
        data.make_splits_by_role(root, cfg, root / "existing.json")


def _kind(spec):
    if not spec:
        return "clean"
    if len(spec) == 2:
        return "both_corrupted"
    return f"{next(iter(spec))}_only_corrupted"


def test_weak_train_profile_changes_schedule_distribution_but_is_deterministic():
    standard = load_config("configs/journal.yaml")["corruption"]
    weak = load_config("configs/journal_weak.yaml")["corruption"]
    standard_counts = {k: 0 for k in ("clean", "camera_only_corrupted", "lidar_only_corrupted", "both_corrupted")}
    weak_counts = copy.deepcopy(standard_counts)
    weak_specs = []
    for i in range(10000):
        seed = C.corruption_seed(7, f"episode/{i:04d}", "schedule", 3, 0)
        standard_counts[_kind(sample_condition(np.random.default_rng(seed), standard))] += 1
        spec = sample_condition(np.random.default_rng(seed), weak)
        weak_counts[_kind(spec)] += 1
        weak_specs.append(spec)
    assert weak_counts["clean"] / 10000 == pytest.approx(0.70, abs=0.02)
    assert standard_counts["clean"] / 10000 == pytest.approx(0.25, abs=0.02)
    assert all(severity == 1 for spec in weak_specs for _, severity in spec.values())
    seed = C.corruption_seed(9, "episode/0042", "schedule", 11, 0)
    assert sample_condition(np.random.default_rng(seed), weak) == sample_condition(np.random.default_rng(seed), weak)


def test_train_profile_config_schema_and_journal_configs():
    standard = load_config("configs/journal.yaml")
    weak = load_config("configs/journal_weak.yaml")
    assert standard["corruption"]["train_profile"] == "standard"
    assert weak["corruption"]["train_profile"] == "weak"
    assert weak["corruption"]["train_severities"] == [1]
    assert weak["corruption"]["train_condition_probabilities"] == {
        "clean": 0.70, "camera_only_corrupted": 0.12, "lidar_only_corrupted": 0.12, "both_corrupted": 0.06,
    }
    assert standard["paths"]["dataset_root"].endswith("dataset_v2")
    assert weak["paths"]["artifacts_root"].endswith("artifacts_journal_weak")
    bad = copy.deepcopy(standard)
    bad.pop("_meta")
    bad["corruption"]["train_profile"] = "strong"
    with pytest.raises(ConfigError):
        validate(bad)


def test_select_checkpoint_rules():
    log = Path(__file__).parent / "fixtures" / "checkpoint_log.jsonl"
    assert select_from_log(log, "val", 0.01)["step"] == 30
    assert select_from_log(log, "fixed_step", 0.01, fixed_step=20)["step"] == 20
    assert select_from_log(log, "early", 0.01)["step"] == 10
    assert select_from_log(log, "oracle_test", 0.01, test_ap={10: 0.1, 20: 0.4, 30: 0.3, 40: 0.2})["step"] == 20
    assert load_test_ap(Path(__file__).parent / "fixtures" / "checkpoint_test_ap.json") == {10: 0.1, 20: 0.4}
    fallback = select_from_log(log, "val", -0.1)
    assert fallback["step"] == 40 and fallback["eligible"] is False
