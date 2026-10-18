"""Dataset access: raw sample loading, manifest, episode-level splits, cached GT targets."""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

from rrf import upstream

RAW_KEYS = ("rgb_front", "rgb_left", "rgb_right", "lidar")


MAX_COLLISION_TICK_FRACTION = 0.25  # episodes stuck in a collision longer than this are excluded


def episodes(dataset_root, include_excluded=False):
    root = Path(dataset_root)
    out = []
    for ep_json in sorted(root.glob("*/episode.json")):
        info = json.loads(ep_json.read_text())
        if info.get("frames", 0) == 0:
            continue
        frac = len(info.get("collision_frames", [])) / max(info.get("ticks", 1), 1)
        info["collision_tick_fraction"] = round(frac, 3)
        info["excluded_reason"] = None if frac <= MAX_COLLISION_TICK_FRACTION else f"collision_tick_fraction {frac:.2f} > {MAX_COLLISION_TICK_FRACTION}"
        if info["excluded_reason"] is None or include_excluded:
            out.append(info)
    return out


def sample_ids(dataset_root, episode_id):
    d = Path(dataset_root) / episode_id / "measurements"
    return [f"{episode_id}/{p.stem}" for p in sorted(d.glob("*.json"))]


def load_raw(dataset_root, sample_id):
    """Raw observation + model-side scalars. GT and metadata are loaded separately."""
    ep, frame = sample_id.split("/")
    d = Path(dataset_root) / ep
    meas = json.loads((d / "measurements" / f"{frame}.json").read_text())
    raw = {
        "sample_id": sample_id,
        "rgb_front": np.asarray(Image.open(d / "rgb_front" / f"{frame}.png").convert("RGB")),
        "rgb_left": np.asarray(Image.open(d / "rgb_left" / f"{frame}.png").convert("RGB")),
        "rgb_right": np.asarray(Image.open(d / "rgb_right" / f"{frame}.png").convert("RGB")),
        "lidar": np.load(d / "lidar" / f"{frame}.npy"),
        "speed": float(meas["speed"]),
        "command": int(meas["command"]),
        "target_point": target_point(meas),
    }
    return raw


def target_point(meas):
    """Same formula as upstream carla_dataset / interfuser_agent (planner frame)."""
    theta = meas["theta"] + np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = R.T.dot(np.array([meas["x_command"] - meas["gps_x"], meas["y_command"] - meas["gps_y"]]))
    return local.astype(np.float32)


def load_gt(dataset_root, sample_id, write_cache=True):
    """(400,7) upstream soft target, cached next to the sample. Cache key = hash of the two
    JSON files + upstream commit, so a changed label definition invalidates it. With
    write_cache=False an existing cache is still read, but a miss leaves the dataset untouched."""
    ep, frame = sample_id.split("/")
    d = Path(dataset_root) / ep
    m_bytes = (d / "measurements" / f"{frame}.json").read_bytes()
    a_bytes = (d / "actors_data" / f"{frame}.json").read_bytes()
    key = hashlib.sha256(m_bytes + a_bytes + b"target_v1").hexdigest()[:16]
    cache_dir = d / "targets"
    cache = cache_dir / f"{frame}_{key}.npy"
    if cache.exists():
        try:
            t = np.load(cache)
            if t.shape == (400, 7):
                return t
        except (ValueError, EOFError, OSError):
            pass  # partially written by a concurrent process: recompute below
    t = upstream.traffic_grid_target(json.loads(m_bytes), json.loads(a_bytes))
    if not write_cache:
        return t
    cache_dir.mkdir(exist_ok=True)
    tmp = cache_dir / f".{frame}_{key}.{os.getpid()}.tmp.npy"
    np.save(tmp, t)
    try:
        os.replace(tmp, cache)  # atomic on the same volume; concurrent writers produce identical files
    except OSError:
        tmp.unlink(missing_ok=True)
    return t


def load_meta(dataset_root, sample_id):
    ep, frame = sample_id.split("/")
    return json.loads((Path(dataset_root) / ep / "measurements" / f"{frame}.json").read_text())


# ------------------------------------------------------------------ splits
def make_splits(dataset_root, cfg_split, out_path, lock_from=None):
    """Episode-level split with nested probe/reliability and model-select/calibration groups.
    Deterministic in split_seed. Refuses to overwrite an existing split file. With `lock_from`
    (an earlier split file) the validation and test groups are copied unchanged and every other
    valid episode goes to train (dataset extension without touching the locked test set)."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; splits are read-only once used")
    all_eps = episodes(dataset_root, include_excluded=True)
    excluded = {e["episode_id"]: e["excluded_reason"] for e in all_eps if e["excluded_reason"]}
    eps = [e["episode_id"] for e in all_eps if not e["excluded_reason"]]
    if len(eps) < 5:
        raise RuntimeError(f"need >= 5 episodes for a split, found {len(eps)}")
    rng = np.random.default_rng(cfg_split["split_seed"])
    if lock_from:
        prev = json.loads(Path(lock_from).read_text())
        locked = set(sum(prev["validation"].values(), [])) | set(prev["test"]["locked_groups"])
        train = sorted(e for e in eps if e not in locked)
        rng.shuffle(train)
        n_probe = max(1, int(round(len(train) * cfg_split["probe_fraction_within_train"])))
        split = {"split_seed": cfg_split["split_seed"], "unit": "episode", "pretraining_overlap": cfg_split["pretraining_overlap"],
                 "locked_from": str(lock_from),
                 "train": {"probe_fit_groups": sorted(train[:n_probe]), "reliability_fit_groups": sorted(train[n_probe:])},
                 "validation": prev["validation"], "test": prev["test"], "excluded_episodes": excluded, "warnings": []}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(split, indent=2))
        return split
    # global episode-level allocation (towns are interleaved by sorting per town before the
    # shuffle so every split tends to contain several towns; exact stratification is not
    # possible with 2-5 episodes per town)
    eps = sorted(eps)
    rng.shuffle(eps)
    n = len(eps)
    n_test = max(1, int(round(n * cfg_split["test_fraction"])))
    n_val = max(1, int(round(n * cfg_split["validation_fraction"])))
    test, val, train = eps[:n_test], eps[n_test:n_test + n_val], eps[n_test + n_val:]
    n_probe = max(1, int(round(len(train) * cfg_split["probe_fraction_within_train"])))
    rng.shuffle(val)
    n_cal = max(1, int(round(len(val) * cfg_split["calibration_fraction_within_validation"]))) if len(val) > 1 else 0
    split = {
        "split_seed": cfg_split["split_seed"], "unit": "episode", "pretraining_overlap": cfg_split["pretraining_overlap"],
        "train": {"probe_fit_groups": sorted(train[:n_probe]), "reliability_fit_groups": sorted(train[n_probe:])},
        "validation": {"model_select_groups": sorted(val[n_cal:]), "calibration_groups": sorted(val[:n_cal])},
        "test": {"locked_groups": sorted(test)},
        "excluded_episodes": excluded,
        "warnings": [],
    }
    for name, groups in (("train", train), ("validation", val), ("test", test)):
        if len(groups) < 3:
            split["warnings"].append(f"{name} has only {len(groups)} episodes; statistics will be weak")
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2))
    return split


_EPISODE_NAME = re.compile(r"^(?P<town>.+)_ep\d+_seed(?P<seed>\d+)$")


def _town_hash(town):
    return sum(ord(c) for c in town) % 97


def _role_split_statistics(dataset_root, groups):
    stats = {"frame_stride": 10}
    for split_name, episode_ids in groups.items():
        towns = {}
        foreground = {}
        for episode_id in episode_ids:
            match = _EPISODE_NAME.fullmatch(episode_id)
            town = match.group("town") if match else episode_id.split("_ep", 1)[0]
            towns[town] = towns.get(town, 0) + 1
            counts = [int(upstream.foreground_label(load_gt(dataset_root, sid, write_cache=False)).sum())
                      for sid in sample_ids(dataset_root, episode_id)[::10]]
            foreground[episode_id] = float(np.mean(counts)) if counts else None
        density = [value for value in foreground.values() if value is not None]
        stats[split_name] = {
            "episodes_per_town": dict(sorted(towns.items())),
            "mean_foreground_cells_per_episode": float(np.mean(density)) if density else None,
            "mean_foreground_cells_by_episode": dict(sorted(foreground.items())),
        }
    return stats


def make_splits_by_role(dataset_root, cfg_split, out_path, test_offset=500, val_offset=600):
    """Split v2 episodes by the role encoded in their seed; incomplete collections are valid."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; splits are read-only once used")
    all_eps = episodes(dataset_root, include_excluded=True)
    excluded = {e["episode_id"]: e["excluded_reason"] for e in all_eps if e["excluded_reason"]}
    train, model_select, calibration, test = [], [], [], []
    for episode in all_eps:
        episode_id = episode["episode_id"]
        if episode["excluded_reason"]:
            continue
        match = _EPISODE_NAME.fullmatch(episode_id)
        role = "train"
        if match:
            base = _town_hash(match.group("town")) * 100
            seed = int(match.group("seed"))
            if 0 <= seed - base - test_offset <= 3:
                role = "test"
            elif seed - base - val_offset == 0:
                role = "model_select"
            elif seed - base - val_offset == 1:
                role = "calibration"
        {"train": train, "model_select": model_select, "calibration": calibration, "test": test}[role].append(episode_id)

    rng = np.random.default_rng(cfg_split["split_seed"])
    rng.shuffle(train)
    n_probe = max(1, int(round(len(train) * cfg_split["probe_fraction_within_train"]))) if train else 0
    split = {
        "split_seed": cfg_split["split_seed"], "unit": "episode", "pretraining_overlap": cfg_split["pretraining_overlap"],
        "role_offsets": {"test": test_offset, "validation": val_offset},
        "train": {"probe_fit_groups": sorted(train[:n_probe]), "reliability_fit_groups": sorted(train[n_probe:])},
        "validation": {"model_select_groups": sorted(model_select), "calibration_groups": sorted(calibration)},
        "test": {"locked_groups": sorted(test)},
        "excluded_episodes": excluded,
        "warnings": [],
    }
    groups = {"train": train, "validation": model_select + calibration, "test": test}
    for name, episode_ids in groups.items():
        if len(episode_ids) < 3:
            split["warnings"].append(f"{name} has only {len(episode_ids)} episodes; collection may be incomplete")
    split["statistics"] = _role_split_statistics(dataset_root, groups)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2))
    return split


def split_episodes(split, name):
    """All episodes of a top-level split (train/validation/test)."""
    return sorted(set(sum(split[name].values(), [])))


def split_samples(dataset_root, split, name, subgroup=None, max_frames=None, seed=0):
    eps = split[name][subgroup] if subgroup else split_episodes(split, name)
    ids = [s for e in eps for s in sample_ids(dataset_root, e)]
    if max_frames and len(ids) > max_frames:
        # keep episode grouping: subsample uniformly within each episode
        keep = max_frames / len(ids)
        rng = np.random.default_rng(seed)
        ids = [s for s in ids if rng.random() < keep]
    return ids
