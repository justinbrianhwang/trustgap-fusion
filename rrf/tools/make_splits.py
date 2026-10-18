"""Build the dataset manifest (JSONL with checksums) and the episode-level split file."""
import argparse
import hashlib
import json
from pathlib import Path

from rrf import data
from rrf.config import load_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(root, out_path, hash_images=True):
    root = Path(root)
    n = 0
    with open(out_path, "w") as f:
        for ep in data.episodes(root, include_excluded=True):
            for sid in data.sample_ids(root, ep["episode_id"]):
                frame = sid.split("/")[1]
                d = root / ep["episode_id"]
                meas = json.loads((d / "measurements" / f"{frame}.json").read_text())
                paths = {k: str(d / k / f"{frame}.{'npy' if k == 'lidar' else 'png'}") for k in ("rgb_front", "rgb_left", "rgb_right", "lidar")}
                paths["measurements"] = str(d / "measurements" / f"{frame}.json")
                paths["actors_data"] = str(d / "actors_data" / f"{frame}.json")
                rec = {"sample_id": sid, "episode_id": ep["episode_id"], "town": ep["town"], "world_frame": meas["world_frame"],
                       "simulation_timestamp": meas["simulation_timestamp"], "source_frames": meas["source_frames"],
                       "excluded_reason": ep["excluded_reason"], "paths": paths,
                       "sha256": {k: sha256(p) for k, p in paths.items()} if hash_images else {k: sha256(paths[k]) for k in ("measurements", "actors_data")},
                       "coordinate_schema": "planner_frame_v1: gps=(-y,x), theta=yaw+pi/2, lidar raw sensor frame (+x left,+y forward)"}
                f.write(json.dumps(rec) + "\n")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--no-image-hash", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--lock-from", default=None, help="earlier split.json whose validation/test groups are kept unchanged")
    mode.add_argument("--by-role", action="store_true", help="assign v2 episodes from the test/validation seed offsets")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg["paths"]["dataset_root"]
    man_dir = Path(cfg["paths"]["artifacts_root"]) / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    n = build_manifest(root, man_dir / "dataset_manifest.jsonl", hash_images=not args.no_image_hash)
    split = (data.make_splits_by_role(root, cfg["split"], man_dir / "split.json") if args.by_role else
             data.make_splits(root, cfg["split"], man_dir / "split.json", lock_from=args.lock_from))
    stats = {"samples": n, "manifest_sha256": sha256(man_dir / "dataset_manifest.jsonl"), "split_sha256": sha256(man_dir / "split.json"),
             "episodes": {k: {g: len(v) for g, v in split[k].items()} for k in ("train", "validation", "test")},
             "frames": {k: len(data.split_samples(root, split, k)) for k in ("train", "validation", "test")},
             "excluded": split["excluded_episodes"], "warnings": split["warnings"]}
    (man_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
