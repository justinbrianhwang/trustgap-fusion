"""Export every stored result as flat CSV files under artifacts/csv/ for external analysis.

Files (one row = one observation; numbers are stored in [0,1] or physical units, never %):
  episodes.csv        one row per collected episode (town, seed, frames, collisions, split, excluded)
  samples.csv         one row per stored frame (episode, split, world_frame, speed, fg cells, n actors)
  conditions.csv      one row per (method, seed, suite, split, condition): AP, losses, latency, q means
  per_episode.csv     one row per (method, seed, suite, split, condition, episode): episode AP
  frames.csv          one row per (method, seed, suite, split, condition, sample): q, losses, realized corruption
  bootstrap.csv       episode-paired bootstrap of method − B2 per condition and seed
  seed_deltas.csv     macro-AP differences method − B2 per seed and kind
  operating_point.csv fixed-threshold FN/FP per method and condition
  reliability.csv     per-(method, seed, block) reliability metrics, q-by-severity rows
  training.csv        every logged training step and validation eval of every run
Written by other tools into the same folder:
  reliability_rows_<method>_seed<s>.csv  one row per (condition, sample): q, reference error E, q*, fused loss (rrf.tools.reliability_report)
  weather.csv         native-weather track, one row per (method, seed, weather, episode) (rrf.tools.evaluate_weather)
  object_ap.csv       class-agnostic object-centre AP per (method, seed, condition) (rrf.tools.object_ap)
  closed_loop.csv     one row per closed-loop run (rrf.closed_loop)
"""
import argparse
import csv
import json
from pathlib import Path

from rrf import data, evaluate
from rrf.config import load_config


def write(path, rows, fields=None):
    rows = list(rows)
    if not rows:
        return 0
    fields = fields or list({k: None for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def split_of(split, ep):
    for top, groups in split.items():
        if isinstance(groups, dict):
            for g, eps in groups.items():
                if ep in eps:
                    return top, g
    return None, None


def run_dirs(art):
    for d in sorted((art / "metrics").glob("*_seed*_*")):
        if not (d / "metrics.json").exists():
            continue
        name = d.name
        method, rest = name.split("_seed", 1)
        seed, rest = rest.split("_", 1)
        suite, split = rest.split("_", 1)
        yield d, method, int(seed), suite, split


def spec_cols(spec):
    cam, lid = spec.get("camera"), spec.get("lidar")
    return {"camera_family": cam[0] if cam else "", "camera_severity": cam[1] if cam else 0,
            "lidar_family": lid[0] if lid else "", "lidar_severity": lid[1] if lid else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--no-frames", action="store_true", help="skip the large frames.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    art = Path(cfg["paths"]["artifacts_root"])
    root = cfg["paths"]["dataset_root"]
    out = art / "csv"
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((art / "manifests" / "split.json").read_text())
    counts = {}

    # episodes + samples
    eps = data.episodes(root, include_excluded=True)
    rows = []
    for e in eps:
        top, grp = split_of(split, e["episode_id"])
        rows.append({"episode_id": e["episode_id"], "town": e["town"], "seed": e["seed"], "frames": e["frames"], "ticks": e["ticks"],
                     "wall_seconds": e["wall_seconds"], "npc_vehicles": e["npc_vehicles"], "npc_walkers": e["npc_walkers"],
                     "collision_events": len(e.get("collision_frames", [])), "collision_tick_fraction": e["collision_tick_fraction"],
                     "excluded_reason": e["excluded_reason"] or "", "split": top or "", "split_group": grp or "",
                     "weather": e["settings"]["weather"], "route_points": e.get("route_points"), "carla_version": e.get("carla_version")})
    counts["episodes.csv"] = write(out / "episodes.csv", rows)
    rows = []
    for e in eps:
        top, grp = split_of(split, e["episode_id"])
        for sid in data.sample_ids(root, e["episode_id"]):
            m = data.load_meta(root, sid)
            fg = int((data.load_gt(root, sid)[:, 0] > 0.01).sum())
            n_act = len(json.loads((Path(root) / e["episode_id"] / "actors_data" / f"{sid.split('/')[1]}.json").read_text()))
            rows.append({"sample_id": sid, "episode_id": e["episode_id"], "town": e["town"], "split": top or "", "split_group": grp or "",
                         "world_frame": m["world_frame"], "simulation_timestamp": m["simulation_timestamp"], "speed_mps": m["speed"],
                         "command": m["command"], "is_junction": m["is_junction"], "traffic_light_state": m.get("traffic_light_state", ""),
                         "collision": m.get("collision", False), "fg_cells": fg, "actors_within_50m": n_act, "lidar_points": None})
    counts["samples.csv"] = write(out / "samples.csv", rows)

    # conditions / per_episode / frames / bootstrap
    cond_rows, ep_rows, frame_rows = [], [], []
    for d, method, seed, suite, spl in run_dirs(art):
        res = json.loads((d / "metrics.json").read_text())
        meta = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        eligible = meta.get("provenance", {}).get("meta", {}).get("eligible", True)
        for cond, r in res.items():
            base = {"method": method, "seed": seed, "suite": suite, "split": spl, "condition": cond,
                    "kind": evaluate.condition_kind(r["spec"], cfg["corruption"]), **spec_cols(r["spec"]), "checkpoint_eligible": eligible}
            q = r.get("q_mean") or [None, None]
            cond_rows.append({**base, "grid_fg_ap": r["grid_fg_ap"], "grid_fg_ap_fg_present": r["grid_fg_ap_fg_present"],
                              "n_samples": r["n_samples"], "n_fg_cells": r["n_fg_cells"], "n_cells": r["n_cells"],
                              "score_l1": r["native_loss"]["score"], "geometry_l1": r["native_loss"]["geometry"], "speed_l1": r["native_loss"]["speed"],
                              "traffic_combined_l1": r["native_loss"]["traffic_combined"],
                              "latency_ms_median": r["latency_ms_per_sample"]["median"], "latency_ms_p95": r["latency_ms_per_sample"]["p95"],
                              "q_camera_mean": q[0], "q_lidar_mean": q[1]})
            for ep, v in r["per_episode_ap"].items():
                ep_rows.append({**base, "episode_id": ep, "grid_fg_ap": v})
        if not args.no_frames and (d / "frames.jsonl").exists():
            for line in (d / "frames.jsonl").read_text().splitlines():
                f = json.loads(line)
                c = f.get("corruption") or {}
                cam, lid = c.get("camera") or {}, c.get("lidar") or {}
                q = f.get("q") or [None, None]
                frame_rows.append({"method": method, "seed": seed, "suite": suite, "split": spl, "condition": f["condition"],
                                   "sample_id": f["sample_id"], "episode_id": f["episode_id"], "fg_cells": f["fg_cells"],
                                   "q_camera": q[0], "q_lidar": q[1], "score_l1": f["native_loss"]["score"],
                                   "geometry_l1": f["native_loss"]["geometry"], "speed_l1": f["native_loss"]["speed"],
                                   "camera_family": cam.get("family", ""), "camera_severity": cam.get("severity", 0),
                                   "lidar_family": lid.get("family", ""), "lidar_severity": lid.get("severity", 0),
                                   "lidar_n_in": lid.get("n_in"), "lidar_n_out": lid.get("n_out"), "lidar_realized_rate": lid.get("realized_rate"),
                                   "lidar_sector_center_deg": lid.get("center_deg"), "lidar_removed_in_roi": lid.get("removed_in_roi")})
    counts["conditions.csv"] = write(out / "conditions.csv", cond_rows)
    counts["per_episode.csv"] = write(out / "per_episode.csv", ep_rows)
    if frame_rows:
        counts["frames.csv"] = write(out / "frames.csv", frame_rows)

    summ = art / "metrics" / "summary_full_test.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        rows = []
        for key, conds in s.get("bootstrap", {}).items():
            method, rest = key.split("_seed", 1)
            seed, base = rest.split("_minus_")
            for cond, b in conds.items():
                rows.append({"method": method, "seed": int(seed), "baseline": base, "condition": cond, "delta_ap": b["delta"],
                             "ci95_low": b["ci95"][0], "ci95_high": b["ci95"][1], "episodes": b["groups"], "repeats": b["repeats"]})
        counts["bootstrap.csv"] = write(out / "bootstrap.csv", rows)
        rows = [{"method": m, "seed": int(sd), "baseline": "B2", "kind": k, "delta_macro_ap": v}
                for m, seeds in s.get("seed_deltas", {}).items() for sd, d in seeds.items() for k, v in d.items()]
        counts["seed_deltas.csv"] = write(out / "seed_deltas.csv", rows)

    rows = []
    for p in sorted((art / "metrics").glob("operating_point_seed*.json")):
        seed = int(p.stem.split("seed")[1])
        for m, r in json.loads(p.read_text()).items():
            if "conditions" not in r:
                rows.append({"method": m, "seed": seed, "status": r.get("status", "not run")}); continue
            for cond, c in r["conditions"].items():
                rows.append({"method": m, "seed": seed, "status": "ok", "threshold": r["threshold"], "target_precision": r["target_precision"],
                             "condition": cond, **c})
    counts["operating_point.csv"] = write(out / "operating_point.csv", rows)

    rows = []
    for p in sorted((art / "metrics").glob("reliability_*_seed*.json")):
        r = json.loads(p.read_text())
        for block in ("overall", "foreground_present", "foreground_absent"):
            b = r[block]
            rows.append({"method": r["method"], "seed": r["seed"], "target": r.get("target", "absolute"), "row_type": "block", "block": block, "n": b.get("n"),
                         "mae_camera": b.get("mae_camera"), "mae_lidar": b.get("mae_lidar"),
                         "spearman_camera_vs_negE": b.get("spearman_camera_vs_negE"), "spearman_lidar_vs_negE": b.get("spearman_lidar_vs_negE"),
                         "spearman_qmin_vs_neg_fused_score_l1": b.get("spearman_qmin_vs_neg_fused_score_l1"),
                         "ranking_accuracy": b.get("ranking_accuracy"), "ranking_pairs": b.get("ranking_pairs")})
            for mod in ("camera", "lidar"):
                for i, t in enumerate(b.get(f"quantiles_{mod}", [])):
                    rows.append({"method": r["method"], "seed": r["seed"], "target": r.get("target", "absolute"), "row_type": f"quantile_{mod}", "block": block, "quantile": i,
                                 "q_low": t["q_range"][0], "q_high": t["q_range"][1], "n": t["n"], "mean_E": t["mean_E"], "mean_fused_score_l1": t["mean_fused_score_l1"]})
        for fam, d in r["q_by_family_severity"].items():
            for sev, v in d.items():
                rows.append({"method": r["method"], "seed": r["seed"], "target": r.get("target", "absolute"), "row_type": "q_by_severity", "family": fam, "severity": int(sev), "n": v["n"],
                             "q_camera_mean": v["q_mean"][0], "q_lidar_mean": v["q_mean"][1], "q_camera_std": v["q_std"][0], "q_lidar_std": v["q_std"][1]})
        rows.append({"method": r["method"], "seed": r["seed"], "target": r.get("target", "absolute"), "row_type": "q_by_severity", "family": "clean", "severity": 0, "n": r["q_clean"]["n"],
                     "q_camera_mean": r["q_clean"]["q_mean"][0], "q_lidar_mean": r["q_clean"]["q_mean"][1]})
    counts["reliability.csv"] = write(out / "reliability.csv", rows)

    rows = []
    for log in sorted((art / "runs").glob("*/log.jsonl")):
        run = log.parent.name
        for line in log.read_text().splitlines():
            d = json.loads(line)
            if "eval" in d:
                e = d["eval"]
                if "summary" in e:
                    s = e["summary"]
                    rows.append({"run": run, "row_type": "eval", "step": e["step"], "val_clean_ap": s["clean_ap"], "val_corruption_macro_ap": s["corruption_macro_ap"],
                                 **{f"val_ap_{c}": v for c, v in s["per_condition_ap"].items()}, "peak_gpu_mem_gb": e.get("peak_gpu_mem_gb")})
                elif "overall" in e:
                    o, f = e["overall"], e["foreground_present"]
                    rows.append({"run": run, "row_type": "eval", "step": e["step"], "rel_mae_all": o.get("mae"), "rel_mae_fg": f.get("mae"),
                                 "rel_spearman_cam_fg": f.get("spearman_camera"), "rel_spearman_lid_fg": f.get("spearman_lidar"), "rel_rank_acc_fg": f.get("ranking_accuracy")})
            else:
                rows.append({"run": run, "row_type": "train", "step": d["step"], "loss": d.get("loss"), "task_loss": d.get("task"), "quality_loss": d.get("quality"),
                             "elapsed_s": d.get("elapsed_s"), "q_camera_mean": (d.get("q_mean") or [None, None])[0], "q_lidar_mean": (d.get("q_mean") or [None, None])[1]})
    counts["training.csv"] = write(out / "training.csv", rows)
    (out / "README.md").write_text(__doc__ + "\nRow counts: " + json.dumps(counts) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
