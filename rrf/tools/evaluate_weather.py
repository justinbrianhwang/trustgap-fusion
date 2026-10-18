"""Native-weather validation track (md §13.3): evaluate methods on the test episodes re-collected
under CARLA weather presets (no synthetic corruption). Each weather episode is compared with its
ClearNoon counterpart (same town/seed/route/traffic seed) as an *episode-group* comparison, not a
frame-paired one (geometry may differ). Outputs artifacts/metrics/weather_<method>_seed<seed>.json
and artifacts/csv/weather.csv (all methods).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from rrf import data, evaluate, upstream
from rrf.config import load_config
from rrf.tools.evaluate import load_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--weather-root", default="dataset_weather")
    ap.add_argument("--clear-root", default=None, help="root holding the ClearNoon counterparts (default: paths.dataset_root)")
    ap.add_argument("--out-tag", default="", help="suffix for weather_<method>_seed<s><tag>.json and weather<tag>.csv")
    ap.add_argument("--frame-stride", type=int, default=1, help="evaluate every n-th frame of each episode (frames are 10 Hz; 4 -> 2.5 Hz)")
    ap.add_argument("--methods", nargs="+", default=["B0", "B2", "P"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = Path(cfg["paths"]["artifacts_root"])
    wroot = Path(args.weather_root)
    croot = args.clear_root or cfg["paths"]["dataset_root"]
    weps = [e for e in data.episodes(wroot, include_excluded=True) if "__" in e["episode_id"]]
    if not weps:
        print("no weather episodes"); return
    groups = {}  # weather name -> [(weather episode id, clean counterpart id)]
    for e in weps:
        base, w = e["episode_id"].split("__")
        groups.setdefault(w, []).append((e["episode_id"], base, e))
    rows = []
    for method in args.methods:
        for seed in (args.seeds if method != "B0" else [0]):
            ck = art / "checkpoints" / f"fusion_{method}_seed{seed}.pt"
            if method != "B0" and not ck.exists():
                continue
            gate, prov = load_method(cfg, method, seed, device)
            gate.model.eval()
            if gate.estimator is not None:
                gate.estimator.eval()
            report = {"method": method, "seed": seed, "weather": {}, "frame_stride": args.frame_stride}
            ref_cache = {}  # ClearNoon counterpart evaluated once per (method, seed) and set of base episodes
            for w, items in sorted(groups.items()):
                ids = [s for eid, _, e in items for s in data.sample_ids(wroot, eid)[::args.frame_stride]]
                res = evaluate.run_suite(lambda b: gate(b), str(wroot), ids, {"clean": {}}, cfg["corruption"]["base_seed"],
                                         art / "runs" / "_weather_tmp", batch_size=16, device=device, save_predictions=False, log_every=10**9)["clean"]
                clean_ids = [s for _, base, _ in items for s in data.sample_ids(croot, base)[::args.frame_stride]]
                key = tuple(clean_ids)
                if key not in ref_cache:
                    ref_cache[key] = evaluate.run_suite(lambda b: gate(b), croot, clean_ids, {"clean": {}}, cfg["corruption"]["base_seed"],
                                                        art / "runs" / "_weather_tmp", batch_size=16, device=device, save_predictions=False, log_every=10**9)["clean"]
                ref = ref_cache[key]
                rep = {"grid_fg_ap": res["grid_fg_ap"], "grid_fg_ap_fg_present": res["grid_fg_ap_fg_present"], "per_episode_ap": res["per_episode_ap"],
                       "score_l1": res["native_loss"]["score"], "q_mean": res["q_mean"], "n_samples": len(ids), "n_fg_cells": res["n_fg_cells"],
                       "clear_noon_counterpart": {"grid_fg_ap": ref["grid_fg_ap"], "per_episode_ap": ref["per_episode_ap"], "q_mean": ref["q_mean"], "n_fg_cells": ref["n_fg_cells"]},
                       "weather_parameters": items[0][2]["settings"].get("weather_parameters"),
                       "excluded_episodes": {eid: e["excluded_reason"] for eid, _, e in items if e["excluded_reason"]}}
                report["weather"][w] = rep
                for eid, base, e in items:
                    rows.append({"method": method, "seed": seed, "weather": w, "episode_id": eid, "clean_episode_id": base, "town": e["town"],
                                 "grid_fg_ap_weather": res["per_episode_ap"].get(eid), "grid_fg_ap_clear_noon": ref["per_episode_ap"].get(base),
                                 "collision_events_weather": len(e.get("collision_frames", [])), "excluded_reason": e["excluded_reason"] or "",
                                 "q_camera_weather": (res["q_mean"] or [None, None])[0], "q_lidar_weather": (res["q_mean"] or [None, None])[1],
                                 "q_camera_clear": (ref["q_mean"] or [None, None])[0], "q_lidar_clear": (ref["q_mean"] or [None, None])[1]})
                print(f"[{method} s{seed}] {w}: AP {res['grid_fg_ap']:.3f} (ClearNoon counterpart {ref['grid_fg_ap']:.3f}) q {res['q_mean']}", flush=True)
            (art / "metrics" / f"weather_{method}_seed{seed}{args.out_tag}.json").write_text(json.dumps(report, indent=2))
    out = art / "csv" / f"weather{args.out_tag}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
