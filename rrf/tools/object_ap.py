"""Class-agnostic object-centre AP (md §12.4, optional) from the stored grid predictions of
every full-test run. Peak extraction + NMS radius are fixed (validation-free defaults, documented
here): score threshold 0.05, NMS 2 m, match radius 1 m (and 2 m as a secondary column).
Outputs artifacts/csv/object_ap.csv.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rrf import data, evaluate, objects
from rrf.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--radii", nargs="+", type=float, default=[1.0, 2.0])
    args = ap.parse_args()
    cfg = load_config(args.config)
    art = Path(cfg["paths"]["artifacts_root"])
    root = cfg["paths"]["dataset_root"]
    gt_cache = {}
    rows = []
    for d in sorted((art / "metrics").glob("*_full_test")):
        method, seed = d.name.split("_seed")[0], int(d.name.split("_seed")[1].split("_")[0])
        for f in sorted(d.glob("pred_*.npz")):
            cond = f.stem[5:]
            p = np.load(f)
            ids = list(p["sample_ids"])
            for sid in ids:
                if sid not in gt_cache:
                    gt_cache[sid] = data.load_gt(root, sid)
            gts = [gt_cache[s] for s in ids]
            spec = json.loads((d / "metrics.json").read_text())[cond]["spec"]
            row = {"method": method, "seed": seed, "condition": cond, "kind": evaluate.condition_kind(spec, cfg["corruption"]), "n_samples": len(ids)}
            for r in args.radii:
                apv, info = objects.center_ap(p["traffic"], gts, radius=r)
                row[f"center_ap@{r:g}m"] = apv
                row["n_gt_objects"] = info["n_gt"]
                row[f"n_detections"] = info["n_det"]
                row[f"recall_all_dets@{r:g}m"] = info.get("recall_at_all_dets")
            rows.append(row)
            print(row, flush=True)
    out = art / "csv" / "object_ap.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list({k: None for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    main()
