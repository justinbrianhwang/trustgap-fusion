"""Fixed operating point per method: threshold calibrated on the clean *calibration* split
(precision >= target), applied unchanged to every test condition.

Requires: evaluate --suite clean --split validation --subgroup calibration_groups
      and evaluate --suite full --split test   for the method.
"""
import argparse
import json
from pathlib import Path

from rrf import evaluate
from rrf.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--methods", nargs="+", default=["B0", "B2", "P"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    art = Path(cfg["paths"]["artifacts_root"]) / "metrics"
    out = {}
    for m in args.methods:
        cal = art / f"{m}_seed{args.seed}_clean_validation-calibration_groups"
        test = art / f"{m}_seed{args.seed}_full_test"
        if not (cal / "pred_clean.npz").exists() or not (test / "metrics.json").exists():
            out[m] = {"status": "not run", "calibration_dir": str(cal), "test_dir": str(test)}
            continue
        out[m] = evaluate.operating_point(cal, test, cfg["metrics"]["clean_calibration_precision_target"], cfg["paths"]["dataset_root"])
    (art / f"operating_point_seed{args.seed}.json").write_text(json.dumps(out, indent=2))
    for m, r in out.items():
        if "conditions" in r:
            print(m, "threshold", r["threshold"], {c: (v["fnr"], v["fpr"]) for c, v in list(r["conditions"].items())[:4]}, "...")
        else:
            print(m, r)


if __name__ == "__main__":
    main()
