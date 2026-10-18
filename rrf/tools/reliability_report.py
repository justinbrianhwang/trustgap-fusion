"""Reliability analysis on the locked test suite for a gated method (default P).

For every (sample, condition) of the full test suite we recompute the reference errors
E_camera/E_lidar and q* from the frozen masked-upstream reference on the corrupted input
(analysis-only labels), and read the model's q from the stored evaluation frames. Reports
MAE, Spearman(q, -E), modality ranking accuracy, q by family/severity, q-quantile vs reference
error and vs the final fused native score loss, split by foreground presence.
Outputs artifacts/metrics/reliability_<method>_seed<seed>.json. With --target relative the
target is the relative one used to train PR (E = max(E_corrupt - E_clean, 0), q* = exp(-E/tau_relative);
E_clean from the clean counterpart of the same sample) and the output is reliability_rel_<method>_seed<seed>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from rrf import corruptions as C
from rrf import data, evaluate, upstream
from rrf.config import load_config
from rrf.probes import q_targets, q_targets_relative
from rrf.train import load_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--method", default="P")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="absolute", choices=["absolute", "relative"])
    args = ap.parse_args()
    rel = args.target == "relative"
    tag = "rel_" if rel else ""
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = cfg["paths"]["dataset_root"]
    art = Path(cfg["paths"]["artifacts_root"]) / "metrics"
    run = art / f"{args.method}_seed{args.seed}_full_test"
    frames = [json.loads(l) for l in (run / "frames.jsonl").read_text().splitlines()]
    frames = [f for f in frames if f["q"] is not None]
    if not frames:
        print("no q recorded for", args.method); return
    suite = evaluate.full_suite(cfg["corruption"])
    reference = load_reference(cfg, device)
    tau = cfg["reliability"]["tau_relative" if rel else "tau_quality"]
    rows = []
    by_cond = {}
    for f in frames:
        by_cond.setdefault(f["condition"], []).append(f)
    from rrf.evaluate import _ConditionSamples, _collate
    with torch.no_grad():
        for cond, fs in by_cond.items():
            spec = suite[cond]
            by_id = {f["sample_id"]: f for f in fs}
            ids = [f["sample_id"] for f in fs]
            loader = torch.utils.data.DataLoader(_ConditionSamples(root, ids, spec, cfg["corruption"]["base_seed"]), batch_size=16,
                                                 shuffle=False, num_workers=4, collate_fn=_collate)
            clean_loader = torch.utils.data.DataLoader(_ConditionSamples(root, ids, {}, cfg["corruption"]["base_seed"]), batch_size=16,
                                                       shuffle=False, num_workers=4, collate_fn=_collate) if rel else [None] * len(loader)
            for item, item0 in zip(loader, clean_loader):
                chunk = [by_id[s] for s in item["sample_id"]]
                batch = {k: item[k].to(device) for k in upstream.MODEL_INPUT_KEYS}
                fg = torch.stack([torch.from_numpy(upstream.foreground_label(data.load_gt(root, f["sample_id"]))) for f in chunk]).to(device)
                if rel:
                    assert item0["sample_id"] == item["sample_id"]
                    qs, e, _, _ = q_targets_relative(reference, batch, {k: item0[k].to(device) for k in upstream.MODEL_INPUT_KEYS}, fg, tau)
                else:
                    qs, e = q_targets(reference, batch, fg, tau)
                for k, f in enumerate(chunk):
                    rows.append({"condition": cond, "sample_id": f["sample_id"], "fg": bool(f["fg_cells"] > 0), "q": f["q"],
                                 "E": e[k].cpu().tolist(), "q_star": qs[k].cpu().tolist(), "fused_score_l1": f["native_loss"]["score"],
                                 "kind": evaluate.condition_kind(spec, cfg["corruption"]), "spec": spec})
    q = np.array([r["q"] for r in rows]); qs = np.array([r["q_star"] for r in rows]); E = np.array([r["E"] for r in rows])
    fused = np.array([r["fused_score_l1"] for r in rows]); fg = np.array([r["fg"] for r in rows])

    def block(mask):
        if mask.sum() < 5:
            return {"n": int(mask.sum())}
        gap = np.abs(qs[mask, 0] - qs[mask, 1]) >= 0.05
        rank = ((q[mask, 0] > q[mask, 1]) == (qs[mask, 0] > qs[mask, 1]))[gap]
        out = {"n": int(mask.sum()), "mae_camera": float(np.abs(q[mask, 0] - qs[mask, 0]).mean()), "mae_lidar": float(np.abs(q[mask, 1] - qs[mask, 1]).mean()),
               "spearman_camera_vs_negE": float(spearmanr(q[mask, 0], -E[mask, 0]).correlation),
               "spearman_lidar_vs_negE": float(spearmanr(q[mask, 1], -E[mask, 1]).correlation),
               "spearman_qmin_vs_neg_fused_score_l1": float(spearmanr(q[mask].min(1), -fused[mask]).correlation),
               "ranking_accuracy": float(rank.mean()) if len(rank) else None, "ranking_pairs": int(len(rank)),
               "q_mean": q[mask].mean(0).tolist(), "q_std": q[mask].std(0).tolist(), "q_star_mean": qs[mask].mean(0).tolist()}
        for i, name in enumerate(("camera", "lidar")):
            qi = q[mask, i]
            edges = np.quantile(qi, [0, .2, .4, .6, .8, 1.0])
            tbl = []
            for a, b in zip(edges, edges[1:]):
                sel = (qi >= a) & (qi <= b)
                tbl.append({"q_range": [float(a), float(b)], "n": int(sel.sum()), "mean_E": float(E[mask][sel, i].mean()) if sel.any() else None,
                            "mean_fused_score_l1": float(fused[mask][sel].mean()) if sel.any() else None})
            out[f"quantiles_{name}"] = tbl
        return out

    report = {"method": args.method, "seed": args.seed, "target": args.target, "n_rows": len(rows),
              "overall": block(np.ones(len(rows), bool)), "foreground_present": block(fg), "foreground_absent": block(~fg)}
    by_severity = {}
    for r in rows:
        for mod, v in r["spec"].items():
            by_severity.setdefault(f"{mod}_{v[0]}", {}).setdefault(v[1], []).append(r["q"])
    report["q_by_family_severity"] = {k: {str(s): {"n": len(v), "q_mean": np.mean(v, 0).tolist(), "q_std": np.std(v, 0).tolist()} for s, v in sorted(d.items())} for k, d in by_severity.items()}
    conds = np.array([r["condition"] for r in rows])
    report["q_clean"] = {"n": int((conds == "clean").sum()), "q_mean": q[conds == "clean"].mean(0).tolist()}
    out = art / f"reliability_{tag}{args.method}_seed{args.seed}.json"
    out.write_text(json.dumps(report, indent=2))
    import csv
    csv_dir = art.parent / "csv"
    csv_dir.mkdir(exist_ok=True)
    with open(csv_dir / f"reliability_rows_{tag}{args.method}_seed{args.seed}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "target", "condition", "kind", "camera_family", "camera_severity", "lidar_family", "lidar_severity", "sample_id", "episode_id",
                    "fg_present", "q_camera", "q_lidar", "E_camera", "E_lidar", "q_star_camera", "q_star_lidar", "fused_score_l1"])
        for r in rows:
            cam, lid = r["spec"].get("camera"), r["spec"].get("lidar")
            w.writerow([args.method, args.seed, args.target, r["condition"], r["kind"], cam[0] if cam else "", cam[1] if cam else 0, lid[0] if lid else "", lid[1] if lid else 0,
                        r["sample_id"], r["sample_id"].split("/")[0], int(r["fg"]), r["q"][0], r["q"][1], r["E"][0], r["E"][1], r["q_star"][0], r["q_star"][1], r["fused_score_l1"]])
    print(json.dumps({k: report[k] for k in ("overall", "foreground_present", "q_clean")}, indent=1))


if __name__ == "__main__":
    main()
