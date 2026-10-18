"""Aggregate artifacts/metrics/* into tables (markdown + json) and per-episode paired bootstrap
of proposed-minus-B2 deltas. Aggregates are recomputed from the stored per-condition metrics."""
import argparse
import json
from pathlib import Path

import numpy as np

from rrf import metrics
from rrf.config import load_config


def load_runs(metrics_root, suite, split):
    runs = {}
    for d in sorted(Path(metrics_root).glob(f"*_{suite}_{split}")):
        if (d / "metrics.json").exists():
            method = d.name.split("_seed")[0]
            seed = int(d.name.split("_seed")[1].split("_")[0])
            runs[(method, seed)] = json.loads((d / "metrics.json").read_text())
    return runs


def fmt(v):
    return "null" if v is None else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--suite", default="full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--baseline", default="B2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    art = Path(cfg["paths"]["artifacts_root"])
    runs = load_runs(art / "metrics", args.suite, args.split)
    if not runs:
        print("no runs found"); return
    conds = list(next(iter(runs.values())).keys())
    from rrf import evaluate
    kinds = {c: evaluate.condition_kind(r["spec"], cfg["corruption"]) for c, r in next(iter(runs.values())).items()}
    lines = [f"# Results: suite={args.suite} split={args.split}", "",
             "Metric: grid_fg_ap (InterFuser traffic-grid foreground AP, all valid cells pooled). Not 3D mAP.", ""]
    # table 1: per-method macro numbers
    lines += ["| method | seed | clean AP | seen macro-AP | unseen-severity macro-AP | unseen-family macro-AP | mixed macro-AP |", "|---|---|---|---|---|---|---|"]
    summary = {}
    for (m, s), r in sorted(runs.items()):
        def macro(kind):
            v = [r[c]["grid_fg_ap"] for c in conds if kinds[c] == kind and r[c]["grid_fg_ap"] is not None]
            return float(np.mean(v)) if v else None
        row = {"clean": r["clean"]["grid_fg_ap"], "seen": macro("seen"), "unseen_severity": macro("unseen_severity"),
               "unseen_family": macro("unseen_family"), "mixed": macro("mixed")}
        summary[f"{m}_seed{s}"] = row
        lines.append(f"| {m} | {s} | {fmt(row['clean'])} | {fmt(row['seen'])} | {fmt(row['unseen_severity'])} | {fmt(row['unseen_family'])} | {fmt(row['mixed'])} |")
    # table 1b: across-seed mean +- std per method, and per-seed proposed-minus-baseline deltas
    by_method = {}
    for (m, s), row in ((k, summary[f"{k[0]}_seed{k[1]}"]) for k in runs):
        by_method.setdefault(m, {})[s] = row
    cols = ["clean", "seen", "unseen_severity", "unseen_family", "mixed"]
    lines += ["", "## Across seeds (mean ± std over available seeds)", "", "| method | seeds | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for m, rows in sorted(by_method.items()):
        vals = []
        for c in cols:
            v = [r[c] for r in rows.values() if r[c] is not None]
            vals.append(f"{np.mean(v):.4f} ± {np.std(v):.4f}" if len(v) > 1 else fmt(v[0]) if v else "null")
        lines.append(f"| {m} | {sorted(rows)} | " + " | ".join(vals) + " |")
    seed_deltas = {}
    for m, rows in sorted(by_method.items()):
        if m == args.baseline or args.baseline not in by_method:
            continue
        common = sorted(set(rows) & set(by_method[args.baseline]))
        if not common:
            continue
        lines += ["", f"### {m} − {args.baseline} per seed", "", "| seed | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for s in common:
            d = {c: (rows[s][c] - by_method[args.baseline][s][c]) if rows[s][c] is not None and by_method[args.baseline][s][c] is not None else None for c in cols}
            seed_deltas.setdefault(m, {})[s] = d
            lines.append(f"| {s} | " + " | ".join("null" if d[c] is None else f"{d[c]:+.4f}" for c in cols) + " |")
        if len(common) > 1:
            lines.append("| mean ± std | " + " | ".join(f"{np.mean([seed_deltas[m][s][c] for s in common]):+.4f} ± {np.std([seed_deltas[m][s][c] for s in common]):.4f}" for c in cols) + " |")
    # table 2: per-condition AP per method
    methods = sorted({m for m, _ in runs})
    lines += ["", "## Per-condition AP", "", "| condition | kind | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    per_cond = {}
    for c in conds:
        vals = []
        for m in methods:
            v = [r[c]["grid_fg_ap"] for (mm, _), r in runs.items() if mm == m]
            v = [x for x in v if x is not None]
            per_cond.setdefault(c, {})[m] = float(np.mean(v)) if v else None
            vals.append(fmt(per_cond[c][m]) + (f" (n={len(v)})" if len(v) > 1 else ""))
        lines.append(f"| {c} | {kinds[c]} | " + " | ".join(vals) + " |")
    # table 3: paired episode bootstrap of P - baseline (same seed)
    boot = {}
    for (m, s), r in runs.items():
        if m == args.baseline or (args.baseline, s) not in runs:
            continue
        base = runs[(args.baseline, s)]
        rows = {}
        for c in conds:
            eps = sorted(set(base[c]["per_episode_ap"]) & set(r[c]["per_episode_ap"]))
            eps = [e for e in eps if base[c]["per_episode_ap"][e] is not None and r[c]["per_episode_ap"][e] is not None]
            if len(eps) >= 2:
                rows[c] = metrics.paired_bootstrap([base[c]["per_episode_ap"][e] for e in eps], [r[c]["per_episode_ap"][e] for e in eps],
                                                   repeats=cfg["metrics"]["bootstrap_repeats"], seed=s)
        boot[f"{m}_seed{s}_minus_{args.baseline}"] = rows
    if boot:
        lines += ["", f"## Episode-level paired bootstrap: method − {args.baseline} (per-episode AP, 95% CI)", ""]
        for k, rows in boot.items():
            lines += [f"### {k}", "", "| condition | delta | CI95 low | CI95 high | episodes |", "|---|---|---|---|---|"]
            for c, b in rows.items():
                lines.append(f"| {c} | {b['delta']:+.4f} | {b['ci95'][0]:+.4f} | {b['ci95'][1]:+.4f} | {b['groups']} |")
    # native losses + latency
    lines += ["", "## Native traffic losses (clean) and latency", "", "| method | seed | score L1 | geometry L1 | speed L1 | latency ms (median / p95) |", "|---|---|---|---|---|---|"]
    for (m, s), r in sorted(runs.items()):
        nl, lt = r["clean"]["native_loss"], r["clean"]["latency_ms_per_sample"]
        lines.append(f"| {m} | {s} | {nl['score']:.4f} | {nl['geometry']:.4f} | {nl['speed']:.4f} | {lt['median']:.1f} / {lt['p95']:.1f} |")
    out_dir = art / "metrics"
    (out_dir / f"summary_{args.suite}_{args.split}.json").write_text(json.dumps({"macro": summary, "per_condition": per_cond, "bootstrap": boot, "seed_deltas": seed_deltas}, indent=2))
    (out_dir / f"summary_{args.suite}_{args.split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
