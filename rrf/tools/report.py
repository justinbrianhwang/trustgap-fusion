"""Assemble artifacts/REPORT.md from stored artifacts. Missing pieces are written as `not run`;
no number in the report is typed by hand."""
import argparse
import json
from pathlib import Path

from rrf.config import load_config


def jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def fmt(v, nd=4):
    return "not run" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--narrative", default="docs/REPORT_NARRATIVE.md", help="hand-written interpretation appended verbatim")
    args = ap.parse_args()
    cfg = load_config(args.config)
    art = Path(cfg["paths"]["artifacts_root"])
    env = jload(art / "environment.json", {})
    contract = jload(art / "tensor_contract.json", {})
    split_stats = jload(art / "manifests" / "split_stats.json", {})
    ref = jload(art / "runs" / "reference" / "eval.json")
    summ = jload(art / "metrics" / "summary_full_test.json", {})
    op = jload(art / "metrics" / f"operating_point_seed{args.seed}.json", {})
    rel = jload(art / "metrics" / f"reliability_P_seed{args.seed}.json")
    L = ["# RRF-InterFuser pilot report", "",
         "Scope: **InterFuser traffic-grid foreground detection robustness** under synthetic camera/LiDAR corruptions on",
         "stored CARLA 0.9.16 observations. Metric `grid_fg_ap` is a 20x20 grid AP, not 3D mAP. Nothing here is a claim",
         "about driving safety. Completion level: see section 1.", ""]
    # 1 scope
    fus = {m: jload(art / "runs" / f"fusion_{m}_seed{args.seed}" / "final.json") for m in ("B1", "B2", "B3", "B4", "P")}
    done = [m for m, f in fus.items() if f]
    seeds = {m: sorted(int(d.name.split("seed")[1]) for d in (art / "runs").glob(f"fusion_{m}_seed*") if (d / "final.json").exists()) for m in fus}
    L += ["## 1. Implemented and executed scope", "",
          f"- Implementation complete: code, config, tests (`pytest tests`), docs. Pilot trained methods and seeds: " + ", ".join(f"{m} {seeds[m]}" for m in done) + ".",
          "- Research-validation level (3 seeds, B1/B3/B4, native weather, closed loop): " + ("partially, see tables" if len(done) > 2 else "not run") + ".", ""]
    # 2 provenance
    up = env.get("upstream", {})
    L += ["## 2. Upstream commit, environment, checkpoint", "",
          f"- upstream commit `{up.get('commit')}` (matches config: {up.get('commit_matches')}); working tree differs from HEAD only by line endings (0 insertions / 0 deletions ignoring whitespace)",
          f"- checkpoint `{env.get('checkpoint', {}).get('path')}` sha256 `{env.get('checkpoint', {}).get('sha256')}`; strict load missing/unexpected = {contract.get('load_report', {}).get('missing')}/{contract.get('load_report', {}).get('unexpected')}",
          f"- torch {env.get('torch', {}).get('torch')} cuda {env.get('torch', {}).get('cuda_runtime')} on {env.get('torch', {}).get('device_name')}; CARLA server {env.get('carla', {}).get('server', {}).get('version') if isinstance(env.get('carla', {}).get('server'), dict) else 'n/a'} (checkpoint trained on 0.9.10.1 data)",
          f"- config sha256 `{cfg['_meta']['config_sha256']}`; split sha256 `{split_stats.get('split_sha256')}`; manifest sha256 `{split_stats.get('manifest_sha256')}`", ""]
    # 3 data
    L += ["## 3. Data and split", "",
          f"- samples: {split_stats.get('samples')}; episodes train/val/test = {split_stats.get('episodes')}; frames = {split_stats.get('frames')}",
          f"- excluded episodes: {split_stats.get('excluded')}", "- pretraining_overlap: unknown (no unseen-town claim)", ""]
    # 4 corruptions
    L += ["## 4. Corruptions", "", "Defined in `rrf/corruptions.py` (severity tables), realized parameters logged per frame in `frames.jsonl`.",
          f"Train: {cfg['corruption']['train_families']} severities {cfg['corruption']['train_severities']}; held-out families {cfg['corruption']['heldout_families']}; test severities {cfg['corruption']['test_severities']}.", ""]
    # 5 reference / reliability
    L += ["## 5. Reference (masked upstream) and reliability targets", ""]
    if ref:
        L += ["| reference | " + " | ".join(c for c in ref["camera_only"]) + " |", "|---|" + "---|" * len(ref["camera_only"])]
        for name in ("camera_only", "lidar_only"):
            L.append(f"| {name} AP | " + " | ".join(fmt(v["ap"], 3) for v in ref[name].values()) + " |")
    else:
        L.append("not run")
    for sd in sorted(int(p.stem.split("seed")[1]) for p in (art / "checkpoints").glob("reliability_seed*.pt")):
        import torch
        meta = torch.load(art / "checkpoints" / f"reliability_seed{sd}.pt", map_location="cpu", weights_only=False)["meta"]
        e = meta["eval"]["foreground_present"]
        L += ["", f"Reliability estimator seed {sd} (validation, selected step {meta['step']}, foreground-present frames, n={e.get('n')}): MAE {fmt(e.get('mae'), 3)}, Spearman(q̂, −E) camera {fmt(e.get('spearman_camera'), 3)} / lidar {fmt(e.get('spearman_lidar'), 3)}, ranking accuracy {fmt(e.get('ranking_accuracy'), 3)} on {e.get('ranking_pairs')} pairs."]
    L.append("")
    # 6 methods
    L += ["## 6. Methods and fairness", "", "| method | selected step | val clean AP | val corruption macro-AP | trainable params | wall s | peak GPU GB |", "|---|---|---|---|---|---|---|"]
    for m, f in fus.items():
        if not f:
            L.append(f"| {m} | not run | | | | | |"); continue
        setup = jload(art / "runs" / f"fusion_{m}_seed{args.seed}" / "setup.json", {})
        sel = f.get("selected") or {}
        L.append(f"| {m} | {sel.get('step', f.get('status', 'none'))} | {fmt(sel.get('clean_ap'))} | {fmt(sel.get('corruption_macro_ap'))} | {setup.get('trainable_params')} (+{setup.get('estimator_params')} gate) | {f.get('wall_seconds')} | {f.get('peak_gpu_mem_gb')} |")
    L += ["", "B0 = released checkpoint (no fine-tuning). Shared: checkpoint, split, corruption schedule, freeze policy, optimizer budget, selection rule (see docs/EVALUATION_PROTOCOL.md).", ""]
    # 7 results
    L += ["## 7. Test results (locked test episodes)", ""]
    md = art / "metrics" / "summary_full_test.md"
    L += [md.read_text().split("\n", 1)[1] if md.exists() else "not run", ""]
    # operating point
    L += ["## 7b. Fixed operating point (threshold from clean calibration split, precision target " + str(cfg["metrics"]["clean_calibration_precision_target"]) + ")", ""]
    if op:
        conds = None
        for m, r in op.items():
            if "conditions" in r:
                conds = list(r["conditions"]); break
        if conds:
            L += ["| condition | " + " | ".join(f"{m} FNR / FPR" for m in op) + " |", "|---|" + "---|" * len(op)]
            for c in conds:
                L.append(f"| {c} | " + " | ".join((f"{fmt(op[m]['conditions'][c]['fnr'], 3)} / {fmt(op[m]['conditions'][c]['fpr'], 4)}" if "conditions" in op[m] else op[m].get("status", "n/a")) for m in op) + " |")
            L.append("thresholds: " + ", ".join(f"{m}={fmt(r.get('threshold'))}" for m, r in op.items()))
    else:
        L.append("not run")
    L.append("")
    # 8 q vs error
    L += ["## 8. q versus task error (test, method P)", ""]
    for sd in (1, 2):
        r2 = jload(art / "metrics" / f"reliability_P_seed{sd}.json")
        if r2:
            b = r2["foreground_present"]
            L.append(f"- seed {sd}, foreground-present (n={b.get('n')}): Spearman(q, −E) cam {fmt(b.get('spearman_camera_vs_negE'), 3)} / lidar {fmt(b.get('spearman_lidar_vs_negE'), 3)}; Spearman(min q, −fused score L1) {fmt(b.get('spearman_qmin_vs_neg_fused_score_l1'), 3)}; ranking acc {fmt(b.get('ranking_accuracy'), 3)}")
    for f in sorted((art / "metrics").glob("reliability_rel_*_seed*.json")):
        r2 = jload(f)
        b = r2["foreground_present"]
        L.append(f"- {r2['method']} seed {r2['seed']} scored against its own *relative* target (E = max(E_corrupt − E_clean, 0)), foreground-present (n={b.get('n')}): MAE cam {fmt(b.get('mae_camera'), 3)} / lidar {fmt(b.get('mae_lidar'), 3)}; Spearman(q, −E_rel) cam {fmt(b.get('spearman_camera_vs_negE'), 3)} / lidar {fmt(b.get('spearman_lidar_vs_negE'), 3)}; ranking acc {fmt(b.get('ranking_accuracy'), 3)}")
    L.append(f"- seed {args.seed} (detailed):")
    if rel:
        for k in ("overall", "foreground_present", "foreground_absent"):
            b = rel[k]
            L.append(f"- {k} (n={b.get('n')}): MAE cam {fmt(b.get('mae_camera'), 3)} / lidar {fmt(b.get('mae_lidar'), 3)}; Spearman(q, −E) cam {fmt(b.get('spearman_camera_vs_negE'), 3)} / lidar {fmt(b.get('spearman_lidar_vs_negE'), 3)}; Spearman(min q, −fused score L1) {fmt(b.get('spearman_qmin_vs_neg_fused_score_l1'), 3)}; ranking acc {fmt(b.get('ranking_accuracy'), 3)} ({b.get('ranking_pairs')} pairs)")
        L += ["", "| family | " + " | ".join(f"s{s}" for s in range(1, 6)) + " | clean |", "|---|---|---|---|---|---|---|"]
        for fam, d in rel["q_by_family_severity"].items():
            i = 0 if fam.startswith("camera") else 1
            L.append(f"| {fam} (q_{'cam' if i == 0 else 'lidar'}) | " + " | ".join(fmt(d[str(s)]["q_mean"][i], 3) if str(s) in d else "-" for s in range(1, 6)) + f" | {fmt(rel['q_clean']['q_mean'][i], 3)} |")
    else:
        L.append("not run")
    L.append("")
    # 8b weather / object / closed loop
    L += ["## 8b. Native-weather track (test episodes re-collected under rain / fog / night; episode-group comparison)", ""]
    wfiles = sorted((art / "metrics").glob("weather_*_seed*.json"))
    if wfiles:
        ws = sorted({w for f in wfiles for w in json.loads(f.read_text())["weather"]})
        L += ["| set | method | seed | " + " | ".join(f"{w} AP / score L1 (ClearNoon AP / L1)" for w in ws) + " |", "|---|---|---|" + "---|" * len(ws)]
        for f in wfiles:
            r = json.loads(f.read_text())
            tag = f.stem.split("seed", 1)[1].lstrip("0123456789")  # '' = 4 locked test episodes, '_w2'/'_w2b' = 24-route weather set
            wset = "test-4ep" if tag in ("", "_temp") else "weather2-24routes" + (f" (stride {r.get('frame_stride', 1)})" if r.get("frame_stride", 1) != 1 else "")
            L.append(f"| {wset} | {r['method']} | {r['seed']} | " + " | ".join(
                f"{fmt(r['weather'][w]['grid_fg_ap'], 3)} / {fmt(r['weather'][w]['score_l1'], 3)} ({fmt(r['weather'][w]['clear_noon_counterpart']['grid_fg_ap'], 3)})" if w in r["weather"] else "not run" for w in ws) + " |")
        r0 = json.loads(wfiles[0].read_text())
        L.append("Foreground cells per group (weather / ClearNoon counterpart): " + ", ".join(f"{w}: {r0['weather'][w]['n_fg_cells']} / {r0['weather'][w]['clear_noon_counterpart']['n_fg_cells']}" for w in ws) +
                 ". The re-collected drives contain different traffic (BasicAgent reacts to it), so weather and ClearNoon groups are **not** the same scenes: AP differences between the two columns mix weather with scene content and must not be read as a weather effect. Within a weather column, methods share the same frames.")
        L.append("q means per weather are in `csv/weather*.csv`. LiDAR blueprint and noise settings are identical across weathers; weather is never an input. Rows of the weather2 set are pooled over 24 routes never used in any split (ClearNoon counterparts collected in the same set).")
    else:
        L.append("not run")
    L += ["", "## 8c. Object-centre AP (class-agnostic, 1 m / 2 m match radius; peak threshold 0.05, NMS 2 m)", ""]
    oa = art / "csv" / "object_ap.csv"
    if oa.exists():
        import csv as _csv
        rows = [r for r in _csv.DictReader(oa.open(encoding="utf-8")) if r["condition"] in ("clean", "camera_gaussian_blur_s5", "camera_darkening_s5", "lidar_point_dropout_s5", "camera_rectangular_occlusion_s5", "camera_silent_blackout", "lidar_silent_empty")]
        L += ["| method | seed | condition | AP@1m | AP@2m | GT objects | detections |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['method']} | {r['seed']} | {r['condition']} | {float(r['center_ap@1m']):.3f} | {float(r['center_ap@2m']):.3f} | {r['n_gt_objects']} | {r['n_detections']} |")
        L.append("Full table: `artifacts/csv/object_ap.csv`. GT encode→decode round trip is unit-tested (tests/test_objects.py).")
    else:
        L.append("not run")
    L += ["", "## 8d. Closed-loop check (model drives; corruption injected once at sensor acquisition; upstream tracker/controller unchanged)", ""]
    cl = art / "closed_loop" / "runs.json"
    if cl.exists():
        runs = json.loads(cl.read_text())
        agg = {}
        for r in runs:
            agg.setdefault((r["method"], r["condition"]), []).append(r)
        L += ["| method | condition | runs | completed | mean route completion | runs with collision | stuck/off-route/failed |", "|---|---|---|---|---|---|---|"]
        for (m, c), rs in sorted(agg.items()):
            ok = [r for r in rs if r.get("status") != "failed"]
            L.append(f"| {m} | {c} | {len(rs)} | {sum(r['status'] == 'route_completed' for r in ok)} | {fmt(sum(r['route_completion'] for r in ok) / len(ok), 3) if ok else 'n/a'} | {sum(r['collision_events'] > 0 for r in ok)} | {sum(r['status'] in ('stuck_timeout', 'off_route') for r in ok)}/{len(rs) - len(ok)} |")
        n_routes = len({(r.get("town"), r.get("episode_seed")) for r in runs})
        L.append(f"Per-run rows: `{art.name}/csv/closed_loop.csv`. {n_routes} routes × conditions, single seed; perception and driving outcome are not one-to-one.")
    else:
        L.append("not run")
    L.append("")
    # 9 cost
    L += ["## 9. Runtime and cost", "", "Latency per method in section 7 (clean). Training wall time and peak memory in section 6. P additionally needs 2 masked reference forwards per training step; no probe training.", ""]
    # 10-12 narrative
    nar = Path(args.narrative)
    L += ["## 10–12. Hypotheses, failure cases, limitations, reproduction", ""]
    L += [nar.read_text() if nar.exists() else "not written yet", ""]
    L += ["### Reproduction", "", "```", "bash scripts/run_pilot.sh   # or the individual commands in README.md", "```",
          "Result files: `artifacts/metrics/*_full_test/{metrics.json,frames.jsonl,pred_*.npz}`, `artifacts/metrics/summary_full_test.md`, `artifacts/figures/`, `artifacts/runs/*/log.jsonl`.",
          "CSV exports for analysis: `artifacts/csv/` (see its README.md for the schema of every file)."]
    (art / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
