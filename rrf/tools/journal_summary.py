"""Apply the pre-registered crossed seed x episode comparisons to saved test predictions."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rrf import common_ap, data, evaluate, metrics, stats, upstream
from rrf.config import load_config
from rrf.tf_probes import render_tf_heatmap
from rrf.transfuser_io import tf_targets


def tf_peak_ap(predictions, labels):
    labels = np.asarray(labels)
    keep = labels >= 0
    return metrics.grid_fg_ap(np.asarray(predictions)[keep], labels[keep] == 1)


def common_ap_1m(predictions, labels):
    return common_ap.common_center_ap(list(predictions), list(labels), 1.0)[0]


METRICS = {
    "grid_fg_ap": lambda predictions, labels: metrics.grid_fg_ap(predictions, labels),
    "heatmap_peak_ap": tf_peak_ap,
    "common_vehicle_center_ap_1m": common_ap_1m,
}


def _profile(method, standard_root, weak_root):
    if method.startswith("weak:"):
        return method.split(":", 1)[1], Path(weak_root)
    return method, Path(standard_root)


def _prediction_dirs(model, method, standard_root, weak_root):
    method, root = _profile(method, standard_root, weak_root)
    pattern = (f"{method}_seed*_full_test" if model == "interfuser"
               else f"transfuser_{method}_seed*_full_test")
    parent = root / ("metrics" if model == "interfuser" else "tf_metrics")
    found = {}
    for directory in sorted(parent.glob(pattern)):
        if not (directory / "pred_clean.npz").exists():
            continue
        seed = int(directory.name.rsplit("_seed", 1)[1].split("_", 1)[0])
        found[seed] = directory
    if not found:
        raise FileNotFoundError(f"no full-test predictions for {model} {method} under {parent}")
    return found


def _condition_specs(directory):
    report = json.loads((directory / "metrics.json").read_text())
    conditions = report.get("conditions", report)
    return {name: row["spec"] for name, row in conditions.items()}


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _tf_labels(dataset_root, sample_ids):
    config = SimpleNamespace(lidar_pos=[1.3, 0.0, 2.5], pixels_per_meter=8,
                             lidar_resolution_width=256, lidar_resolution_height=256)
    labels = []
    for sample_id in sample_ids:
        target = render_tf_heatmap(tf_targets(dataset_root, sample_id, config), config).numpy()
        labels.append(np.where(target >= 0.99, 1, np.where(target < 0.01, 0, -1)).astype(np.int8))
    return np.asarray(labels)


def load_crossed_inputs(model, method, metric, kinds, config, standard_root, weak_root,
                        label_cache=None):
    """Load one method as the seed/condition/sample structure consumed by ``rrf.stats``."""
    directories = _prediction_dirs(model, method, standard_root, weak_root)
    first = next(iter(directories.values()))
    specs = _condition_specs(first)
    selected = sorted(name for name, spec in specs.items()
                      if evaluate.condition_kind(spec, config["corruption"]) in kinds)
    if not selected:
        raise ValueError(f"no conditions of kinds {kinds} for {model} {method}")
    output = {}
    label_cache = {} if label_cache is None else label_cache
    for seed, directory in directories.items():
        output[seed] = {}
        for condition in selected:
            with np.load(directory / f"pred_{condition}.npz") as saved:
                sample_ids = [str(value) for value in saved["sample_ids"]]
                cache_key = (metric, tuple(sample_ids))
                if cache_key not in label_cache:
                    if metric == "grid_fg_ap":
                        label_cache[cache_key] = np.asarray([
                            upstream.foreground_label(data.load_gt(
                                config["paths"]["dataset_root"], sample_id, write_cache=False))
                            for sample_id in sample_ids])
                    elif metric == "heatmap_peak_ap":
                        label_cache[cache_key] = _tf_labels(config["paths"]["dataset_root"], sample_ids)
                    else:
                        label_cache[cache_key] = _object_array([
                            common_ap.actor_centres(config["paths"]["dataset_root"], sample_id)[0]
                            for sample_id in sample_ids])
                if metric == "grid_fg_ap":
                    predictions = saved["traffic"][:, :, 0]
                elif metric == "heatmap_peak_ap":
                    predictions = saved["center_heatmap"]
                elif model == "interfuser":
                    predictions = _object_array([common_ap.interfuser_detections(value)
                                                 for value in saved["traffic"]])
                else:
                    counts = saved["box_counts"]
                    predictions = _object_array([common_ap.transfuser_detections(value, count)
                                                 for value, count in zip(saved["boxes"], counts)])
            output[seed][condition] = {"sample_ids": sample_ids, "predictions": predictions,
                                       "labels": label_cache[cache_key]}
    return output


def run_comparisons(spec, config, standard_root, weak_root, repeats=2000, seed=0):
    label_cache, rows = {}, []
    for index, comparison in enumerate(spec):
        missing = {"model", "method", "baseline", "metric", "kinds", "delta", "family"} - set(comparison)
        if missing:
            raise ValueError(f"comparison {index} missing {sorted(missing)}")
        if comparison["metric"] not in METRICS:
            raise ValueError(f"unsupported metric {comparison['metric']!r}")
        for kind in comparison["kinds"]:
            inputs = []
            for method in (comparison["baseline"], comparison["method"]):
                inputs.append(load_crossed_inputs(
                    comparison["model"], method, comparison["metric"], [kind], config,
                    standard_root, weak_root, label_cache))
            result = stats.crossed_bootstrap_delta(
                inputs[0], inputs[1], METRICS[comparison["metric"]], n=repeats, seed=seed + index)
            rows.append({**comparison, "kind": kind, "result": result})
    for family in sorted({row["family"] for row in rows}):
        members = [row for row in rows if row["family"] == family]
        adjusted = stats.holm_on_cis([row["result"] for row in members])
        raw = [stats.holm_on_cis([row["result"]])[0] for row in members]
        for row, pvalue, pvalue_holm in zip(members, raw, adjusted):
            result = row["result"]
            result["pvalue_two_sided"] = pvalue
            result["pvalue_holm"] = pvalue_holm
            result["holm_reject_zero"] = bool(pvalue_holm < 0.05)
            result["verdict"] = stats.verdict(result["ci90"], result["ci95"], row["delta"])
            result["verdict_holm"] = ("inconclusive" if result["verdict"] in ("improves", "worse")
                                      and not result["holm_reject_zero"] else result["verdict"])
            del result["bootstrap_deltas"]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/journal.yaml")
    parser.add_argument("--comparisons", default="configs/journal_comparisons.json")
    parser.add_argument("--artifacts-root")
    parser.add_argument("--weak-artifacts-root", default="artifacts_journal_weak")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    standard_root = Path(args.artifacts_root or config["paths"]["artifacts_root"])
    weak_root = Path(args.weak_artifacts_root)
    spec = json.loads(Path(args.comparisons).read_text())
    repeats = args.repeats or config["metrics"]["bootstrap_repeats"]
    rows = run_comparisons(spec, config, standard_root, weak_root, repeats, args.seed)
    output = Path(args.output or standard_root / "metrics" / "journal_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"repeats": repeats, "comparisons": rows}, indent=2))
    lines = ["# Journal confirmatory comparisons", "",
             "| model | method - baseline | metric | kind | delta | CI95 | CI90 | Holm p | verdict |",
             "|---|---|---|---|---:|---|---|---:|---|"]
    for row in rows:
        result = row["result"]
        lines.append(f"| {row['model']} | {row['method']} - {row['baseline']} | {row['metric']} | "
                     f"{row['kind']} | {result['delta']:+.4f} | "
                     f"[{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}] | "
                     f"[{result['ci90'][0]:+.4f}, {result['ci90'][1]:+.4f}] | "
                     f"{result['pvalue_holm']:.4g} | {result['verdict_holm']} |")
    output.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
