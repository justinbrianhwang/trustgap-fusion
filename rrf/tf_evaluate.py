"""Evaluate released TransFuser variants on the shared corruption suites."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rrf import data, metrics
from rrf.config import load_config
from rrf.diag_common import reduced_suite
from rrf.evaluate import full_suite
from rrf.tf_corruptions import corrupt_tf_sample
from rrf.tf_probes import render_tf_heatmap
from rrf.transfuser_io import (VARIANTS, load_raw_tf, load_tf_model, predict_tf,
                               preprocess_tf, tf_targets)

METHODS = ("B0", "B2", "BQ", "B3", "B4", "P", "PG", "PR", "PS", "PGS", "PT", "PGT")


def center_distance_ap(predictions, targets, pixels_per_meter, radius=1.0,
                       score_threshold=0.05):
    """Class-agnostic centre AP with score-ordered one-to-one distance matching."""
    scored, total_gt = [], 0
    for prediction, target in zip(predictions, targets):
        prediction, target = np.asarray(prediction), np.asarray(target)
        detections = prediction[prediction[:, 7] >= score_threshold]
        detections = detections[np.argsort(-detections[:, 7], kind="stable")]
        used = np.zeros(len(target), dtype=bool)
        total_gt += len(target)
        for row in detections:
            best = -1
            if len(target):
                distances = np.linalg.norm(target[:, :2] - row[None, :2], axis=1) / pixels_per_meter
                distances[used] = np.inf
                candidate = int(np.argmin(distances))
                if distances[candidate] <= radius:
                    best = candidate
                    used[best] = True
            scored.append((float(row[7]), best >= 0))
    if total_gt == 0:
        return None, {"n_gt": 0, "n_det": len(scored)}
    scored.sort(key=lambda item: -item[0])
    truth = np.asarray([item[1] for item in scored], dtype=bool)
    if not len(truth):
        return 0.0, {"n_gt": total_gt, "n_det": 0, "recall_at_all_dets": 0.0}
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    ap = float(precision[truth].sum() / total_gt)
    return ap, {"n_gt": total_gt, "n_det": len(scored),
                "recall_at_all_dets": float(truth.sum() / total_gt)}


def _sample_ids(root, split, artifacts_root):
    available = {episode["episode_id"] for episode in data.episodes(root)}
    manifest = Path(artifacts_root) / "manifests" / "split.json"
    if manifest.exists():
        spec = json.loads(manifest.read_text())
        selected = available & set(data.split_episodes(spec, split))
        if selected:
            return [sample for episode in sorted(selected) for sample in data.sample_ids(root, episode)]
    return [sample for episode in sorted(available) for sample in data.sample_ids(root, episode)]


def _stack(items):
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


@torch.no_grad()
def run_suite(model, config, dataset_root, sample_ids, suite, seed, output_dir,
              frame_stride=1, batch_size=4, device="cuda"):
    sample_ids = sample_ids[::frame_stride]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_path = output_dir / "frames.jsonl"
    results = {}
    with frames_path.open("w") as frame_file:
        for condition, spec in suite.items():
            started = time.time()
            predicted_boxes, target_boxes, heatmaps, target_heatmaps = [], [], [], []
            audits, latencies = [], []
            for offset in range(0, len(sample_ids), batch_size):
                ids = sample_ids[offset:offset + batch_size]
                prepared, batch_audits = [], []
                for sample_id in ids:
                    corrupt, audit = corrupt_tf_sample(load_raw_tf(dataset_root, sample_id), spec, seed)
                    prepared.append(preprocess_tf(corrupt, config))
                    batch_audits.append(audit)
                    audits.append(audit)
                batch = _stack(prepared)
                if device == "cuda":
                    torch.cuda.synchronize()
                tick = time.time()
                output = model(batch) if hasattr(model, "gate_enabled_by_default") else predict_tf(model, batch)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.extend([(time.time() - tick) / len(ids)] * len(ids))
                boxes = output["boxes"].cpu().numpy()
                maps = output["center_heatmap"][:, 0].cpu().numpy()
                for index, (sample_id, audit) in enumerate(zip(ids, batch_audits)):
                    truth = tf_targets(dataset_root, sample_id, config)
                    target = render_tf_heatmap(truth, config).cpu().numpy()
                    predicted_boxes.append(boxes[index])
                    target_boxes.append(truth)
                    heatmaps.append(maps[index])
                    target_heatmaps.append(target)
                    q = output.get("q")
                    frame_file.write(json.dumps({
                        "condition": condition, "sample_id": sample_id,
                        "episode_id": sample_id.split("/")[0],
                        "q": q[index].detach().cpu().tolist() if q is not None else None,
                        "n_gt_objects": len(truth), "heatmap_peak_cells": int((target >= 0.99).sum()),
                        "corruption": {key: audit[key] and {k: v for k, v in audit[key].items() if k != "views"}
                                       for key in ("camera", "lidar")},
                    }) + "\n")
            target_stack = np.asarray(target_heatmaps)
            keep = (target_stack >= 0.99) | (target_stack < 0.01)  # tails ignored (CenterNet convention)
            labels = (target_stack >= 0.99)[keep]
            heat = np.asarray(heatmaps)[keep]
            labels_05 = np.asarray(target_heatmaps) >= 0.5
            ap1, info1 = center_distance_ap(predicted_boxes, target_boxes, config.pixels_per_meter, 1.0)
            ap2, info2 = center_distance_ap(predicted_boxes, target_boxes, config.pixels_per_meter, 2.0)
            results[condition] = {
                "spec": spec, "n_samples": len(sample_ids),
                "vehicle_center_ap_1m": ap1, "vehicle_center_ap_2m": ap2,
                "vehicle_center_1m": info1, "vehicle_center_2m": info2,
                "heatmap_peak_ap": metrics.grid_fg_ap(heat, labels),
                "heatmap_ap_fg05": metrics.grid_fg_ap(np.asarray(heatmaps), labels_05),  # sensitivity variant
                "n_heatmap_peak_cells": int(labels.sum()), "n_heatmap_cells_scored": int(labels.size),
                "latency_ms_per_sample": {"median": float(np.median(latencies) * 1000),
                                          "p95": float(np.percentile(latencies, 95) * 1000)},
                "audit_example": {key: audits[0][key] and {k: v for k, v in audits[0][key].items() if k != "views"}
                                  for key in ("camera", "lidar")},
                "elapsed_s": round(time.time() - started, 1),
            }
            print(f"[{condition}] AP@1m={ap1} AP@2m={ap2} heatmap_peak_AP={results[condition]['heatmap_peak_ap']}", flush=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="transfuser")
    parser.add_argument("--method", choices=METHODS, default="B0")
    parser.add_argument("--suite", choices=("reduced", "full"), default="reduced")
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-file")
    parser.add_argument("--stages", default="0,1,2,3",
                        help="gated fusion stages; must match a non-default training checkpoint")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    started = time.time()
    project = load_config(args.config)
    suite = reduced_suite(project["corruption"]) if args.suite == "reduced" else full_suite(project["corruption"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.method == "B0":
        model, config = load_tf_model(args.variant, args.seed_file, device)
    else:
        from rrf.tf_train import load_trained_method, parse_stages
        project["paths"]["artifacts_root"] = str(Path(args.artifacts_root).resolve())
        model, config = load_trained_method(project, args.variant, args.method, args.seed,
                                            parse_stages(args.stages), device, args.seed_file)
    ids = _sample_ids(args.root, args.split, args.artifacts_root)
    if not ids:
        raise RuntimeError(f"no samples found under {args.root}")
    output = Path(args.artifacts_root) / "tf_metrics" / (
        f"{args.variant}_{args.method}_seed{args.seed}_{args.suite}_{args.split}")
    results = run_suite(model, config, args.root, ids, suite, args.seed, output,
                        args.frame_stride, args.batch_size, device)
    summary = {"variant": args.variant, "method": args.method, "suite": args.suite,
               "split": args.split, "seed": args.seed, "frame_stride": args.frame_stride,
               "n_samples": len(ids[::args.frame_stride]), "device": device,
               "wall_seconds": round(time.time() - started, 1), "conditions": results}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=2))


if __name__ == "__main__":
    main()
