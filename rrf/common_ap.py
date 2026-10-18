"""Vehicle-centre AP shared by InterFuser and TransFuser saved predictions.

InterFuser's 20x20 grid has 1 m cells and covers lateral [-10, 10] m and forward
[-2, 18] m about the vehicle centre. TransFuser covers lateral [-16, 16] m and
forward [0, 32] m about its LiDAR; the LiDAR is 1.3 m ahead of the vehicle centre,
so its vehicle-frame forward extent is [1.3, 33.3] m. Their intersection is
lateral [-10, 10] m and vehicle-frame forward [1.3, 18] m.

Both models are scored against vehicle-only actor centres. TransFuser's head is
vehicle-only. InterFuser's seven output channels contain no class prediction, so
its detections remain class-agnostic; walker GT is excluded and counted in the
output to make this limitation visible.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from rrf import objects
from rrf.config import load_config

COMMON_REGION = {"right_min_m": -10.0, "right_max_m": 10.0,
                 "forward_min_m": 1.3, "forward_max_m": 18.0}
REGIONS = {
    "interfuser_vehicle_frame": {"right_min_m": -10.0, "right_max_m": 10.0,
                                  "forward_min_m": -2.0, "forward_max_m": 18.0,
                                  "cell_size_m": 1.0},
    "transfuser_lidar_frame": {"right_min_m": -16.0, "right_max_m": 16.0,
                                "forward_min_m": 0.0, "forward_max_m": 32.0,
                                "pixels_per_m": 8.0},
    "transfuser_vehicle_frame": {"right_min_m": -16.0, "right_max_m": 16.0,
                                  "forward_min_m": 1.3, "forward_max_m": 33.3},
    "common_vehicle_frame": COMMON_REGION,
}


def in_common_region(right, forward):
    return (COMMON_REGION["right_min_m"] <= right <= COMMON_REGION["right_max_m"] and
            COMMON_REGION["forward_min_m"] <= forward <= COMMON_REGION["forward_max_m"])


def transfuser_detections(boxes, count=None, pixels_per_m=8.0, lidar_offset_m=1.3,
                          score_threshold=0.05):
    """Convert decoded TransFuser boxes to ``(score, right, forward)`` vehicle coordinates."""
    boxes = np.asarray(boxes)
    boxes = boxes[:len(boxes) if count is None else int(count)]
    detections = []
    for box in boxes:
        if float(box[7]) < score_threshold:
            continue
        right = float(box[0]) / pixels_per_m - 16.0
        forward = 32.0 - float(box[1]) / pixels_per_m + lidar_offset_m
        if in_common_region(right, forward):
            detections.append((float(box[7]), right, forward))
    return detections


def interfuser_detections(traffic, score_threshold=0.05, nms_radius=2.0):
    return objects.decode_centres(traffic, score_threshold, nms_radius, in_common_region)


def actor_centres(dataset_root, sample_id):
    """Return vehicle and walker centres in the ego vehicle-centre frame."""
    episode, frame = sample_id.split("/", 1)
    directory = Path(dataset_root) / episode
    measurement = json.loads((directory / "measurements" / f"{frame}.json").read_text())
    actors = json.loads((directory / "actors_data" / f"{frame}.json").read_text())
    yaw = math.radians(float(measurement.get(
        "yaw_deg", math.degrees(float(measurement["theta"]) - math.pi / 2))))
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    vehicles, walkers = [], []
    for actor in actors.values():
        actor_type = int(actor.get("tpe", -1))
        if actor_type not in (0, 1):
            continue
        dx = float(actor["loc"][0]) - float(measurement["x"])
        dy = float(actor["loc"][1]) - float(measurement["y"])
        forward = cos_yaw * dx + sin_yaw * dy
        right = -sin_yaw * dx + cos_yaw * dy
        if actor_type == 0 and math.hypot(forward, right) < 0.1:
            continue
        if in_common_region(right, forward):
            (vehicles if actor_type == 0 else walkers).append((right, forward))
    return vehicles, walkers


def common_center_ap(predictions, targets, radius):
    """Score already-decoded centres with score-ordered, one-to-one distance matching."""
    scored, n_gt = [], 0
    for detections, truth in zip(predictions, targets):
        matched, count = objects.match(detections, truth, radius)
        scored.extend(matched)
        n_gt += count
    if n_gt == 0:
        return None, {"n_gt": 0, "n_det": len(scored)}
    scored.sort(key=lambda item: -item[0])
    truth = np.asarray([item[1] for item in scored], dtype=bool)
    if not len(truth):
        return 0.0, {"n_gt": n_gt, "n_det": 0, "recall_at_all_dets": 0.0}
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    return float(precision[truth].sum() / n_gt), {
        "n_gt": n_gt, "n_det": len(scored),
        "recall_at_all_dets": float(truth.sum() / n_gt),
    }


def evaluate_predictions(pred_dir, model, dataset_root):
    """Evaluate all ``pred_*.npz`` files in one evaluator output directory."""
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.glob("pred_*.npz"))
    if not files:
        raise FileNotFoundError(f"no pred_*.npz files in {pred_dir}")
    truth_cache = {}
    results, reference_ids = {}, None
    for path in files:
        with np.load(path) as saved:
            sample_ids = [str(value) for value in saved["sample_ids"]]
            if reference_ids is None:
                reference_ids = sample_ids
            elif sample_ids != reference_ids:
                raise ValueError("prediction files do not contain identical sample_ids")
            for sample_id in sample_ids:
                if sample_id not in truth_cache:
                    truth_cache[sample_id] = actor_centres(dataset_root, sample_id)
            targets = [truth_cache[sample_id][0] for sample_id in sample_ids]
            if model == "interfuser":
                predictions = [interfuser_detections(value) for value in saved["traffic"]]
            else:
                counts = saved["box_counts"] if "box_counts" in saved else [len(value) for value in saved["boxes"]]
                predictions = [transfuser_detections(value, count)
                               for value, count in zip(saved["boxes"], counts)]
        condition = path.stem[len("pred_"):]
        ap1, info1 = common_center_ap(predictions, targets, 1.0)
        ap2, info2 = common_center_ap(predictions, targets, 2.0)
        per_episode = {}
        for episode in sorted({sample_id.split("/", 1)[0] for sample_id in sample_ids}):
            indices = [i for i, sample_id in enumerate(sample_ids)
                       if sample_id.split("/", 1)[0] == episode]
            per_episode[episode] = {
                "ap_1m": common_center_ap([predictions[i] for i in indices],
                                           [targets[i] for i in indices], 1.0)[0],
                "ap_2m": common_center_ap([predictions[i] for i in indices],
                                           [targets[i] for i in indices], 2.0)[0],
            }
        results[condition] = {"vehicle_center_ap_1m": ap1, "vehicle_center_ap_2m": ap2,
                              "vehicle_center_1m": info1, "vehicle_center_2m": info2,
                              "per_episode": per_episode}
    vehicle_count = sum(len(truth_cache[sample_id][0]) for sample_id in reference_ids)
    walker_count = sum(len(truth_cache[sample_id][1]) for sample_id in reference_ids)
    output = {
        "model": model, "feasible": True, "regions": REGIONS,
        "n_samples": len(reference_ids), "vehicle_gt_instances_in_region": vehicle_count,
        "walker_gt_instances_in_region": walker_count,
        "prediction_class_limitation": ("InterFuser detections are class-agnostic although GT is vehicle-only; "
                                        f"{walker_count} walker GT instances occupy the scored region."
                                        if model == "interfuser" else None),
        "conditions": results,
    }
    (pred_dir / "common_ap.json").write_text(json.dumps(output, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/journal.yaml")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--model", required=True, choices=("interfuser", "transfuser"))
    args = parser.parse_args()
    config = load_config(args.config)
    output = evaluate_predictions(args.pred_dir, args.model, config["paths"]["dataset_root"])
    print(json.dumps({key: output[key] for key in (
        "model", "feasible", "regions", "n_samples", "vehicle_gt_instances_in_region",
        "walker_gt_instances_in_region", "prediction_class_limitation")}, indent=2))


if __name__ == "__main__":
    main()
