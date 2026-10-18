"""Frozen-checkpoint diagnostics for TransFuser reliability and fusion.

The primary per-sample loss is ``tf_probes.heatmap_reference_error`` between
the predicted centre heatmap and the rendered CenterNet target (peak cells are
positive, Gaussian tails are ignored).  Matched vehicle centres at 1 m are
recorded as the second, object-level per-sample quality measure.  Every AP
summary reports both ``heatmap_peak_ap`` and ``vehicle_center_ap_1m``.

For the ``transfuser`` backbone, removal utility is measured at the sensor
input and by stage-level removal at all four fusion stages. The latter can
leave information already fused into the retained branch at an earlier stage.
Mean-feature replacement remains a stage-level sensitivity analysis.
Oracle gates use the InterFuser candidate grid, and trace uses the same fixed
q grid while recording camera/LiDAR attention mass for every GPT stage/layer.

The released ``late_fusion`` and ``geometric_fusion`` variants have no shared
camera/LiDAR attention, while ``latentTF`` uses positional BEV tokens rather
than LiDAR features.  The specified gated interventions, mean-feature utility,
oracle/shuffle analyses, and modality attention trace therefore do not apply;
requests for them are retained in ``summary.json`` as explicit skips.  B0 can
still be selected for every released variant without loading a training
checkpoint.  Trained methods are loaded from
``<artifacts_root>/tf_checkpoints/<variant>/fusion_<method>_seed<s>.pt``.
"""
import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from rrf import data, metrics, stats
from rrf.config import load_config
from rrf.diag_common import (GateTemporalAccumulator, GateXfitAccumulator,
                             cross_episode_permutation, merge_summary, partial_summary,
                             reduced_suite, write_csv)
from rrf.evaluate import condition_kind, full_suite
from rrf.gated import ReliabilityEstimator, StaticGate
from rrf.tf_corruptions import corrupt_tf_sample
from rrf.tf_evaluate import center_distance_ap
from rrf.tf_gated import GatedTransFuser
from rrf.tf_probes import (heatmap_reference_error, q_targets,
                           q_targets_relative, render_tf_heatmap)
from rrf.transfuser_io import (VARIANTS, load_raw_tf, load_tf_model,
                               preprocess_tf, tf_targets)


GRID = (0.05, 0.2, 0.5, 1.0)
PARTIAL_FACTORS = (0.5, 0.8, 1.25, 2.0)
TRACE_GRID = (0.05, 0.1, 0.2, 0.5, 1.0)
ANALYSES = ("utility", "partial", "oracle_rel", "oracle_gate", "oracle_gate_xfit",
            "oracle_gate_temporal", "shuffle", "trace")
MECHANISMS = {"PS": "token_scaling", "PGS": "token_scaling",
              "PT": "key_scaling", "PGT": "key_scaling"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_estimator(state, device):
    if not state:
        return None
    stat_dim = int(state["mu"].shape[1])
    weights = [value for key, value in state.items()
               if key.startswith("mlp.") and key.endswith(".weight")]
    estimator = ReliabilityEstimator(stat_dim, int(weights[0].shape[0]), len(weights) - 1)
    estimator.load_state_dict(state)
    return estimator.to(device)


def load_method(variant, method, seed, artifacts_root, device, project_config, seed_file=None):
    """Load a released B0 or a diagnostic-compatible trained TransFuser gate."""
    model, config = load_tf_model(variant, seed_file, device)
    if method == "B0":
        checkpoint = Path(config.checkpoint_path)
        gate = GatedTransFuser(model, None).to(device) if variant == "transfuser" else None
        return gate, model, config, {"checkpoint": str(checkpoint),
                                     "checkpoint_sha256": _sha256(checkpoint)}
    if variant != "transfuser":
        return None, model, config, {
            "checkpoint": None,
            "status": f"trained gated method {method} is not applicable to {variant}",
        }
    checkpoint = Path(artifacts_root) / "tf_checkpoints" / variant / f"fusion_{method}_seed{seed}.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    estimator_state = payload.get("estimator")
    estimator = (StaticGate().to(device) if method == "B4"
                 else _checkpoint_estimator(estimator_state, device))
    if method == "B4" and estimator_state:
        estimator.load_state_dict(estimator_state)
    meta = payload.get("meta", {})
    mechanism = meta.get("mechanism", MECHANISMS.get(method, "log_prior"))
    stages = tuple(meta.get("stages", (0, 1, 2, 3)))
    gate = GatedTransFuser(
        model, estimator, mechanism=mechanism, stages=stages,
        alpha=project_config["fusion"]["alpha"], eps=project_config["fusion"]["epsilon"],
        gate_enabled_by_default=method != "BQ").to(device)
    gate_state = payload.get("gate") or {}
    if gate.key_gamma is not None and gate_state.get("key_gamma") is not None:
        gate.key_gamma.data.copy_(torch.as_tensor(gate_state["key_gamma"], device=device))
    return gate, model, config, {"checkpoint": str(checkpoint),
                                 "checkpoint_sha256": _sha256(checkpoint), "meta": meta}


def _stack(items):
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def _loader(root, ids, spec, seed, config, batch_size):
    for offset in range(0, len(ids), batch_size):
        batch_ids = ids[offset:offset + batch_size]
        prepared = []
        for sample_id in batch_ids:
            raw, _ = corrupt_tf_sample(load_raw_tf(root, sample_id), spec, seed)
            prepared.append(preprocess_tf(raw, config))
        yield batch_ids, _stack(prepared)


def _split_samples(root, split, name):
    """Accept normal episode splits and sample-id leaves used by isolated smoke runs."""
    leaves = sum(split[name].values(), [])
    if leaves and all("/" in value for value in leaves):
        return sorted(set(leaves))
    return data.split_samples(root, split, name)


def _truth(ids, targets, config, device):
    boxes = [targets[sample_id] for sample_id in ids]
    heatmaps = torch.stack([render_tf_heatmap(box, config) for box in boxes]).to(device)
    return heatmaps, boxes


def matched_center_count_1m(prediction, target, pixels_per_meter, score_threshold=0.05):
    """Count score-ordered, one-to-one predicted/target centre matches within 1 m."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    detections = prediction[prediction[:, 7] >= score_threshold]
    detections = detections[np.argsort(-detections[:, 7], kind="stable")]
    used = np.zeros(len(target), dtype=bool)
    matched = 0
    for row in detections:
        if not len(target):
            break
        distances = np.linalg.norm(target[:, :2] - row[None, :2], axis=1) / pixels_per_meter
        distances[used] = np.inf
        best = int(np.argmin(distances))
        if distances[best] <= 1.0:
            used[best] = True
            matched += 1
    return matched


def prediction_losses(output, target_heatmaps, target_boxes, pixels_per_meter):
    errors = heatmap_reference_error(output["center_heatmap"][:, 0], target_heatmaps)
    boxes = output["boxes"].detach().cpu().numpy()
    matches = torch.tensor([matched_center_count_1m(pred, target, pixels_per_meter)
                            for pred, target in zip(boxes, target_boxes)], device=errors.device)
    return errors, matches


def removal_outputs(gate, batch, mean_features):
    """Run sensor and stage interventions without copying or changing weights."""
    outputs = {}
    for modality in ("camera", "lidar"):
        outputs[("sensor-level removal", "sensor", modality)] = gate(
            batch, sensor_remove=modality)
        outputs[("stage-level removal (all stages)", "stage", modality)] = gate(
            batch, modality_mask=modality)
        outputs[("stage-level mean replacement (all stages)", "stage", modality)] = gate(
            batch, modality_replace=modality, mean_features=mean_features)
    return outputs


@torch.no_grad()
def mean_training_features(gate, root, ids, config, seed, cache, device, batch_size=4):
    """Cache per-stage/channel clean feature means from at most 512 training samples."""
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)["mean"]
    if not ids:
        raise RuntimeError("training split contains no samples for mean-feature utility")
    rng = np.random.default_rng(512)
    if len(ids) > 512:
        ids = [ids[i] for i in sorted(rng.choice(len(ids), 512, replace=False))]
    backbone = gate.model._model
    image_layers = [backbone.image_encoder.features.layer1, backbone.image_encoder.features.layer2,
                    backbone.image_encoder.features.layer3, backbone.image_encoder.features.layer4]
    lidar_layers = [backbone.lidar_encoder._model.layer1, backbone.lidar_encoder._model.layer2,
                    backbone.lidar_encoder._model.layer3, backbone.lidar_encoder._model.layer4]
    totals, counts, handles = {}, Counter(), []

    def capture(stage, modality):
        def hook(_module, _inputs, output):
            key = (stage, modality)
            value = output.detach().sum((0, 2, 3)).cpu()
            totals[key] = totals.get(key, torch.zeros_like(value)) + value
            counts[key] += output.shape[0] * output.shape[2] * output.shape[3]
        return hook

    for stage, (image_layer, lidar_layer) in enumerate(zip(image_layers, lidar_layers)):
        handles.append(image_layer.register_forward_hook(capture(stage, "camera")))
        handles.append(lidar_layer.register_forward_hook(capture(stage, "lidar")))
    try:
        for _, batch in _loader(root, ids, {}, seed, config, batch_size):
            gate(batch, gate_enabled=False)
    finally:
        for handle in handles:
            handle.remove()
    means = {stage: {modality: totals[(stage, modality)] / counts[(stage, modality)]
                     for modality in ("camera", "lidar")} for stage in range(4)}
    torch.save({"mean": means, "n_samples": len(ids)}, cache)
    return means


def _condition_fields(condition, spec, cfg_corr):
    camera, lidar = spec.get("camera"), spec.get("lidar")
    return {"condition": condition, "kind": condition_kind(spec, cfg_corr),
            "camera_family": camera[0] if camera else "",
            "camera_severity": camera[1] if camera else "",
            "lidar_family": lidar[0] if lidar else "",
            "lidar_severity": lidar[1] if lidar else ""}


def _base_rows(ids, condition, spec, cfg_corr, boxes):
    fields = _condition_fields(condition, spec, cfg_corr)
    return [dict(sample_id=sample_id, episode_id=sample_id.split("/")[0],
                 fg_present=bool(len(box)), **fields)
            for sample_id, box in zip(ids, boxes)]


def _append_predictions(store, variant, condition, output):
    store[variant][condition]["heatmaps"].append(output["center_heatmap"][:, 0].detach().cpu().numpy())
    store[variant][condition]["boxes"].extend(output["boxes"].detach().cpu().numpy())


def _metric_summary(predictions, truth):
    summary = {}
    for variant, conditions in predictions.items():
        summary[variant] = {}
        for condition, prediction in conditions.items():
            target_heatmaps = np.concatenate(truth[condition]["heatmaps"])
            heatmaps = np.concatenate(prediction["heatmaps"])
            valid = (target_heatmaps >= 0.99) | (target_heatmaps < 0.01)
            ap, _ = center_distance_ap(prediction["boxes"], truth[condition]["boxes"],
                                       truth[condition]["pixels_per_meter"], 1.0)
            summary[variant][condition] = {
                "heatmap_peak_ap": metrics.grid_fg_ap(heatmaps[valid], target_heatmaps[valid] >= 0.99),
                "vehicle_center_ap_1m": ap,
            }
    return summary


def _new_predictions():
    return defaultdict(lambda: defaultdict(lambda: {"heatmaps": [], "boxes": []}))


def _remember_truth(store, condition, heatmaps, boxes, pixels_per_meter):
    entry = store.setdefault(condition, {"heatmaps": [], "boxes": [],
                                         "pixels_per_meter": pixels_per_meter})
    entry["heatmaps"].append(heatmaps.detach().cpu().numpy())
    entry["boxes"].extend(boxes)


def _spearman(rows, x, y):
    left = np.asarray([row[x] for row in rows])
    right = np.asarray([row[y] for row in rows])
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).correlation)


def _empty_ci(rows):
    return {"estimate": None, "ci_low": None, "ci_high": None,
            "n_episodes": len({row["episode_id"] for row in rows}), "repeats": 2000}


def _utility_summary(rows, seed):
    summary = {}
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        summary[variant] = {}
        groups = {"overall": variant_rows,
                  "fg_present": [row for row in variant_rows if row["fg_present"]],
                  "fg_absent": [row for row in variant_rows if not row["fg_present"]]}
        groups.update({kind: [row for row in variant_rows
                              if row["kind"] == kind and row["fg_present"]]
                       for kind in sorted({row["kind"] for row in variant_rows})})
        for group, group_rows in groups.items():
            summary[variant][group] = {}
            for modality in ("camera", "lidar"):
                modality_rows = [row for row in group_rows if row["modality"] == modality]
                pairs = (("q_vs_neg_E", "q", "neg_E"),
                         ("q_vs_utility", "q", "utility"),
                         ("E_vs_neg_utility", "E", "neg_utility"))
                summary[variant][group][modality] = {
                    name: (stats.episode_bootstrap_ci(
                        modality_rows, lambda sample, a=a, b=b: _spearman(sample, a, b),
                        n=2000, seed=seed) if modality_rows else _empty_ci(modality_rows))
                    for name, a, b in pairs}
    return summary


@torch.no_grad()
def run_utility(gate, reference, project, split, ids, suite, targets, output_dir,
                device, batch_size, seed, config):
    train_ids = _split_samples(project["dataset_root"], split, "train")
    means = mean_training_features(gate, project["dataset_root"], train_ids, config,
                                   project["corruption"]["base_seed"],
                                   output_dir / "mean_features_train512.pt", device, batch_size)
    rows, predictions, truth = [], _new_predictions(), {}
    for condition, spec in suite.items():
        for batch_ids, batch in _loader(project["dataset_root"], ids, spec,
                                        project["corruption"]["base_seed"], config, batch_size):
            heatmaps, boxes = _truth(batch_ids, targets, config, device)
            full = gate(batch)
            full_error, full_matches = prediction_losses(full, heatmaps, boxes, config.pixels_per_meter)
            _remember_truth(truth, condition, heatmaps, boxes, config.pixels_per_meter)
            _append_predictions(predictions, "full", condition, full)
            q_star, errors = q_targets(reference, batch, heatmaps,
                                       project["reliability"]["tau_quality"])
            variants = removal_outputs(gate, batch, means)
            bases = _base_rows(batch_ids, condition, spec, project["corruption"], boxes)
            for (variant, removal_level, modality), output in variants.items():
                error, matches = prediction_losses(output, heatmaps, boxes, config.pixels_per_meter)
                _append_predictions(predictions, f"{variant}_{modality}", condition, output)
                modality_index = 0 if modality == "camera" else 1
                for index, base in enumerate(bases):
                    utility = float(error[index] - full_error[index])
                    rows.append(dict(base, variant=variant, removal_level=removal_level,
                                     modality=modality,
                                     q=float(full["q"][index, modality_index]),
                                     q_star=float(q_star[index, modality_index]),
                                     E=float(errors[index, modality_index]),
                                     neg_E=float(-errors[index, modality_index]),
                                     q_camera=float(full["q"][index, 0]), q_lidar=float(full["q"][index, 1]),
                                     q_star_camera=float(q_star[index, 0]), q_star_lidar=float(q_star[index, 1]),
                                     E_camera=float(errors[index, 0]), E_lidar=float(errors[index, 1]),
                                     full_heatmap_reference_error=float(full_error[index]),
                                     heatmap_reference_error=float(error[index]),
                                     full_matched_center_count_1m=int(full_matches[index]),
                                     matched_center_count_1m=int(matches[index]),
                                     utility=utility, neg_utility=-utility,
                                     matched_center_utility=int(full_matches[index] - matches[index])))
    write_csv(output_dir / "utility.csv", rows)
    return {"label": "removal utility", "role": "sensitivity",
            "correlations": _utility_summary(rows, seed),
            "ap": _metric_summary(predictions, truth)}


@torch.no_grad()
def run_partial(gate, reference, project, ids, suite, targets, output_dir,
                device, batch_size, seed, config):
    """Measure local loss changes when one learned quality is scaled at a time."""
    rows = []
    for condition, spec in suite.items():
        for batch_ids, batch in _loader(project["dataset_root"], ids, spec,
                                        project["corruption"]["base_seed"], config, batch_size):
            heatmaps, boxes = _truth(batch_ids, targets, config, device)
            full = gate(batch)
            full_error, full_matches = prediction_losses(
                full, heatmaps, boxes, config.pixels_per_meter)
            _, errors = q_targets(reference, batch, heatmaps,
                                  project["reliability"]["tau_quality"])
            bases = _base_rows(batch_ids, condition, spec, project["corruption"], boxes)
            for modality_index, modality in enumerate(("camera", "lidar")):
                for factor in PARTIAL_FACTORS:
                    q_score = full["q"].clone()
                    q_score[:, modality_index] = (
                        q_score[:, modality_index] * factor).clamp(gate.eps, 1.0)
                    output = gate(batch, q_override=q_score)
                    error, matches = prediction_losses(
                        output, heatmaps, boxes, config.pixels_per_meter)
                    for index, base in enumerate(bases):
                        rows.append(dict(
                            base, modality=modality, factor=factor,
                            q=float(full["q"][index, modality_index]),
                            intervened_q=float(q_score[index, modality_index]),
                            E=float(errors[index, modality_index]),
                            full_heatmap_reference_error=float(full_error[index]),
                            heatmap_reference_error=float(error[index]),
                            delta_heatmap_reference_error=float(error[index] - full_error[index]),
                            full_negative_matched_center_count_1m=-int(full_matches[index]),
                            negative_matched_center_count_1m=-int(matches[index]),
                            delta_negative_matched_center_count_1m=int(
                                full_matches[index] - matches[index])))
    write_csv(output_dir / "partial.csv", rows)
    return {"foreground_by_kind": partial_summary(
        rows, ("heatmap_reference_error", "negative_matched_center_count_1m"), seed)}


@torch.no_grad()
def run_oracle_rel(gate, reference, project, ids, suite, targets, output_dir,
                   device, batch_size, config, method):
    rows, predictions, truth = [], _new_predictions(), {}
    variants = ["absolute"] + (["relative"] if method == "PR" else [])
    for condition, spec in suite.items():
        corrupt = _loader(project["dataset_root"], ids, spec, project["corruption"]["base_seed"], config, batch_size)
        clean = _loader(project["dataset_root"], ids, {}, project["corruption"]["base_seed"], config, batch_size)
        for (batch_ids, batch), (_, clean_batch) in zip(corrupt, clean):
            heatmaps, boxes = _truth(batch_ids, targets, config, device)
            absolute_q, absolute_error = q_targets(reference, batch, heatmaps,
                                                   project["reliability"]["tau_quality"])
            candidates = {"absolute": (absolute_q, absolute_error)}
            if "relative" in variants:
                relative_q, relative_error, _, _ = q_targets_relative(
                    reference, batch, clean_batch, heatmaps, project["reliability"]["tau_relative"])
                candidates["relative"] = (relative_q, relative_error)
            _remember_truth(truth, condition, heatmaps, boxes, config.pixels_per_meter)
            bases = _base_rows(batch_ids, condition, spec, project["corruption"], boxes)
            for variant, (q_score, reference_error) in candidates.items():
                output = gate(batch, q_override=q_score)
                error, matches = prediction_losses(output, heatmaps, boxes, config.pixels_per_meter)
                _append_predictions(predictions, variant, condition, output)
                for index, base in enumerate(bases):
                    rows.append(dict(base, variant=variant,
                                     q_camera=float(q_score[index, 0]), q_lidar=float(q_score[index, 1]),
                                     E_camera=float(reference_error[index, 0]), E_lidar=float(reference_error[index, 1]),
                                     heatmap_reference_error=float(error[index]),
                                     matched_center_count_1m=int(matches[index])))
    write_csv(output_dir / "oracle_rel.csv", rows)
    return {"ap": _metric_summary(predictions, truth)}


@torch.no_grad()
def run_oracle_gate(gate, project, ids, suite, targets, output_dir, device,
                    batch_size, config, xfit=None, temporal=None):
    candidates = [(camera, lidar) for camera in GRID for lidar in GRID]
    rows, predictions, truth = [], _new_predictions(), {}
    loss_sum, loss_count, chosen = Counter(), Counter(), Counter()
    learned_loss_violations = 0
    add_temporal = temporal is not None and not temporal.has_data
    for condition, spec in suite.items():
        for batch_ids, batch in _loader(project["dataset_root"], ids, spec,
                                        project["corruption"]["base_seed"], config, batch_size):
            heatmaps, boxes = _truth(batch_ids, targets, config, device)
            _remember_truth(truth, condition, heatmaps, boxes, config.pixels_per_meter)
            bases = _base_rows(batch_ids, condition, spec, project["corruption"], boxes)
            outputs, losses = [], []
            row_start = len(rows)
            for camera, lidar in candidates:
                q_score = torch.tensor([camera, lidar], device=device).expand(len(batch_ids), 2)
                output = gate(batch, q_override=q_score)
                error, matches = prediction_losses(output, heatmaps, boxes, config.pixels_per_meter)
                outputs.append(output)
                losses.append(error)
                key = f"{camera:g},{lidar:g}"
                loss_sum[key] += float(error.sum())
                loss_count[key] += len(error)
                if key == "1,1":
                    _append_predictions(predictions, "neutral", condition, output)
                for index, base in enumerate(bases):
                    rows.append(dict(base, variant=key, q_camera=camera, q_lidar=lidar,
                                     heatmap_reference_error=float(error[index]),
                                     matched_center_count_1m=int(matches[index]), selected=False))
            learned_output = gate(batch)
            learned_error, learned_matches = prediction_losses(
                learned_output, heatmaps, boxes, config.pixels_per_meter)
            outputs.append(learned_output)
            losses.append(learned_error)
            for index, base in enumerate(bases):
                rows.append(dict(base, variant="learned_q",
                                 q_camera=float(learned_output["q"][index, 0]),
                                 q_lidar=float(learned_output["q"][index, 1]),
                                 heatmap_reference_error=float(learned_error[index]),
                                 matched_center_count_1m=int(learned_matches[index]),
                                 selected=False))
            best = torch.stack(losses, 1).argmin(1).cpu().numpy()
            candidate_heatmaps = np.stack([
                output["center_heatmap"][:, 0].cpu().numpy() for output in outputs])
            learned = candidate_heatmaps[-1]
            violations = torch.stack(losses, 1).min(1).values > learned_error + 1e-8
            learned_loss_violations += int(violations.sum())
            assert not violations.any(), "in-sample oracle loss exceeded learned-q loss"
            if xfit is not None:
                xfit.add(condition, candidate_heatmaps, learned, heatmaps.cpu().numpy())
            if add_temporal:
                temporal.add(
                    condition, batch_ids, candidate_heatmaps,
                    torch.stack(losses).cpu().numpy(), learned, heatmaps.cpu().numpy(),
                    np.stack([output["boxes"].cpu().numpy() for output in outputs]),
                    learned_output["boxes"].cpu().numpy(), boxes)
            for index, candidate_index in enumerate(best):
                rows[row_start + int(candidate_index) * len(batch_ids) + index]["selected"] = True
                key = ("learned_q" if candidate_index == len(candidates)
                       else f"{candidates[candidate_index][0]:g},{candidates[candidate_index][1]:g}")
                chosen[(_condition_fields(condition, spec, project["corruption"])["kind"], key)] += 1
                selected = {name: value[index:index + 1] if torch.is_tensor(value) else value
                            for name, value in outputs[candidate_index].items()
                            if name in ("center_heatmap", "boxes")}
                _append_predictions(predictions, "oracle", condition, selected)
    write_csv(output_dir / "oracle_gate.csv", rows)
    best_fixed = min(loss_sum, key=lambda key: loss_sum[key] / loss_count[key])
    fixed_q = torch.tensor([float(value) for value in best_fixed.split(",")], device=device)
    add_temporal_fixed = temporal is not None and temporal.has_data and not temporal.fixed_complete
    for condition, spec in suite.items():
        for batch_ids, batch in _loader(project["dataset_root"], ids, spec,
                                        project["corruption"]["base_seed"], config, batch_size):
            fixed_output = gate(batch, q_override=fixed_q.expand(len(batch_ids), 2))
            _append_predictions(predictions, "best_fixed", condition, fixed_output)
            if add_temporal_fixed:
                temporal.add_fixed(condition, fixed_output["center_heatmap"][:, 0].cpu().numpy(),
                                   fixed_output["boxes"].cpu().numpy())
    metric_summary = _metric_summary(predictions, truth)
    return {"oracle_ap": metric_summary["oracle"], "neutral_ap": metric_summary["neutral"],
            "best_fixed_candidate": best_fixed,
            "best_fixed_mean_heatmap_reference_error": loss_sum[best_fixed] / loss_count[best_fixed],
            "best_fixed_ap": metric_summary["best_fixed"],
            "in_sample_oracle_learned_loss_violations": learned_loss_violations,
            "chosen_histogram_by_kind": {
                kind: {key: count for (entry_kind, key), count in chosen.items() if entry_kind == kind}
                for kind in sorted({kind for kind, _ in chosen})}}


@torch.no_grad()
def run_oracle_gate_xfit(gate, project, ids, suite, targets, device,
                         batch_size, config, xfit, temporal=None):
    """Select candidates on one fixed heatmap half and score only on the other."""
    candidates = [(camera, lidar) for camera in GRID for lidar in GRID]
    add_temporal = temporal is not None and not temporal.has_data
    if not xfit.loss_counts.any():
        for condition, spec in suite.items():
            for batch_ids, batch in _loader(project["dataset_root"], ids, spec,
                                            project["corruption"]["base_seed"], config, batch_size):
                heatmaps, _ = _truth(batch_ids, targets, config, device)
                outputs, losses = [], []
                for camera, lidar in candidates:
                    q_score = torch.tensor([camera, lidar], device=device).expand(len(batch_ids), 2)
                    output = gate(batch, q_override=q_score)
                    outputs.append(output)
                    if temporal is not None:
                        losses.append(heatmap_reference_error(output["center_heatmap"][:, 0], heatmaps))
                learned_output = gate(batch)
                learned = learned_output["center_heatmap"][:, 0].cpu().numpy()
                outputs.append(learned_output)
                if temporal is not None:
                    losses.append(heatmap_reference_error(
                        learned_output["center_heatmap"][:, 0], heatmaps))
                candidate_heatmaps = np.stack([
                    output["center_heatmap"][:, 0].cpu().numpy() for output in outputs])
                xfit.add(condition, candidate_heatmaps, learned, heatmaps.cpu().numpy())
                if add_temporal:
                    temporal.add(
                        condition, batch_ids, candidate_heatmaps,
                        torch.stack(losses).cpu().numpy(), learned, heatmaps.cpu().numpy(),
                        np.stack([output["boxes"].cpu().numpy() for output in outputs]),
                        learned_output["boxes"].cpu().numpy(),
                        [targets[sample_id] for sample_id in batch_ids])
    best = xfit.best_indices
    temporal_best = temporal.best_index if temporal is not None and temporal.has_data else None
    add_temporal_fixed = temporal is not None and temporal.has_data and not temporal.fixed_complete
    for condition, spec in suite.items():
        for batch_ids, batch in _loader(project["dataset_root"], ids, spec,
                                        project["corruption"]["base_seed"], config, batch_size):
            heatmaps, _ = _truth(batch_ids, targets, config, device)
            fixed = {}
            fixed_indices = set(map(int, best))
            if temporal_best is not None:
                fixed_indices.add(temporal_best)
            for index in fixed_indices:
                q_score = torch.tensor(candidates[index], device=device).expand(len(batch_ids), 2)
                fixed[index] = gate(batch, q_override=q_score)
            for fold, index in enumerate(best):
                xfit.add_fixed(condition, fold,
                               fixed[int(index)]["center_heatmap"][:, 0].cpu().numpy(),
                               heatmaps.cpu().numpy())
            if add_temporal_fixed:
                output = fixed[temporal_best]
                temporal.add_fixed(condition, output["center_heatmap"][:, 0].cpu().numpy(),
                                   output["boxes"].cpu().numpy())
    return xfit.summary()


@torch.no_grad()
def run_oracle_gate_temporal(gate, project, ids, suite, targets, device,
                             batch_size, config, temporal):
    """Choose each frame's candidate from the prior evaluated frame in its episode."""
    candidates = [(camera, lidar) for camera in GRID for lidar in GRID]
    if not temporal.has_data:
        for condition, spec in suite.items():
            for batch_ids, batch in _loader(project["dataset_root"], ids, spec,
                                            project["corruption"]["base_seed"], config, batch_size):
                heatmaps, boxes = _truth(batch_ids, targets, config, device)
                outputs, losses = [], []
                for camera, lidar in candidates:
                    q_score = torch.tensor([camera, lidar], device=device).expand(len(batch_ids), 2)
                    output = gate(batch, q_override=q_score)
                    outputs.append(output)
                    losses.append(heatmap_reference_error(output["center_heatmap"][:, 0], heatmaps))
                learned_output = gate(batch)
                outputs.append(learned_output)
                losses.append(heatmap_reference_error(
                    learned_output["center_heatmap"][:, 0], heatmaps))
                temporal.add(
                    condition, batch_ids,
                    np.stack([output["center_heatmap"][:, 0].cpu().numpy() for output in outputs]),
                    torch.stack(losses).cpu().numpy(),
                    learned_output["center_heatmap"][:, 0].cpu().numpy(), heatmaps.cpu().numpy(),
                    np.stack([output["boxes"].cpu().numpy() for output in outputs]),
                    learned_output["boxes"].cpu().numpy(), boxes)
    if not temporal.fixed_complete:
        fixed_q = torch.tensor(candidates[temporal.best_index], device=device)
        for condition, spec in suite.items():
            for batch_ids, batch in _loader(project["dataset_root"], ids, spec,
                                            project["corruption"]["base_seed"], config, batch_size):
                output = gate(batch, q_override=fixed_q.expand(len(batch_ids), 2))
                temporal.add_fixed(condition, output["center_heatmap"][:, 0].cpu().numpy(),
                                   output["boxes"].cpu().numpy())
    summary = temporal.summary()
    for variant, conditions in summary["ap"].items():
        for condition, heatmap_ap in conditions.items():
            center_ap, _ = center_distance_ap(
                temporal.payloads[variant][condition], temporal.target_payloads[condition],
                config.pixels_per_meter, 1.0)
            conditions[condition] = {"heatmap_peak_ap": heatmap_ap,
                                     "vehicle_center_ap_1m": center_ap}
    return summary


@torch.no_grad()
def run_shuffle(gate, project, ids, suite, targets, output_dir, device,
                batch_size, config, seed):
    records, condition_data = [], {}
    for condition, spec in suite.items():
        scores, boxes, qualities = [], [], []
        for batch_ids, batch in _loader(project["dataset_root"], ids, spec,
                                        project["corruption"]["base_seed"], config, batch_size):
            output = gate(batch)
            qualities.append(output["q"].cpu())
            scores.append(output["center_heatmap"][:, 0].cpu().numpy())
            boxes.extend(output["boxes"].cpu().numpy())
            records.extend((condition, sample_id, output["q"][index].cpu())
                           for index, sample_id in enumerate(batch_ids))
        condition_data[condition] = {"spec": spec, "q": torch.cat(qualities),
                                     "heatmaps": np.concatenate(scores), "boxes": boxes}
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(records))
    global_source = {records[index][:2]: records[source] for index, source in enumerate(permutation)}
    rows, predictions, truth = [], _new_predictions(), {}
    offsets = defaultdict(int)
    episodes = [sample_id.split("/")[0] for sample_id in ids]
    for condition, payload in condition_data.items():
        within = cross_episode_permutation(episodes, seed + sum(condition.encode("utf-8")))
        for batch_ids, batch in _loader(project["dataset_root"], ids, payload["spec"],
                                        project["corruption"]["base_seed"], config, batch_size):
            heatmaps, target_boxes = _truth(batch_ids, targets, config, device)
            _remember_truth(truth, condition, heatmaps, target_boxes, config.pixels_per_meter)
            start, stop = offsets[condition], offsets[condition] + len(batch_ids)
            learned_q = payload["q"][start:stop].to(device)
            global_q = torch.stack([global_source[(condition, sample_id)][2]
                                    for sample_id in batch_ids]).to(device)
            within_q = payload["q"][within[start:stop]].to(device)
            variants = {"learned": learned_q, "global": global_q, "within_condition": within_q}
            bases = _base_rows(batch_ids, condition, payload["spec"], project["corruption"], target_boxes)
            for variant, q_score in variants.items():
                output = gate(batch, q_override=q_score)
                error, matches = prediction_losses(output, heatmaps, target_boxes, config.pixels_per_meter)
                _append_predictions(predictions, variant, condition, output)
                for index, base in enumerate(bases):
                    if variant == "global":
                        source_id = global_source[(condition, batch_ids[index])][1]
                    elif variant == "within_condition":
                        source_id = ids[within[start + index]]
                    else:
                        source_id = batch_ids[index]
                    rows.append(dict(base, variant=variant,
                                     q_camera=float(q_score[index, 0]), q_lidar=float(q_score[index, 1]),
                                     source_sample_id=source_id, source_episode_id=source_id.split("/")[0],
                                     heatmap_reference_error=float(error[index]),
                                     matched_center_count_1m=int(matches[index])))
            offsets[condition] = stop
    write_csv(output_dir / "shuffle.csv", rows)
    return {"ap": _metric_summary(predictions, truth)}


def _attention_columns(attention, index):
    columns = {}
    for stage, layers in attention.items():
        for layer, mass in layers.items():
            prefix = f"{stage}_{layer}"
            columns[f"{prefix}_camera_mass"] = float(mass[index, 0])
            columns[f"{prefix}_lidar_mass"] = float(mass[index, 1])
    return columns


@torch.no_grad()
def run_trace(gate, project, ids, suite, targets, output_dir, device,
              batch_size, config, seed):
    trace_conditions = ("clean", "camera_gaussian_blur_s5", "lidar_point_dropout_s5")
    missing = [condition for condition in trace_conditions if condition not in suite]
    if missing:
        raise RuntimeError(f"trace conditions missing from suite: {missing}")
    pool = [(condition, sample_id) for condition in trace_conditions for sample_id in ids]
    rng = np.random.default_rng(seed)
    selected = [pool[index] for index in sorted(rng.choice(len(pool), min(64, len(pool)), replace=False))]
    selected_ids = defaultdict(list)
    for condition, sample_id in selected:
        selected_ids[condition].append(sample_id)
    rows, predictions, truth = [], _new_predictions(), {}
    for condition, condition_ids in selected_ids.items():
        spec = suite[condition]
        for batch_ids, batch in _loader(project["dataset_root"], condition_ids, spec,
                                        project["corruption"]["base_seed"], config, batch_size):
            heatmaps, boxes = _truth(batch_ids, targets, config, device)
            neutral = gate(batch, q_override=torch.ones(len(batch_ids), 2, device=device))
            _remember_truth(truth, condition, heatmaps, boxes, config.pixels_per_meter)
            bases = _base_rows(batch_ids, condition, spec, project["corruption"], boxes)
            for modality in ("camera", "lidar"):
                for value in TRACE_GRID:
                    q_score = torch.ones(len(batch_ids), 2, device=device)
                    q_score[:, 0 if modality == "camera" else 1] = value
                    output = gate(batch, q_override=q_score, return_attention=True)
                    error, matches = prediction_losses(output, heatmaps, boxes, config.pixels_per_meter)
                    delta = (output["center_heatmap"] - neutral["center_heatmap"]).abs().flatten(1).mean(1)
                    variant = f"{modality}_{value:g}"
                    _append_predictions(predictions, variant, condition, output)
                    for index, base in enumerate(bases):
                        rows.append(dict(base, variant=variant,
                                         q_camera=float(q_score[index, 0]), q_lidar=float(q_score[index, 1]),
                                         heatmap_reference_error=float(error[index]),
                                         matched_center_count_1m=int(matches[index]),
                                         mean_abs_delta_score=float(delta[index]),
                                         **_attention_columns(output["attention"], index)))
    write_csv(output_dir / "trace.csv", rows)
    return {"n_fixed_samples": len(selected),
            "conditions": {condition: len(values) for condition, values in selected_ids.items()},
            "ap": _metric_summary(predictions, truth)}


def _skip_reason(variant, method, analysis):
    if variant != "transfuser":
        return (f"{variant} does not expose TransFuser's shared camera/LiDAR GPT stages; "
                f"{analysis} is not applicable")
    if method == "B0" and analysis != "utility":
        return f"B0 supports utility only; {analysis} requires a trained gated method"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/journal.yaml")
    parser.add_argument("--variant", choices=VARIANTS, default="transfuser")
    parser.add_argument("--method", default="B0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--split-file", "--split-json", dest="split_file")
    parser.add_argument("--suite", choices=("reduced", "full"), default="reduced")
    parser.add_argument("--root")
    parser.add_argument("--artifacts-root")
    parser.add_argument("--seed-file")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--analyses", default="utility")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    analyses = [value.strip() for value in args.analyses.split(",") if value.strip()]
    unknown = set(analyses) - set(ANALYSES)
    if unknown:
        parser.error(f"unknown analyses: {sorted(unknown)}")
    if "oracle_gate_temporal" in analyses:
        analyses = [name for name in analyses if name != "oracle_gate_temporal"] + ["oracle_gate_temporal"]

    started = time.time()
    cfg = load_config(args.config)
    root = str(Path(args.root or cfg["paths"]["dataset_root"]))
    artifacts_root = Path(args.artifacts_root or cfg["paths"]["artifacts_root"])
    split_path = Path(args.split_file) if args.split_file else artifacts_root / "manifests" / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    ids = _split_samples(root, split, args.split)[::args.frame_stride]
    if not ids:
        raise RuntimeError(f"no {args.split} samples found under {root}")
    suite = reduced_suite(cfg["corruption"]) if args.suite == "reduced" else full_suite(cfg["corruption"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gate, _model, config, provenance = load_method(
        args.variant, args.method, args.seed, artifacts_root, device, cfg, args.seed_file)
    if gate is not None:
        gate.eval()
    reference = None
    if gate is not None and set(analyses) & {"utility", "partial", "oracle_rel"}:
        if args.method == "B0":
            reference = gate
        else:
            reference_model, _ = load_tf_model("transfuser", args.seed_file, device)
            reference = GatedTransFuser(reference_model, None).to(device).eval()
    output_dir = artifacts_root / "tf_diagnostics" / f"{args.variant}_{args.method}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {sample_id: tf_targets(root, sample_id, config) for sample_id in ids}
    project = {"dataset_root": root, "corruption": cfg["corruption"], "reliability": cfg["reliability"]}
    candidates = ([f"{camera:g},{lidar:g}" for camera in GRID for lidar in GRID]
                  + ["learned_q"])
    xfit = (GateXfitAccumulator(64 * 64, candidates, candidates.index("1,1"),
                                args.seed, "transfuser")
            if "oracle_gate_xfit" in analyses else None)
    temporal = (GateTemporalAccumulator(candidates, candidates.index("1,1"), "transfuser")
                if "oracle_gate_temporal" in analyses else None)

    runners = {
        "utility": lambda: run_utility(gate, reference, project, split, ids, suite, targets,
                                        output_dir, device, args.batch_size, args.seed, config),
        "partial": lambda: run_partial(gate, reference, project, ids, suite, targets,
                                        output_dir, device, args.batch_size, args.seed, config),
        "oracle_rel": lambda: run_oracle_rel(gate, reference, project, ids, suite, targets,
                                              output_dir, device, args.batch_size, config, args.method),
        "oracle_gate": lambda: run_oracle_gate(gate, project, ids, suite, targets, output_dir,
                                                device, args.batch_size, config, xfit, temporal),
        "oracle_gate_xfit": lambda: run_oracle_gate_xfit(
            gate, project, ids, suite, targets, device, args.batch_size, config, xfit, temporal),
        "oracle_gate_temporal": lambda: run_oracle_gate_temporal(
            gate, project, ids, suite, targets, device, args.batch_size, config, temporal),
        "shuffle": lambda: run_shuffle(gate, project, ids, suite, targets, output_dir,
                                        device, args.batch_size, config, args.seed),
        "trace": lambda: run_trace(gate, project, ids, suite, targets, output_dir,
                                    device, args.batch_size, config, args.seed),
    }
    summary = {"variant": args.variant, "method": args.method, "seed": args.seed,
               "split": args.split, "suite": args.suite, "frame_stride": args.frame_stride,
               "n_samples": len(ids), "n_conditions": len(suite), "device": device,
               "provenance": provenance, "analyses": {}}
    summary_path = output_dir / "summary.json"
    merge_summary(summary_path, summary)
    for analysis in analyses:
        tick = time.time()
        reason = _skip_reason(args.variant, args.method, analysis)
        if reason:
            print(f"skipping {analysis}: {reason}", flush=True)
            summary["analyses"][analysis] = {"skipped": reason, "wall_seconds": 0.0}
        else:
            print(f"running {analysis}...", flush=True)
            summary["analyses"][analysis] = runners[analysis]()
            summary["analyses"][analysis]["wall_seconds"] = round(time.time() - tick, 1)
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    summary["wall_seconds"] = round(time.time() - started, 1)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
