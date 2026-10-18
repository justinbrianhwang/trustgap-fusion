"""Frozen-checkpoint diagnostics for reliability targets and fusion interventions."""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from rrf import data, evaluate, metrics, upstream
from rrf.config import load_config
from rrf.diag_common import GateTemporalAccumulator, GateXfitAccumulator, partial_summary
from rrf.gated import VIEW_MODALITY, assemble, tokenize
from rrf.probes import q_targets, q_targets_relative, reference_error
from rrf.stats import episode_bootstrap_ci
from rrf.tools.evaluate import load_method
from rrf.train import GATED_METHODS, load_reference


REDUCED_NAMES = (
    "clean",
    "camera_gaussian_blur_s3",
    "camera_gaussian_blur_s5",
    "camera_darkening_s5",
    "lidar_point_dropout_s3",
    "lidar_point_dropout_s5",
    "camera_rectangular_occlusion_s5",
    "lidar_azimuth_sector_dropout_s5",
    "mixed_gaussian_blur_point_dropout_s3",
    "camera_silent_blackout",
    "lidar_silent_empty",
)
GRID = (0.05, 0.2, 0.5, 1.0)
PARTIAL_FACTORS = (0.5, 0.8, 1.25, 2.0)
TRACE_GRID = (0.05, 0.1, 0.2, 0.5, 1.0)
_LOADER_WORKERS = 4


def reduced_suite(cfg_corr):
    full = evaluate.full_suite(cfg_corr)
    missing = [name for name in REDUCED_NAMES if name not in full]
    if missing:
        raise RuntimeError(f"reduced-suite conditions missing from full_suite: {missing}")
    return {name: full[name] for name in REDUCED_NAMES}


def per_sample_native_score(output, target):
    """Training's balanced foreground/background channel-0 L1, separately per sample."""
    m1, m0 = target[..., 0].ge(0.01), target[..., 0].le(0.01)
    err = (output[..., 0] - target[..., 0]).abs()
    lp1 = (err * m1).sum(1) / m1.sum(1).clamp_min(1)
    lp0 = (err * m0).sum(1) / m0.sum(1).clamp_min(1)
    return 0.5 * lp0 + 0.5 * lp1


def prediction_losses(output, target, fg):
    return per_sample_native_score(output, target), reference_error(output[..., 0], fg)


def cross_episode_permutation(episode_ids, seed):
    """A seeded permutation whose source episode differs from every destination episode."""
    episodes = np.asarray(episode_ids)
    if len(set(episodes)) < 2:
        raise ValueError("within-condition shuffle needs at least two episodes")
    rng = np.random.default_rng(seed)
    names, counts = np.unique(episodes, return_counts=True)
    shift = int(counts.max())
    if 2 * shift > len(episodes):
        raise ValueError("one episode holds more than half of the samples; no cross-episode permutation exists")
    # random episode order, random order inside each episode, then rotate by the largest episode size:
    # position i receives the sample at i + shift, which is always in a different episode.
    order = np.concatenate([rng.permutation(np.flatnonzero(episodes == e)) for e in rng.permutation(names)])
    perm = np.empty(len(episodes), dtype=int)
    perm[order] = order[(np.arange(len(order)) + shift) % len(order)]
    return perm


class AttentionCapture:
    """Temporarily force and capture per-head weights from fusion attention modules."""

    def __init__(self, gate):
        self.gate = gate
        self.weights = {}
        self.handles = []

    @staticmethod
    def _request_weights(module, args, kwargs):
        args = list(args)
        if len(args) >= 5:
            args[4] = True
        else:
            kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return tuple(args), kwargs

    def __enter__(self):
        modules = {}
        modules.update({f"encoder_{i}": layer.self_attn for i, layer in enumerate(self.gate.model.encoder.layers)})
        modules.update({f"decoder_{i}": layer.multihead_attn for i, layer in enumerate(self.gate.model.decoder.layers)})
        for name, module in modules.items():
            self.handles.append(module.register_forward_pre_hook(self._request_weights, with_kwargs=True))
            self.handles.append(module.register_forward_hook(
                lambda mod, inp, out, key=name: self.weights.__setitem__(key, out[1].detach() if out[1] is not None else None)
            ))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()

    def masses(self, layout):
        """Per-query camera/LiDAR mass [B,L,2], averaged over heads, for every layer."""
        modality = torch.tensor([VIEW_MODALITY[v] for v in layout])
        out = {}
        for name, weight in self.weights.items():
            if weight is None:
                raise RuntimeError(f"{name} did not return attention weights")
            if weight.ndim == 3:
                weight = weight[:, None]
            mod = modality.to(weight.device)
            mass = torch.stack([weight[..., mod == i].sum(-1) for i in range(2)], -1).mean(1)
            out[name] = mass.cpu()
        return out


@torch.no_grad()
def attention_masses(gate, batch, q_override):
    """Run one forward and return its output plus per-layer per-query modality masses."""
    with AttentionCapture(gate) as capture:
        output = gate(batch, q_override=q_override)
    return output, capture.masses(output["layout"])


def _condition_fields(condition, spec, cfg_corr):
    cam, lid = spec.get("camera"), spec.get("lidar")
    return {
        "condition": condition,
        "kind": evaluate.condition_kind(spec, cfg_corr),
        "camera_family": cam[0] if cam else "",
        "camera_severity": cam[1] if cam else "",
        "lidar_family": lid[0] if lid else "",
        "lidar_severity": lid[1] if lid else "",
    }


def _base_rows(ids, condition, spec, cfg_corr, fg):
    fields = _condition_fields(condition, spec, cfg_corr)
    return [dict(sample_id=sid, episode_id=sid.split("/")[0], fg_present=bool(fg[i].any()), **fields)
            for i, sid in enumerate(ids)]


def _write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"no rows produced for {path.stem}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _loader(root, ids, spec, seed, batch_size):
    global _LOADER_WORKERS
    dataset = evaluate._ConditionSamples(root, ids, spec, seed)
    def make(workers):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                           num_workers=workers, collate_fn=evaluate._collate)
    yielded = False
    try:
        for item in make(_LOADER_WORKERS):
            yielded = True
            yield item
    except PermissionError:
        if yielded:
            raise
        _LOADER_WORKERS = 0
        print("DataLoader workers unavailable; retrying with num_workers=0", flush=True)
        yield from make(0)


def _model_batch(item, device):
    return {key: item[key].to(device) for key in upstream.MODEL_INPUT_KEYS}


def _truth(ids, targets, device):
    target = torch.stack([targets[sid] for sid in ids]).to(device)
    fg = target[..., 0] > 0.01
    return target, fg


def _ap(scores, labels):
    return metrics.grid_fg_ap(np.concatenate(scores), np.concatenate(labels)) if scores else None


@torch.no_grad()
def mean_training_tokens(gate, cfg, split, out_dir, device, batch_size=16):
    cache = out_dir / "mean_tokens_train512.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)["mean"]
    ids = data.split_samples(cfg["paths"]["dataset_root"], split, "train")
    rng = np.random.default_rng(512)
    if len(ids) > 512:
        ids = [ids[i] for i in sorted(rng.choice(len(ids), 512, replace=False))]
    total, count = None, 0
    for item in _loader(cfg["paths"]["dataset_root"], ids, {}, cfg["corruption"]["base_seed"], batch_size):
        batch = _model_batch(item, device)
        features, _ = assemble(tokenize(gate.model, batch))
        value = features.sum(1).cpu()
        total = value if total is None else total + value
        count += features.shape[1]
    mean = total / count
    torch.save({"mean": mean, "n_samples": count}, cache)
    return mean


def _spearman(rows, x, y):
    return float(spearmanr([r[x] for r in rows], [r[y] for r in rows]).correlation)


def _utility_summary(rows, seed):
    summary = {}
    for variant in sorted({r["variant"] for r in rows}):
        vr = [r for r in rows if r["variant"] == variant]
        summary[variant] = {}
        groups = {"overall": vr, "fg_present": [r for r in vr if r["fg_present"]], "fg_absent": [r for r in vr if not r["fg_present"]]}
        groups.update({k: [r for r in vr if r["kind"] == k and r["fg_present"]] for k in sorted({r["kind"] for r in vr})})  # per kind: fg-present only (H1 protocol)
        for group, gr in groups.items():
            summary[variant][group] = {}
            for modality in ("camera", "lidar"):
                mr = [r for r in gr if r["modality"] == modality]
                specs = (("q_vs_neg_E", "q", "neg_E"), ("q_vs_utility", "q", "utility"),
                         ("E_vs_neg_utility", "E", "neg_utility"))
                summary[variant][group][modality] = {
                    name: episode_bootstrap_ci(mr, lambda rs, a=a, b=b: _spearman(rs, a, b), n=2000, seed=seed)
                    for name, a, b in specs
                }
    return summary


@torch.no_grad()
def run_utility(gate, reference, cfg, split, ids, suite, targets, out_dir, device, batch_size, seed):
    token_means = mean_training_tokens(gate, cfg, split, out_dir, device, batch_size)
    rows, labels = [], defaultdict(list)
    scores = defaultdict(lambda: defaultdict(list))
    for condition, spec in suite.items():
        for item in _loader(cfg["paths"]["dataset_root"], ids, spec, cfg["corruption"]["base_seed"], batch_size):
            batch_ids = item["sample_id"]
            batch = _model_batch(item, device)
            target, fg = _truth(batch_ids, targets, device)
            full = gate(batch)
            full_native, full_bse = prediction_losses(full["traffic"], target, fg)
            labels[condition].append(fg.cpu().numpy().ravel())
            scores["full"][condition].append(full["traffic"][..., 0].cpu().numpy().ravel())
            q_star, errors = q_targets(reference, batch, fg, cfg["reliability"]["tau_quality"])
            variants = {}
            for modality in ("camera", "lidar"):
                availability = torch.ones(len(batch_ids), 4, device=device)
                if modality == "camera":
                    availability[:, :3] = 0
                else:
                    availability[:, 3] = 0
                variants[("hardmask", modality)] = gate(batch, availability=availability)
                variants[("meantoken", modality)] = gate(batch, token_means=token_means, replace_modality=modality)
            bases = _base_rows(batch_ids, condition, spec, cfg["corruption"], fg)
            for (kind, modality), output in variants.items():
                native, bse = prediction_losses(output["traffic"], target, fg)
                scores[f"{kind}_{modality}"][condition].append(output["traffic"][..., 0].cpu().numpy().ravel())
                mi = 0 if modality == "camera" else 1
                for i, base in enumerate(bases):
                    utility = float(native[i] - full_native[i])
                    rows.append(dict(base, variant=kind, modality=modality, q=float(full["q"][i, mi]),
                                     q_star=float(q_star[i, mi]), E=float(errors[i, mi]), neg_E=float(-errors[i, mi]),
                                     q_camera=float(full["q"][i, 0]), q_lidar=float(full["q"][i, 1]),
                                     q_star_camera=float(q_star[i, 0]), q_star_lidar=float(q_star[i, 1]),
                                     E_camera=float(errors[i, 0]), E_lidar=float(errors[i, 1]),
                                     full_native_score_loss=float(full_native[i]), native_score_loss=float(native[i]),
                                     full_balanced_squared_error=float(full_bse[i]), balanced_squared_error=float(bse[i]),
                                     utility=utility, neg_utility=-utility))
    _write_csv(out_dir / "utility.csv", rows)
    return {"label": "removal utility", "role": "sensitivity",
            "correlations": _utility_summary(rows, seed), "ap": _ap_summary(scores, labels)}


@torch.no_grad()
def run_partial(gate, reference, cfg, ids, suite, targets, out_dir, device,
                batch_size, seed):
    """Measure local loss changes when one learned quality is scaled at a time."""
    rows = []
    for condition, spec in suite.items():
        for item in _loader(cfg["paths"]["dataset_root"], ids, spec,
                            cfg["corruption"]["base_seed"], batch_size):
            batch_ids, batch = item["sample_id"], _model_batch(item, device)
            target, fg = _truth(batch_ids, targets, device)
            full = gate(batch)
            full_native, full_bse = prediction_losses(full["traffic"], target, fg)
            _, errors = q_targets(reference, batch, fg, cfg["reliability"]["tau_quality"])
            bases = _base_rows(batch_ids, condition, spec, cfg["corruption"], fg)
            for modality_index, modality in enumerate(("camera", "lidar")):
                for factor in PARTIAL_FACTORS:
                    q = full["q"].clone()
                    q[:, modality_index] = (q[:, modality_index] * factor).clamp(gate.eps, 1.0)
                    output = gate(batch, q_override=q)
                    native, bse = prediction_losses(output["traffic"], target, fg)
                    for index, base in enumerate(bases):
                        rows.append(dict(
                            base, modality=modality, factor=factor,
                            q=float(full["q"][index, modality_index]),
                            intervened_q=float(q[index, modality_index]),
                            E=float(errors[index, modality_index]),
                            full_native_score_loss=float(full_native[index]),
                            native_score_loss=float(native[index]),
                            delta_native_score_loss=float(native[index] - full_native[index]),
                            full_balanced_squared_error=float(full_bse[index]),
                            balanced_squared_error=float(bse[index]),
                            delta_balanced_squared_error=float(bse[index] - full_bse[index])))
    _write_csv(out_dir / "partial.csv", rows)
    return {"foreground_by_kind": partial_summary(
        rows, ("native_score_loss", "balanced_squared_error"), seed)}


def _ap_summary(score_store, label_store):
    return {variant: {condition: _ap(scores, label_store[condition]) for condition, scores in by_cond.items()}
            for variant, by_cond in score_store.items()}


@torch.no_grad()
def run_oracle_rel(gate, reference, cfg, ids, suite, targets, out_dir, device, batch_size):
    rows, scores = [], defaultdict(lambda: defaultdict(list))
    labels = defaultdict(list)
    variants = ["absolute"] + (["relative"] if gate._diagnose_method == "PR" else [])
    for condition, spec in suite.items():
        corrupt = _loader(cfg["paths"]["dataset_root"], ids, spec, cfg["corruption"]["base_seed"], batch_size)
        clean = _loader(cfg["paths"]["dataset_root"], ids, {}, cfg["corruption"]["base_seed"], batch_size)
        for item, clean_item in zip(corrupt, clean):
            batch_ids = item["sample_id"]
            batch, clean_batch = _model_batch(item, device), _model_batch(clean_item, device)
            target, fg = _truth(batch_ids, targets, device)
            q_abs, e_abs = q_targets(reference, batch, fg, cfg["reliability"]["tau_quality"])
            qs = {"absolute": (q_abs, e_abs)}
            if "relative" in variants:
                q_rel, e_rel, _, _ = q_targets_relative(reference, batch, clean_batch, fg, cfg["reliability"]["tau_relative"])
                qs["relative"] = (q_rel, e_rel)
            bases = _base_rows(batch_ids, condition, spec, cfg["corruption"], fg)
            labels[condition].append(fg.cpu().numpy().ravel())
            for variant, (q, error) in qs.items():
                output = gate(batch, q_override=q)
                native, bse = prediction_losses(output["traffic"], target, fg)
                scores[variant][condition].append(output["traffic"][..., 0].cpu().numpy().ravel())
                for i, base in enumerate(bases):
                    rows.append(dict(base, variant=variant, q_camera=float(q[i, 0]), q_lidar=float(q[i, 1]),
                                     E_camera=float(error[i, 0]), E_lidar=float(error[i, 1]),
                                     native_score_loss=float(native[i]), balanced_squared_error=float(bse[i])))
    _write_csv(out_dir / "oracle_rel.csv", rows)
    return {"ap": _ap_summary(scores, labels)}


@torch.no_grad()
def run_oracle_gate(gate, cfg, ids, suite, targets, out_dir, device, batch_size,
                    xfit=None, temporal=None):
    candidates = [(a, b) for a in GRID for b in GRID]
    rows = []
    loss_sum, loss_count = Counter(), Counter()
    oracle_ap, neutral_ap = {}, {}
    chosen = Counter()
    learned_loss_violations = 0
    add_temporal = temporal is not None and not temporal.has_data
    for condition, spec in suite.items():
        condition_labels, neutral_scores, oracle_scores = [], [], []
        for item in _loader(cfg["paths"]["dataset_root"], ids, spec, cfg["corruption"]["base_seed"], batch_size):
            batch_ids, batch = item["sample_id"], _model_batch(item, device)
            target, fg = _truth(batch_ids, targets, device)
            condition_labels.append(fg.cpu().numpy().ravel())
            bases = _base_rows(batch_ids, condition, spec, cfg["corruption"], fg)
            batch_outputs, batch_losses = [], []
            row_start = len(rows)
            for q_cam, q_lidar in candidates:
                q = torch.tensor([q_cam, q_lidar], device=device).expand(len(batch_ids), 2)
                output = gate(batch, q_override=q)
                native, bse = prediction_losses(output["traffic"], target, fg)
                batch_outputs.append(output["traffic"][..., 0].cpu().numpy())
                batch_losses.append(native)
                key = f"{q_cam:g},{q_lidar:g}"
                loss_sum[key] += float(native.sum())
                loss_count[key] += len(native)
                if key == "1,1":
                    neutral_scores.append(batch_outputs[-1].ravel())
                for i, base in enumerate(bases):
                    rows.append(dict(base, variant=key, q_camera=q_cam, q_lidar=q_lidar,
                                     native_score_loss=float(native[i]), balanced_squared_error=float(bse[i]), selected=False))
            learned_output = gate(batch)
            learned_native, learned_bse = prediction_losses(learned_output["traffic"], target, fg)
            batch_outputs.append(learned_output["traffic"][..., 0].cpu().numpy())
            batch_losses.append(learned_native)
            for i, base in enumerate(bases):
                rows.append(dict(base, variant="learned_q",
                                 q_camera=float(learned_output["q"][i, 0]),
                                 q_lidar=float(learned_output["q"][i, 1]),
                                 native_score_loss=float(learned_native[i]),
                                 balanced_squared_error=float(learned_bse[i]), selected=False))
            loss_matrix = torch.stack(batch_losses, 1)
            learned = batch_outputs[-1]
            if xfit is not None:
                xfit.add(condition, np.stack(batch_outputs), learned, target[..., 0].cpu().numpy())
            if add_temporal:
                temporal.add(condition, batch_ids, np.stack(batch_outputs),
                             torch.stack(batch_losses).cpu().numpy(), learned,
                             target[..., 0].cpu().numpy())
            best = loss_matrix.argmin(1).cpu().numpy()
            violations = loss_matrix.min(1).values > learned_native + 1e-8
            learned_loss_violations += int(violations.sum())
            assert not violations.any(), "in-sample oracle loss exceeded learned-q loss"
            for i, ci in enumerate(best):
                rows[row_start + int(ci) * len(batch_ids) + i]["selected"] = True
                key = ("learned_q" if ci == len(candidates)
                       else f"{candidates[ci][0]:g},{candidates[ci][1]:g}")
                chosen[(_condition_fields(condition, spec, cfg["corruption"])["kind"], key)] += 1
                oracle_scores.append(batch_outputs[ci][i].ravel())
        oracle_ap[condition] = _ap(oracle_scores, condition_labels)
        neutral_ap[condition] = _ap(neutral_scores, condition_labels)
    _write_csv(out_dir / "oracle_gate.csv", rows)
    best_fixed = min(loss_sum, key=lambda key: loss_sum[key] / loss_count[key])
    best_q = torch.tensor([float(x) for x in best_fixed.split(",")], device=device)
    best_fixed_ap = {}
    add_temporal_fixed = temporal is not None and temporal.has_data and not temporal.fixed_complete
    for condition, spec in suite.items():
        condition_scores, condition_labels = [], []
        for item in _loader(cfg["paths"]["dataset_root"], ids, spec, cfg["corruption"]["base_seed"], batch_size):
            batch_ids, batch = item["sample_id"], _model_batch(item, device)
            _, fg = _truth(batch_ids, targets, device)
            q = best_q.expand(len(batch_ids), 2)
            fixed_scores = gate(batch, q_override=q)["traffic"][..., 0].cpu().numpy()
            condition_scores.append(fixed_scores.ravel())
            condition_labels.append(fg.cpu().numpy().ravel())
            if add_temporal_fixed:
                temporal.add_fixed(condition, fixed_scores)
        best_fixed_ap[condition] = _ap(condition_scores, condition_labels)
    return {
        "oracle_ap": oracle_ap,
        "neutral_ap": neutral_ap,
        "best_fixed_candidate": best_fixed,
        "best_fixed_mean_native_score_loss": loss_sum[best_fixed] / loss_count[best_fixed],
        "best_fixed_ap": best_fixed_ap,
        "in_sample_oracle_learned_loss_violations": learned_loss_violations,
        "chosen_histogram_by_kind": {kind: {key: count for (k, key), count in chosen.items() if k == kind}
                                     for kind in sorted({k for k, _ in chosen})},
    }


@torch.no_grad()
def run_oracle_gate_xfit(gate, cfg, ids, suite, targets, device, batch_size, xfit,
                         temporal=None):
    """Select candidates on one fixed cell half and score only on the other."""
    candidates = [(a, b) for a in GRID for b in GRID]
    add_temporal = temporal is not None and not temporal.has_data
    if not xfit.loss_counts.any():
        for condition, spec in suite.items():
            for item in _loader(cfg["paths"]["dataset_root"], ids, spec,
                                cfg["corruption"]["base_seed"], batch_size):
                batch_ids, batch = item["sample_id"], _model_batch(item, device)
                target, _ = _truth(batch_ids, targets, device)
                outputs, losses = [], []
                for q_camera, q_lidar in candidates:
                    q = torch.tensor([q_camera, q_lidar], device=device).expand(len(batch_ids), 2)
                    output = gate(batch, q_override=q)["traffic"]
                    outputs.append(output[..., 0].cpu().numpy())
                    if temporal is not None:
                        losses.append(per_sample_native_score(output, target))
                learned_output = gate(batch)["traffic"]
                learned = learned_output[..., 0].cpu().numpy()
                outputs.append(learned)
                if temporal is not None:
                    losses.append(per_sample_native_score(learned_output, target))
                xfit.add(condition, np.stack(outputs), learned, target[..., 0].cpu().numpy())
                if add_temporal:
                    temporal.add(condition, batch_ids, np.stack(outputs),
                                 torch.stack(losses).cpu().numpy(), learned,
                                 target[..., 0].cpu().numpy())
    best = xfit.best_indices
    temporal_best = temporal.best_index if temporal is not None and temporal.has_data else None
    add_temporal_fixed = temporal is not None and temporal.has_data and not temporal.fixed_complete
    for condition, spec in suite.items():
        for item in _loader(cfg["paths"]["dataset_root"], ids, spec,
                            cfg["corruption"]["base_seed"], batch_size):
            batch_ids, batch = item["sample_id"], _model_batch(item, device)
            target, _ = _truth(batch_ids, targets, device)
            fixed = {}
            fixed_indices = set(map(int, best))
            if temporal_best is not None:
                fixed_indices.add(temporal_best)
            for index in fixed_indices:
                q = torch.tensor(candidates[index], device=device).expand(len(batch_ids), 2)
                fixed[index] = gate(batch, q_override=q)["traffic"][..., 0].cpu().numpy()
            for fold, index in enumerate(best):
                xfit.add_fixed(condition, fold, fixed[int(index)], target[..., 0].cpu().numpy())
            if add_temporal_fixed:
                temporal.add_fixed(condition, fixed[temporal_best])
    return xfit.summary()


@torch.no_grad()
def run_oracle_gate_temporal(gate, cfg, ids, suite, targets, device, batch_size, temporal):
    """Choose each frame's candidate from the prior evaluated frame in its episode."""
    candidates = [(a, b) for a in GRID for b in GRID]
    if not temporal.has_data:
        for condition, spec in suite.items():
            for item in _loader(cfg["paths"]["dataset_root"], ids, spec,
                                cfg["corruption"]["base_seed"], batch_size):
                batch_ids, batch = item["sample_id"], _model_batch(item, device)
                target, _ = _truth(batch_ids, targets, device)
                outputs, losses = [], []
                for q_camera, q_lidar in candidates:
                    q = torch.tensor([q_camera, q_lidar], device=device).expand(len(batch_ids), 2)
                    output = gate(batch, q_override=q)["traffic"]
                    outputs.append(output[..., 0].cpu().numpy())
                    losses.append(per_sample_native_score(output, target))
                learned_output = gate(batch)["traffic"]
                learned = learned_output[..., 0].cpu().numpy()
                outputs.append(learned)
                losses.append(per_sample_native_score(learned_output, target))
                temporal.add(condition, batch_ids, np.stack(outputs),
                             torch.stack(losses).cpu().numpy(), learned,
                             target[..., 0].cpu().numpy())
    if not temporal.fixed_complete:
        best_q = torch.tensor(candidates[temporal.best_index], device=device)
        for condition, spec in suite.items():
            for item in _loader(cfg["paths"]["dataset_root"], ids, spec,
                                cfg["corruption"]["base_seed"], batch_size):
                batch_ids, batch = item["sample_id"], _model_batch(item, device)
                scores = gate(batch, q_override=best_q.expand(len(batch_ids), 2))[
                    "traffic"][..., 0].cpu().numpy()
                temporal.add_fixed(condition, scores)
    return temporal.summary()


@torch.no_grad()
def run_shuffle(gate, cfg, ids, suite, targets, out_dir, device, batch_size, seed):
    records, condition_data = [], {}
    for condition, spec in suite.items():
        qs, learned_scores = [], []
        for item in _loader(cfg["paths"]["dataset_root"], ids, spec, cfg["corruption"]["base_seed"], batch_size):
            batch_ids, batch = item["sample_id"], _model_batch(item, device)
            target, fg = _truth(batch_ids, targets, device)
            output = gate(batch)
            qs.append(output["q"].cpu())
            learned_scores.append(output["traffic"][..., 0].cpu().numpy())
            for i, sid in enumerate(batch_ids):
                records.append((condition, sid, output["q"][i].cpu()))
        condition_data[condition] = {"spec": spec, "q": torch.cat(qs), "learned": np.concatenate(learned_scores)}
    rng = np.random.default_rng(seed)
    global_perm = rng.permutation(len(records))
    global_source = {records[i][:2]: records[j] for i, j in enumerate(global_perm)}
    rows, labels = [], defaultdict(list)
    scores = defaultdict(lambda: defaultdict(list))
    offset = defaultdict(int)
    for condition, payload in condition_data.items():
        eps = [sid.split("/")[0] for sid in ids]
        within = cross_episode_permutation(eps, seed + sum(condition.encode("utf-8")))
        for item in _loader(cfg["paths"]["dataset_root"], ids, payload["spec"], cfg["corruption"]["base_seed"], batch_size):
            batch_ids, batch = item["sample_id"], _model_batch(item, device)
            target, fg = _truth(batch_ids, targets, device)
            start = offset[condition]
            stop = start + len(batch_ids)
            learned_q = payload["q"][start:stop].to(device)
            global_q = torch.stack([global_source[(condition, sid)][2] for sid in batch_ids]).to(device)
            within_q = payload["q"][within[start:stop]].to(device)
            variants = {"learned": (learned_q, payload["learned"][start:stop]), "global": (global_q, None), "within_condition": (within_q, None)}
            labels[condition].append(fg.cpu().numpy().ravel())
            bases = _base_rows(batch_ids, condition, payload["spec"], cfg["corruption"], fg)
            for variant, (q, known_score) in variants.items():
                output = gate(batch, q_override=q)
                native, bse = prediction_losses(output["traffic"], target, fg)
                score = output["traffic"][..., 0].cpu().numpy() if known_score is None else known_score
                scores[variant][condition].append(score.ravel())
                for i, base in enumerate(bases):
                    if variant == "global":
                        source_sid = global_source[(condition, batch_ids[i])][1]
                    elif variant == "within_condition":
                        source_sid = ids[within[start + i]]
                    else:
                        source_sid = batch_ids[i]
                    rows.append(dict(base, variant=variant, q_camera=float(q[i, 0]), q_lidar=float(q[i, 1]),
                                     source_sample_id=source_sid, source_episode_id=source_sid.split("/")[0],
                                     native_score_loss=float(native[i]), balanced_squared_error=float(bse[i])))
            offset[condition] = stop
    _write_csv(out_dir / "shuffle.csv", rows)
    return {"ap": _ap_summary(scores, labels)}


@torch.no_grad()
def run_trace(gate, cfg, ids, suite, targets, out_dir, device, batch_size, seed):
    trace_conditions = ["clean", "camera_gaussian_blur_s5", "lidar_point_dropout_s5"]
    missing = [c for c in trace_conditions if c not in suite]
    if missing:
        raise RuntimeError(f"trace conditions missing from suite: {missing}")
    pool = [(condition, sid) for condition in trace_conditions for sid in ids]
    rng = np.random.default_rng(seed)
    selected = [pool[i] for i in sorted(rng.choice(len(pool), min(64, len(pool)), replace=False))]
    chosen_ids = defaultdict(list)
    for condition, sid in selected:
        chosen_ids[condition].append(sid)
    rows, labels = [], defaultdict(list)
    scores = defaultdict(lambda: defaultdict(list))
    for condition, selected_ids in chosen_ids.items():
        spec = suite[condition]
        for item in _loader(cfg["paths"]["dataset_root"], selected_ids, spec, cfg["corruption"]["base_seed"], batch_size):
            batch_ids, batch = item["sample_id"], _model_batch(item, device)
            target, fg = _truth(batch_ids, targets, device)
            neutral_q = torch.ones(len(batch_ids), 2, device=device)
            neutral = gate(batch, q_override=neutral_q)["traffic"]
            bases = _base_rows(batch_ids, condition, spec, cfg["corruption"], fg)
            labels[condition].append(fg.cpu().numpy().ravel())
            for modality in ("camera", "lidar"):
                for value in TRACE_GRID:
                    q = torch.ones(len(batch_ids), 2, device=device)
                    q[:, 0 if modality == "camera" else 1] = value
                    output, masses = attention_masses(gate, batch, q)
                    native, bse = prediction_losses(output["traffic"], target, fg)
                    delta = (output["traffic"][..., 0] - neutral[..., 0]).abs().mean(1)
                    variant = f"{modality}_{value:g}"
                    scores[variant][condition].append(output["traffic"][..., 0].cpu().numpy().ravel())
                    enc = [masses[f"encoder_{i}"].mean(1) for i in range(len(gate.model.encoder.layers))]
                    dec = [masses[f"decoder_{i}"][:, :400].mean(1) for i in range(len(gate.model.decoder.layers))]
                    for i, base in enumerate(bases):
                        rows.append(dict(base, variant=variant, q_camera=float(q[i, 0]), q_lidar=float(q[i, 1]),
                                         native_score_loss=float(native[i]), balanced_squared_error=float(bse[i]),
                                         mean_abs_delta_score=float(delta[i]),
                                         encoder_camera_mass=json.dumps([float(x[i, 0]) for x in enc]),
                                         encoder_lidar_mass=json.dumps([float(x[i, 1]) for x in enc]),
                                         decoder_camera_mass=json.dumps([float(x[i, 0]) for x in dec]),
                                         decoder_lidar_mass=json.dumps([float(x[i, 1]) for x in dec])))
    _write_csv(out_dir / "trace.csv", rows)
    return {"n_fixed_samples": len(selected), "conditions": {k: len(v) for k, v in chosen_ids.items()},
            "ap": _ap_summary(scores, labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/extended.yaml")
    ap.add_argument("--method", default="P")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--suite", default="reduced", choices=["reduced", "full"])
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--analyses", default="utility,oracle_rel,oracle_gate,shuffle,trace")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if args.frame_stride < 1:
        ap.error("--frame-stride must be >= 1")
    analyses = [x.strip() for x in args.analyses.split(",") if x.strip()]
    allowed = {"utility", "partial", "oracle_rel", "oracle_gate", "oracle_gate_xfit",
               "oracle_gate_temporal", "shuffle", "trace"}
    unknown = set(analyses) - allowed
    if unknown:
        ap.error(f"unknown analyses: {sorted(unknown)}")
    if "oracle_gate_temporal" in analyses:
        analyses = [name for name in analyses if name != "oracle_gate_temporal"] + ["oracle_gate_temporal"]
    gated_only = {"partial", "oracle_rel", "oracle_gate", "oracle_gate_xfit",
                  "oracle_gate_temporal", "shuffle", "trace"}
    diagnostic_gated = set(GATED_METHODS) | {"B3", "B3A", "B4"}
    if set(analyses) & gated_only and args.method not in diagnostic_gated:
        ap.error(f"{sorted(set(analyses) & gated_only)} require a gated method, got {args.method}")

    started = time.time()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_path = Path(cfg["paths"]["artifacts_root"]) / "manifests" / "split.json"
    split = json.loads(manifest_path.read_text())
    ids = data.split_samples(cfg["paths"]["dataset_root"], split, args.split)[::args.frame_stride]
    suite = reduced_suite(cfg["corruption"]) if args.suite == "reduced" else evaluate.full_suite(cfg["corruption"])
    gate, provenance = load_method(cfg, args.method, args.seed, device)
    gate._diagnose_method = args.method
    gate.eval()
    reference = load_reference(cfg, device) if set(analyses) & {"utility", "partial", "oracle_rel"} else None
    out_dir = Path(cfg["paths"]["artifacts_root"]) / "diagnostics" / f"{args.method}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {sid: torch.from_numpy(data.load_gt(cfg["paths"]["dataset_root"], sid)) for sid in ids}
    candidates = [f"{a:g},{b:g}" for a in GRID for b in GRID] + ["learned_q"]
    xfit = (GateXfitAccumulator(400, candidates, candidates.index("1,1"), args.seed, "interfuser")
            if "oracle_gate_xfit" in analyses else None)
    temporal = (GateTemporalAccumulator(candidates, candidates.index("1,1"), "interfuser")
                if "oracle_gate_temporal" in analyses else None)

    runners = {
        "utility": lambda: run_utility(gate, reference, cfg, split, ids, suite, targets, out_dir, device, args.batch_size, args.seed),
        "partial": lambda: run_partial(gate, reference, cfg, ids, suite, targets, out_dir,
                                        device, args.batch_size, args.seed),
        "oracle_rel": lambda: run_oracle_rel(gate, reference, cfg, ids, suite, targets, out_dir, device, args.batch_size),
        "oracle_gate": lambda: run_oracle_gate(gate, cfg, ids, suite, targets, out_dir, device,
                                                args.batch_size, xfit, temporal),
        "oracle_gate_xfit": lambda: run_oracle_gate_xfit(
            gate, cfg, ids, suite, targets, device, args.batch_size, xfit, temporal),
        "oracle_gate_temporal": lambda: run_oracle_gate_temporal(
            gate, cfg, ids, suite, targets, device, args.batch_size, temporal),
        "shuffle": lambda: run_shuffle(gate, cfg, ids, suite, targets, out_dir, device, args.batch_size, args.seed),
        "trace": lambda: run_trace(gate, cfg, ids, suite, targets, out_dir, device, args.batch_size, args.seed),
    }
    summary = {"method": args.method, "seed": args.seed, "split": args.split, "suite": args.suite,
               "frame_stride": args.frame_stride, "n_samples": len(ids), "n_conditions": len(suite),
               "device": device, "provenance": provenance, "analyses": {}}
    prev = out_dir / "summary.json"  # keep analyses from earlier runs of the same checkpoint
    if prev.exists():
        summary["analyses"] = {**json.loads(prev.read_text()).get("analyses", {}), **summary.get("analyses", {})}
    for analysis in analyses:
        t0 = time.time()
        print(f"running {analysis}...", flush=True)
        summary["analyses"][analysis] = runners[analysis]()
        summary["analyses"][analysis]["wall_seconds"] = round(time.time() - t0, 1)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    summary["wall_seconds"] = round(time.time() - started, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
