"""TransFuser reliability training and CenterNet fusion fine-tuning.

The detection objective follows ``team_code_transfuser/model.py``: Gaussian focal
centre heatmap, 0.1-weighted width/height L1, offset L1, yaw class cross entropy,
yaw residual Smooth-L1, and velocity L1.  Brake is excluded because the shared
collector has no brake ground truth (``tf_targets`` therefore stores a placeholder).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rrf import data
from rrf.config import load_config
from rrf.gated import ReliabilityEstimator, StaticGate
from rrf.tf_corruptions import condition_name, corrupt_tf_sample, corruption_seed
from rrf.tf_evaluate import run_suite
from rrf.tf_gated import GatedTransFuser, RAW_STAT_DIM
from rrf.tf_probes import q_targets, q_targets_relative, render_tf_heatmap
from rrf.transfuser_io import VARIANTS, load_raw_tf, load_tf_model, preprocess_tf, tf_targets


METHODS = ("B2", "BQ", "B3", "B4", "P", "PG", "PR", "PS", "PGS", "PT", "PGT")
GATED_METHODS = ("P", "BQ", "PG", "PR", "PS", "PGS", "PT", "PGT")
GATE_ONLY = ("PG", "PGS", "PGT")
MECHANISM = {"PS": "token_scaling", "PGS": "token_scaling",
             "PT": "key_scaling", "PGT": "key_scaling"}

# Exact released-model parameter prefixes allowed to update. Adaptive pooling and
# bilinear upsampling around GPT have no parameters; the listed projections/FPN
# layers carry fused LiDAR features into the CenterNet detection head.
TF_TRAINABLE_PREFIXES = (
    "_model.transformer1", "_model.transformer2", "_model.transformer3",
    "_model.transformer4", "_model.change_channel_conv_image",
    "_model.change_channel_conv_lidar", "_model.up_conv5", "_model.up_conv4",
    "_model.up_conv3", "_model.c5_conv", "head",
)

TF_INPUT_KEYS = ("rgb", "lidar", "target_point", "target_point_image",
                 "velocity", "bev_points", "cam_points")


# ------------------------------------------------------------------ CenterNet targets/loss
def encode_detection_targets(head, boxes, prediction_shape, config):
    """Encode ``tf_targets`` boxes exactly in the released head's 64x64 layout."""
    batch, _, height, width = prediction_shape
    device = boxes[0].device
    dtype = boxes[0].dtype
    target = {
        "center_heatmap": torch.zeros((batch, 1, height, width), device=device, dtype=dtype),
        "box_wh": torch.zeros((batch, 2, height, width), device=device, dtype=dtype),
        "box_offset": torch.zeros((batch, 2, height, width), device=device, dtype=dtype),
        "yaw_class": torch.zeros((batch, height, width), device=device, dtype=torch.long),
        "yaw_residual": torch.zeros((batch, 1, height, width), device=device, dtype=dtype),
        "box_velocity": torch.zeros((batch, 1, height, width), device=device, dtype=dtype),
        "weight": torch.zeros((batch, 1, height, width), device=device, dtype=dtype),
    }
    x_ratio = width / float(config.lidar_resolution_width)
    y_ratio = height / float(config.lidar_resolution_height)
    for batch_id, rows in enumerate(boxes):
        target["center_heatmap"][batch_id, 0] = render_tf_heatmap(
            rows, config, (height, width), device=device)
        for row in rows:
            x, y = row[0] * x_ratio, row[1] * y_ratio
            xi, yi = int(x), int(y)
            if not (0 <= xi < width and 0 <= yi < height):
                continue
            target["box_wh"][batch_id, :, yi, xi] = torch.stack(
                (row[2] * x_ratio, row[3] * y_ratio))
            target["box_offset"][batch_id, :, yi, xi] = torch.stack((x - xi, y - yi))
            yaw_class, yaw_residual = head.angle2class(row[4])
            target["yaw_class"][batch_id, yi, xi] = yaw_class
            target["yaw_residual"][batch_id, 0, yi, xi] = yaw_residual
            target["box_velocity"][batch_id, 0, yi, xi] = row[5]
            target["weight"][batch_id, 0, yi, xi] = 1.0
    return target


def native_detection_loss(output, boxes, head, config):
    """TransFuser CenterNet loss terms, excluding its unavailable brake target."""
    target = encode_detection_targets(head, boxes, output["center_heatmap"].shape, config)
    heatmap = output["center_heatmap"].clamp(1e-12, 1.0 - 1e-12)
    heat_target = target["center_heatmap"]
    positive = heat_target.eq(1).to(heatmap.dtype)
    negative = heat_target.lt(1).to(heatmap.dtype)
    negative_weight = (1 - heat_target).pow(4)
    avg_factor = positive.sum().clamp_min(1.0)
    heat_loss = (-(heatmap.log()) * (1 - heatmap).pow(2) * positive
                 -(1 - heatmap).log() * heatmap.pow(2) * negative_weight * negative).sum() / avg_factor
    weight = target["weight"]
    weight2 = weight.expand(-1, 2, -1, -1)
    wh_loss = ((output["box_wh"] - target["box_wh"]).abs() * weight2).sum() / (avg_factor * 2) * 0.1
    offset_loss = ((output["box_offset"] - target["box_offset"]).abs() * weight2).sum() / (avg_factor * 2)
    yaw_class_loss = (F.cross_entropy(output["yaw_class"], target["yaw_class"], reduction="none")
                      * weight[:, 0]).sum() / avg_factor
    yaw_residual_loss = (F.smooth_l1_loss(output["yaw_residual"], target["yaw_residual"],
                                          reduction="none") * weight).sum() / avg_factor
    velocity_loss = ((output["box_velocity"] - target["box_velocity"]).abs() * weight).sum() / avg_factor
    parts = {
        "center_heatmap": heat_loss, "box_wh": wh_loss, "box_offset": offset_loss,
        "yaw_class": yaw_class_loss, "yaw_residual": yaw_residual_loss,
        "velocity": velocity_loss,
    }
    return sum(parts.values()), parts


# ------------------------------------------------------------------ data/schedule
def scheduled_condition(cfg_corr, experiment_seed, sample_id, epoch):
    """Draw the shared InterFuser/TransFuser condition for one sample and epoch."""
    from rrf.train import sample_condition

    rng = np.random.default_rng(corruption_seed(
        experiment_seed, sample_id, "schedule", epoch, 0))
    return sample_condition(rng, cfg_corr)


class TrainSamples(Dataset):
    """Raw dual-rig sample, shared scheduled corruption, TF inputs and targets."""

    def __init__(self, root, sample_ids, cfg_corr, experiment_seed, config,
                 epoch=0, with_clean=False):
        self.root, self.ids, self.cfg_corr = root, list(sample_ids), cfg_corr
        self.seed, self.config, self.epoch = experiment_seed, config, epoch
        self.with_clean = with_clean

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        spec = scheduled_condition(self.cfg_corr, self.seed, sample_id, self.epoch)
        raw = load_raw_tf(self.root, sample_id)
        corrupted, _ = corrupt_tf_sample(raw, spec, self.seed, repeat=self.epoch)
        item = preprocess_tf(corrupted, self.config)
        if self.with_clean:
            for key, value in preprocess_tf(raw, self.config).items():
                item["clean_" + key] = value
        boxes = torch.from_numpy(tf_targets(self.root, sample_id, self.config))
        item.update(target_boxes=boxes,
                    target_heatmap=render_tf_heatmap(boxes, self.config),
                    sample_id=sample_id, condition=condition_name(spec))
        return item


def collate(items):
    keys = [key for key in TF_INPUT_KEYS if key in items[0]]
    keys += [key for key in items[0] if key.startswith("clean_")]
    batch = {key: torch.stack([item[key] for item in items]) for key in keys}
    batch["target_heatmap"] = torch.stack([item["target_heatmap"] for item in items])
    batch["target_boxes"] = [item["target_boxes"] for item in items]
    batch["sample_id"] = [item["sample_id"] for item in items]
    batch["condition"] = [item["condition"] for item in items]
    return batch


def to_device(batch, device):
    return {key: ([item.to(device) for item in value]
                  if key == "target_boxes" else value.to(device)
                  if torch.is_tensor(value) else value)
            for key, value in batch.items()}


def infinite_loader(make_dataset, batch_size, workers, seed):
    epoch = 0
    while True:
        dataset = make_dataset(epoch)
        if not len(dataset):
            raise RuntimeError("training split contains no TransFuser samples")
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=workers, collate_fn=collate,
                            generator=generator, drop_last=len(dataset) >= batch_size,
                            persistent_workers=False)
        yield from loader
        epoch += 1


# ------------------------------------------------------------------ helpers
def parse_stages(value):
    try:
        stages = tuple(sorted({int(item) for item in value.split(",") if item != ""}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--stages must be comma-separated integers 0..3") from exc
    if not stages or not set(stages) <= {0, 1, 2, 3}:
        raise argparse.ArgumentTypeError("--stages must be a non-empty subset of 0,1,2,3")
    return stages


def stage_suffix(stages):
    return "" if tuple(stages) == (0, 1, 2, 3) else "_st" + "".join(map(str, stages))


def rel_ckpt_name(seed, target):
    return f"reliability_seed{seed}.pt" if target == "absolute" else f"reliability_rel_seed{seed}.pt"


def fusion_name(method, seed, stages):
    return f"fusion_{method}{stage_suffix(stages)}_seed{seed}"


def checkpoint_dir(cfg, variant):
    path = Path(cfg["paths"]["artifacts_root"]) / "tf_checkpoints" / variant
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(cfg, variant, name):
    path = Path(cfg["paths"]["artifacts_root"]) / "tf_runs" / variant / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_split(cfg, split_file=None):
    path = Path(split_file) if split_file else Path(cfg["paths"]["artifacts_root"]) / "manifests" / "split.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))


def split_samples(root, split, name, subgroup=None, max_frames=None, seed=0):
    """Read normal episode splits plus sample-id leaves used by isolated smoke splits."""
    leaves = split[name][subgroup] if subgroup else sum(split[name].values(), [])
    if leaves and all("/" in value for value in leaves):
        ids = sorted(set(leaves))
        if max_frames and len(ids) > max_frames:
            ids = list(np.random.default_rng(seed).choice(ids, max_frames, replace=False))
        return ids
    return data.split_samples(root, split, name, subgroup, max_frames, seed)


def log_line(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def stat_dim(model):
    channels = model._model.image_encoder.features.feature_info[1]["num_chs"]
    return 2 * channels + 2 + RAW_STAT_DIM


def set_trainable(model, prefixes=TF_TRAINABLE_PREFIXES):
    trainable = total = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(prefixes)
        total += parameter.numel()
        trainable += parameter.numel() if parameter.requires_grad else 0
    return trainable, total


def train_mode(model, prefixes=TF_TRAINABLE_PREFIXES):
    """Train selected fusion/detection modules; keep every frozen module in eval mode."""
    model.eval()
    for name, module in model.named_modules():
        if name.startswith(prefixes):
            module.train()
    return model


def clean_batch(batch):
    return {key: batch["clean_" + key] for key in TF_INPUT_KEYS
            if "clean_" + key in batch}


def quality_targets(reference, batch, cfg, target):
    if target == "relative":
        q, relative, _, _ = q_targets_relative(
            reference, batch, clean_batch(batch), batch["target_heatmap"],
            cfg["reliability"]["tau_relative"])
        return q, relative
    return q_targets(reference, batch, batch["target_heatmap"],
                     cfg["reliability"]["tau_quality"])


def load_reference(variant, seed_file, device):
    model, _ = load_tf_model(variant, seed_file, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return GatedTransFuser(model).to(device).eval()


def build_method(cfg, variant, method, seed, stages, device, seed_file=None,
                 load_reliability=True):
    model, config = load_tf_model(variant, seed_file, device)
    estimator = reference = None
    dimension = stat_dim(model)
    if method in GATED_METHODS:
        estimator = ReliabilityEstimator(dimension,
            cfg["reliability"]["estimator_hidden_dim"],
            cfg["reliability"]["estimator_hidden_layers"]).to(device)
        if load_reliability:
            target = "relative" if method == "PR" else "absolute"
            checkpoint = torch.load(checkpoint_dir(cfg, variant) / rel_ckpt_name(seed, target),
                                    map_location=device, weights_only=False)
            estimator.load_state_dict(checkpoint["estimator"])
            reference = load_reference(variant, seed_file, device)
    elif method == "B3":
        estimator = ReliabilityEstimator(dimension,
            cfg["reliability"]["estimator_hidden_dim"],
            cfg["reliability"]["estimator_hidden_layers"]).to(device)
        if load_reliability:
            checkpoint = torch.load(checkpoint_dir(cfg, variant) / rel_ckpt_name(seed, "absolute"),
                                    map_location=device, weights_only=False)
            estimator.mu.copy_(checkpoint["estimator"]["mu"])
            estimator.sigma.copy_(checkpoint["estimator"]["sigma"])
    elif method == "B4":
        estimator = StaticGate().to(device)
    gate = GatedTransFuser(model, estimator, MECHANISM.get(method, "log_prior"),
                           stages, cfg["fusion"]["alpha"], cfg["fusion"]["epsilon"],
                           gate_enabled_by_default=method != "BQ").to(device)
    return gate, reference, config


def gate_state(gate):
    return ({"key_gamma": gate.key_gamma.detach().cpu().clone()}
            if gate.key_gamma is not None else None)


def load_gate_state(gate, state):
    if state and gate.key_gamma is not None:
        gate.key_gamma.data.copy_(state["key_gamma"].to(gate.key_gamma.device))


def load_trained_method(cfg, variant, method, seed, stages, device, seed_file=None):
    gate, _, config = build_method(cfg, variant, method, seed, stages, device,
                                   seed_file, load_reliability=False)
    path = checkpoint_dir(cfg, variant) / f"{fusion_name(method, seed, stages)}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    gate.model.load_state_dict(checkpoint["model"])
    if gate.estimator is not None and checkpoint.get("estimator") is not None:
        gate.estimator.load_state_dict(checkpoint["estimator"])
    load_gate_state(gate, checkpoint.get("gate"))
    return gate.eval(), config


def _checkpoint(gate, method, seed, stages, step, summary, cfg, clone=False):
    def state(module):
        values = module.state_dict()
        return ({key: value.detach().cpu().clone() for key, value in values.items()}
                if clone else values)

    return {
        "model": state(gate.model),
        "estimator": state(gate.estimator) if gate.estimator else None,
        "gate": gate_state(gate),
        "meta": {"method": method, "seed": seed, "stages": list(stages), "step": step,
                 "eval": summary, "config_sha256": cfg["_meta"]["config_sha256"]},
    }


# ------------------------------------------------------------------ reliability
def stage_reliability(cfg, args, device):
    split = load_split(cfg, args.split_file)
    root, rc = cfg["paths"]["dataset_root"], cfg["reliability"]
    torch.manual_seed(args.seed)
    model, config = load_tf_model(args.variant, args.seed_file, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    gate = GatedTransFuser(model).to(device).eval()
    reference = load_reference(args.variant, args.seed_file, device)
    train_ids = split_samples(root, split, "train")
    val_ids = split_samples(root, split, "validation", "model_select_groups",
                            max_frames=300, seed=1)
    estimator = ReliabilityEstimator(stat_dim(model), rc["estimator_hidden_dim"],
                                     rc["estimator_hidden_layers"]).to(device)
    normalizer_data = TrainSamples(root, train_ids, cfg["corruption"], args.seed,
                                   config, with_clean=args.target == "relative")
    normalizer_loader = DataLoader(normalizer_data, batch_size=16, shuffle=False,
                                   num_workers=cfg["training"]["workers"], collate_fn=collate)
    statistics = []
    with torch.no_grad():
        for index, batch in enumerate(normalizer_loader):
            batch = to_device(batch, device)
            statistics.append(gate(batch, gate_enabled=False)["stats"].cpu())
            if index * 16 >= 800:
                break
    estimator.fit_normalizer(torch.cat(statistics).to(device))
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=rc["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    loader = infinite_loader(lambda epoch: TrainSamples(
        root, train_ids, cfg["corruption"], args.seed, config, epoch,
        with_clean=args.target == "relative"), rc["batch_size"],
        cfg["training"]["workers"], args.seed)
    directory = run_dir(cfg, args.variant, rel_ckpt_name(args.seed, args.target)[:-3])
    log, best, started = directory / "log.jsonl", math.inf, time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, rc["steps"] + 1):
        batch = to_device(next(loader), device)
        estimator.train()
        with torch.no_grad():
            stats = gate(batch, gate_enabled=False)["stats"]
            q_star, _ = quality_targets(reference, batch, cfg, args.target)
        availability = torch.ones((len(batch["sample_id"]), 4), device=device)
        q_hat = estimator(stats, availability)
        foreground = batch["target_heatmap"].flatten(1).ge(0.99).any(1)
        if not foreground.any():
            continue
        loss = F.smooth_l1_loss(q_hat[foreground], q_star[foreground].detach())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(estimator.parameters(),
                                       cfg["training"]["gradient_clip_norm"])
        optimizer.step()
        if step % 50 == 0:
            record = {"step": step, "loss": float(loss),
                      "q_hat_mean": q_hat.mean(0).tolist(),
                      "q_star_mean": q_star.mean(0).tolist(),
                      "elapsed_s": round(time.time() - started)}
            log_line(log, record)
            print(record, flush=True)
        if step % cfg["training"]["evaluation_interval_steps"] == 0 or step == rc["steps"]:
            evaluation = eval_reliability(gate, reference, estimator, root, val_ids,
                                          cfg, args.seed, device, config, args.target)
            evaluation["step"] = step
            log_line(log, {"eval": evaluation})
            criterion = evaluation["foreground_present"].get(
                "mae", evaluation["overall"].get("mae", math.inf))
            if criterion < best:
                best = criterion
                torch.save({"estimator": estimator.state_dict(),
                            "meta": {"step": step, "eval": evaluation,
                                     "seed": args.seed, "target": args.target,
                                     "config_sha256": cfg["_meta"]["config_sha256"]}},
                           checkpoint_dir(cfg, args.variant) /
                           rel_ckpt_name(args.seed, args.target))
    elapsed = time.time() - started
    final = {"variant": args.variant, "target": args.target, "seed": args.seed,
             "steps": rc["steps"], "batch_size": rc["batch_size"],
             "wall_seconds": round(elapsed, 1),
             "seconds_per_step": round(elapsed / rc["steps"], 3),
             "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                                 if device == "cuda" else None),
             "best_validation_mae": best}
    (directory / "final.json").write_text(json.dumps(final, indent=2))
    print(final, flush=True)


@torch.no_grad()
def eval_reliability(gate, reference, estimator, root, val_ids, cfg, seed,
                     device, config, target="absolute"):
    from scipy.stats import spearmanr

    estimator.eval()
    dataset = TrainSamples(root, val_ids, cfg["corruption"], seed + 999, config,
                           with_clean=target == "relative")
    loader = DataLoader(dataset, batch_size=16, shuffle=False,
                        num_workers=cfg["training"]["workers"], collate_fn=collate)
    predicted, desired, errors, foreground, conditions = [], [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        stats = gate(batch, gate_enabled=False)["stats"]
        availability = torch.ones((len(batch["sample_id"]), 4), device=device)
        predicted.append(estimator(stats, availability).cpu())
        q_star, error = quality_targets(reference, batch, cfg, target)
        desired.append(q_star.cpu())
        errors.append(error.cpu())
        foreground.append(batch["target_heatmap"].flatten(1).ge(0.99).any(1).cpu())
        conditions += batch["condition"]
    predicted = torch.cat(predicted).numpy()
    desired = torch.cat(desired).numpy()
    errors = torch.cat(errors).numpy()
    foreground = torch.cat(foreground).numpy()

    def block(mask):
        if mask.sum() < 3:
            return {"n": int(mask.sum())}
        gap = np.abs(desired[mask, 0] - desired[mask, 1]) >= 0.05
        ranks = ((predicted[mask, 0] > predicted[mask, 1]) ==
                 (desired[mask, 0] > desired[mask, 1]))[gap]
        return {
            "n": int(mask.sum()), "mae": float(np.abs(predicted[mask] - desired[mask]).mean()),
            "mae_camera": float(np.abs(predicted[mask, 0] - desired[mask, 0]).mean()),
            "mae_lidar": float(np.abs(predicted[mask, 1] - desired[mask, 1]).mean()),
            "spearman_camera": float(spearmanr(predicted[mask, 0], -errors[mask, 0]).correlation),
            "spearman_lidar": float(spearmanr(predicted[mask, 1], -errors[mask, 1]).correlation),
            "ranking_accuracy": float(ranks.mean()) if len(ranks) else None,
            "ranking_pairs": int(len(ranks)),
            "q_hat_mean": predicted[mask].mean(0).tolist(),
            "q_star_mean": desired[mask].mean(0).tolist(),
            "q_hat_std": predicted[mask].std(0).tolist(),
        }

    result = {"overall": block(np.ones(len(predicted), bool)),
              "foreground_present": block(foreground),
              "foreground_absent": block(~foreground)}
    result["by_condition"] = {
        condition: {"n": int((mask := np.array([item == condition for item in conditions])).sum()),
                    "q_hat_mean": predicted[mask].mean(0).tolist(),
                    "q_star_mean": desired[mask].mean(0).tolist()}
        for condition in sorted(set(conditions))}
    return result


# ------------------------------------------------------------------ fusion
VALIDATION_FAMILIES = {"camera": ("gaussian_noise", "gaussian_blur", "darkening"),
                       "lidar": ("point_dropout", "xyz_jitter")}


def validation_suite():
    suite = {"clean": {}}
    for modality, families in VALIDATION_FAMILIES.items():
        for family in families:
            suite[f"{modality}_{family}_s2"] = {modality: (family, 2)}
    return suite


@torch.no_grad()
def eval_fusion(gate, root, val_ids, cfg, seed, device, config, output_dir,
                gate_enabled=None):
    gate.eval()
    old_default = gate.gate_enabled_by_default
    if gate_enabled is not None:
        gate.gate_enabled_by_default = gate_enabled
    try:
        results = run_suite(gate, config, root, val_ids, validation_suite(), seed + 777,
                            output_dir, batch_size=16, device=device)
    finally:
        gate.gate_enabled_by_default = old_default
    aps = {condition: result["heatmap_peak_ap"] for condition, result in results.items()}
    corrupt = [value for condition, value in aps.items()
               if condition != "clean" and value is not None]
    return {"summary": {
        "clean_ap": aps["clean"],
        "corruption_macro_ap": float(np.mean(corrupt)) if corrupt else None,
        "per_condition_ap": aps,
    }}


def stage_fusion(cfg, args, device):
    split = load_split(cfg, args.split_file)
    root, training, method = cfg["paths"]["dataset_root"], cfg["training"], args.method
    torch.manual_seed(args.seed)
    gate, reference, config = build_method(cfg, args.variant, method, args.seed,
                                           args.stages, device, args.seed_file)
    prefixes = () if method in GATE_ONLY else TF_TRAINABLE_PREFIXES
    trainable, total = set_trainable(gate.model, prefixes)
    groups = []
    model_parameters = [parameter for parameter in gate.model.parameters() if parameter.requires_grad]
    if model_parameters:
        groups.append({"params": model_parameters, "lr": training["learning_rate"]})
    if gate.estimator is not None:
        groups.append({"params": list(gate.estimator.parameters()),
                       "lr": cfg["reliability"]["learning_rate"]})
    if gate.key_gamma is not None:
        groups.append({"params": [gate.key_gamma],
                       "lr": cfg["reliability"]["learning_rate"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=training["weight_decay"])
    train_ids = split_samples(root, split, "train")
    val_ids = split_samples(root, split, "validation", "model_select_groups",
                            max_frames=200, seed=1)
    target_kind = "relative" if method == "PR" else "absolute"
    loader = infinite_loader(lambda epoch: TrainSamples(
        root, train_ids, cfg["corruption"], args.seed, config, epoch,
        with_clean=method == "PR"), training["batch_size"], training["workers"], args.seed)
    name = fusion_name(method, args.seed, args.stages)
    directory = run_dir(cfg, args.variant, name)
    log = directory / "log.jsonl"
    setup = {
        "variant": args.variant, "method": method, "seed": args.seed,
        "stages": list(args.stages), "trainable_prefixes": list(prefixes),
        "trainable_params": trainable, "total_params": total,
        "estimator_params": sum(p.numel() for p in gate.estimator.parameters()) if gate.estimator else 0,
        "effective_batch": training["batch_size"] * training["gradient_accumulation_steps"],
        "optimizer_updates": training["fusion_steps"],
        "detection_loss_terms": ["center_heatmap_focal", "box_wh_l1_x0.1",
                                 "box_offset_l1", "yaw_class_cross_entropy",
                                 "yaw_residual_smooth_l1", "velocity_l1"],
        "excluded_loss": "brake: shared dataset has no brake ground truth",
        "config_sha256": cfg["_meta"]["config_sha256"],
    }
    (directory / "setup.json").write_text(json.dumps(setup, indent=2))
    baseline = eval_fusion(gate, root, val_ids, cfg, args.seed, device, config,
                           directory / "_val", gate_enabled=False)
    log_line(log, {"eval": {"step": 0, **baseline}})
    clean_floor = baseline["summary"]["clean_ap"] - cfg["metrics"]["max_clean_ap_drop"]
    best, best_record = -math.inf, None
    fallback = {"clean_ap": -math.inf, "checkpoint": None, "record": None}
    accumulation = training["gradient_accumulation_steps"]
    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, training["fusion_steps"] + 1):
        train_mode(gate.model, prefixes)
        if gate.estimator is not None:
            gate.estimator.train()
        record = {"step": step, "loss": 0.0, "task": 0.0, "quality": 0.0}
        optimizer.zero_grad()
        for _ in range(accumulation):
            batch = to_device(next(loader), device)
            output = gate(batch)
            task, parts = native_detection_loss(output, batch["target_boxes"],
                                                gate.model.head, config)
            loss = task
            if method in GATED_METHODS and training["lambda_quality"] > 0:
                q_star, _ = quality_targets(reference, batch, cfg, target_kind)
                foreground = batch["target_heatmap"].flatten(1).ge(0.99).any(1)
                if foreground.any():
                    quality = F.smooth_l1_loss(output["q"][foreground],
                                               q_star[foreground].detach())
                    loss = loss + training["lambda_quality"] * quality
                    record["quality"] += float(quality.detach()) / accumulation
            (loss / accumulation).backward()
            record["loss"] += float(loss.detach()) / accumulation
            record["task"] += float(task.detach()) / accumulation
            for key, value in parts.items():
                record[key] = record.get(key, 0.0) + float(value.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_([parameter for group in groups
                                       for parameter in group["params"]],
                                      training["gradient_clip_norm"])
        optimizer.step()
        if step % 25 == 0 or training["fusion_steps"] <= 20:
            record["elapsed_s"] = round(time.time() - started, 1)
            if output.get("q") is not None:
                record["q_mean"] = output["q"].detach().mean(0).tolist()
            log_line(log, record)
            print(record, flush=True)
        if step % training["evaluation_interval_steps"] == 0 or step == training["fusion_steps"]:
            evaluation = eval_fusion(gate, root, val_ids, cfg, args.seed, device, config,
                                     directory / "_val")
            evaluation["step"] = step
            evaluation["peak_gpu_mem_gb"] = (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                                               if device == "cuda" else None)
            log_line(log, {"eval": evaluation})
            checkpoint = _checkpoint(gate, method, args.seed, args.stages, step,
                                     evaluation["summary"], cfg)
            if args.keep_all_checkpoints:
                trajectory = checkpoint_dir(cfg, args.variant) / "trajectory"
                trajectory.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, trajectory / f"{name}_step{step}.pt")
            clean_ap = evaluation["summary"]["clean_ap"]
            score = evaluation["summary"]["corruption_macro_ap"]
            if clean_ap is not None and clean_ap > fallback["clean_ap"]:
                fallback = {"clean_ap": clean_ap,
                            "checkpoint": _checkpoint(gate, method, args.seed, args.stages,
                                                      step, evaluation["summary"], cfg, clone=True),
                            "record": {"step": step, **evaluation["summary"]}}
            if clean_ap is not None and clean_ap >= clean_floor and score is not None and score > best:
                best, best_record = score, {"step": step, **evaluation["summary"]}
                torch.save(checkpoint, checkpoint_dir(cfg, args.variant) / f"{name}.pt")
    final = {
        "variant": args.variant, "method": method, "seed": args.seed,
        "stages": list(args.stages), "selected": best_record,
        "clean_floor": clean_floor, "step0": baseline["summary"],
        "selection_rule": "max validation corruption macro heatmap_peak_ap among checkpoints with clean heatmap_peak_ap >= step0 clean AP - max_clean_ap_drop",
        "wall_seconds": round(time.time() - started, 1),
        "seconds_per_step": round((time.time() - started) / training["fusion_steps"], 3),
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                            if device == "cuda" else None),
    }
    if best_record is None:
        if fallback["checkpoint"] is None:
            raise RuntimeError("validation produced no clean heatmap AP checkpoint")
        fallback["checkpoint"]["meta"]["eligible"] = False
        torch.save(fallback["checkpoint"], checkpoint_dir(cfg, args.variant) / f"{name}.pt")
        final["status"] = "no checkpoint satisfied the clean-drop rule; saved highest-clean-AP checkpoint flagged ineligible"
        final["fallback"] = fallback["record"]
    (directory / "final.json").write_text(json.dumps(final, indent=2))
    print(final, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/journal.yaml")
    parser.add_argument("--variant", choices=VARIANTS, default="transfuser")
    parser.add_argument("--stage", required=True, choices=("reliability", "fusion"))
    parser.add_argument("--method", choices=METHODS, default="P")
    parser.add_argument("--target", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", type=parse_stages, default=(0, 1, 2, 3))
    parser.add_argument("--steps", type=int, help="override stage step count for smoke runs")
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    parser.add_argument("--split-file", help="optional split.json override (used by isolated smoke runs)")
    parser.add_argument("--seed-file", help="released TransFuser seed checkpoint filename")
    parser.add_argument("--dataset-root", help="optional dataset root override for isolated runs")
    parser.add_argument("--artifacts-root", help="optional output root override for isolated runs")
    parser.add_argument("--workers", type=int, help="optional DataLoader worker override")
    args = parser.parse_args()
    if args.variant != "transfuser":
        parser.error("training currently requires --variant transfuser (the gated wrapper follows its four GPT stages)")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be >= 1")
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg["paths"]["dataset_root"] = str(Path(args.dataset_root).resolve())
    if args.artifacts_root:
        cfg["paths"]["artifacts_root"] = str(Path(args.artifacts_root).resolve())
    if args.workers is not None:
        if args.workers < 0:
            parser.error("--workers must be >= 0")
        cfg["training"]["workers"] = args.workers
    if args.steps is not None:
        cfg["reliability"]["steps"] = args.steps
        cfg["training"]["fusion_steps"] = args.steps
        cfg["training"]["evaluation_interval_steps"] = min(
            cfg["training"]["evaluation_interval_steps"], args.steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    {"reliability": stage_reliability, "fusion": stage_fusion}[args.stage](cfg, args, device)


if __name__ == "__main__":
    main()
