"""YAML config loading with strict key validation (unknown keys / bad sums are errors)."""
import hashlib
from pathlib import Path

import yaml

from rrf import ROOT

WEAK_TRAIN_PROBABILITIES = {
    "clean": 0.70,
    "camera_only_corrupted": 0.12,
    "lidar_only_corrupted": 0.12,
    "both_corrupted": 0.06,
}

# Every allowed section -> allowed keys. A key missing here is rejected so typos never
# silently fall back to defaults.
SCHEMA = {
    "project": {"name", "scope", "schema_version"},
    "paths": {"upstream_repo", "carla_root", "dataset_root", "upstream_checkpoint", "artifacts_root"},
    "upstream": {"repository", "commit", "model_name", "expected_carla_release", "strict_original_checkpoint"},
    "execution": {"device", "seed", "overwrite_existing_outputs"},
    "simulation": {
        "host", "port", "traffic_manager_port", "synchronous_mode", "fixed_delta_seconds",
        "capture_every_n_ticks", "no_rendering_mode", "warmup_ticks", "frames_per_episode",
        "num_vehicles", "num_walkers", "ego_blueprint", "ego_target_speed_kmh", "weather",
        "towns", "episodes_per_town",
    },
    "split": {
        "unit", "train_fraction", "validation_fraction", "test_fraction",
        "probe_fraction_within_train", "calibration_fraction_within_validation", "split_seed",
        "pretraining_overlap",
    },
    "corruption": {
        "base_seed", "primary_camera_scope", "train_families", "train_severities",
        "test_severities", "heldout_families", "train_condition_probabilities",
        "silent_full_loss_fraction_of_single_corruption", "train_profile",
    },
    "reliability": {"tau_quality", "tau_relative", "add_raw_statistics", "estimator_hidden_dim", "estimator_hidden_layers", "steps", "learning_rate", "batch_size"},
    "fusion": {"method", "alpha", "epsilon", "apply_encoder_self_attention", "apply_decoder_cross_attention", "freeze_tokenizer"},
    "training": {
        "optimizer", "learning_rate", "weight_decay", "batch_size", "gradient_accumulation_steps",
        "gradient_clip_norm", "fusion_steps", "evaluation_interval_steps", "lambda_quality", "workers",
    },
    "metrics": {
        "foreground_target_threshold", "clean_calibration_precision_target", "near_distance_m",
        "max_clean_ap_drop", "bootstrap_repeats",
    },
}


class ConfigError(ValueError):
    pass


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)
    validate(cfg)
    cfg["_meta"] = {
        "config_path": str(path),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
    }
    # Resolve paths against the project root.
    for key, value in cfg["paths"].items():
        if value is None:
            continue
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else ROOT / p)
    return cfg


def validate(cfg):
    if not isinstance(cfg, dict):
        raise ConfigError("config root must be a mapping")
    unknown_sections = set(cfg) - set(SCHEMA)
    if unknown_sections:
        raise ConfigError(f"unknown config sections: {sorted(unknown_sections)}")
    missing_sections = set(SCHEMA) - set(cfg)
    if missing_sections:
        raise ConfigError(f"missing config sections: {sorted(missing_sections)}")
    for section, allowed in SCHEMA.items():
        unknown = set(cfg[section]) - allowed
        if unknown:
            raise ConfigError(f"unknown keys in [{section}]: {sorted(unknown)}")

    split = cfg["split"]
    total = split["train_fraction"] + split["validation_fraction"] + split["test_fraction"]
    if abs(total - 1.0) > 1e-9:
        raise ConfigError(f"split fractions sum to {total}, expected 1.0")

    corr = cfg["corruption"]
    corr.setdefault("train_profile", "standard")
    if corr["train_profile"] not in ("standard", "weak"):
        raise ConfigError("corruption.train_profile must be 'standard' or 'weak'")
    if corr["train_profile"] == "weak":
        corr["train_severities"] = [1]
        corr["train_condition_probabilities"] = dict(WEAK_TRAIN_PROBABILITIES)
    probs = corr["train_condition_probabilities"]
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-9:
        raise ConfigError(f"train_condition_probabilities sum to {total}, expected 1.0")
    for modality in ("camera", "lidar"):
        overlap = set(corr["train_families"][modality]) & set(corr["heldout_families"][modality])
        if overlap:
            raise ConfigError(f"held-out {modality} families also in train: {sorted(overlap)}")
    if not set(corr["train_severities"]) <= set(corr["test_severities"]):
        raise ConfigError("train_severities must be a subset of test_severities")
