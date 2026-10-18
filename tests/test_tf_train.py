import copy
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

if "rrf_tf" not in Path(sys.executable).parent.name:
    pytest.skip("TransFuser training tests require the rrf_tf interpreter", allow_module_level=True)

import torch
from torch import nn

from rrf.config import load_config
from rrf.gated import ReliabilityEstimator
from rrf.tf_corruptions import corruption_seed
from rrf.tf_train import (GATED_METHODS, PRETRAINED_ROUTER_METHODS,
                          TF_TRAINABLE_PREFIXES, encode_detection_targets,
                          native_detection_loss, scheduled_condition, set_trainable,
                          stage_fusion, train_mode)
from rrf.train import fusion_objective, sample_condition


class _Head:
    num_dir_bins = 12

    def angle2class(self, angle):
        angle = angle % (2 * np.pi)
        width = 2 * np.pi / self.num_dir_bins
        shifted = (angle + width / 2) % (2 * np.pi)
        cls = torch.div(shifted, width, rounding_mode="trunc")
        return cls.long(), shifted - (cls * width + width / 2)


def _config():
    return SimpleNamespace(lidar_resolution_width=256, lidar_resolution_height=256)


def test_perfect_native_detection_prediction_has_minimal_terms():
    boxes = [torch.tensor([[101.25, 149.75, 16.0, 32.0, 0.3, 4.0, 0.0]])]
    targets = encode_detection_targets(_Head(), boxes, (1, 1, 64, 64), _config())
    peak = targets["center_heatmap"].eq(1)
    output = {
        "center_heatmap": torch.where(peak, torch.full_like(targets["center_heatmap"], 1 - 1e-6),
                                      torch.full_like(targets["center_heatmap"], 1e-6)),
        "box_wh": targets["box_wh"].clone(),
        "box_offset": targets["box_offset"].clone(),
        "yaw_residual": targets["yaw_residual"].clone(),
        "box_velocity": targets["box_velocity"].clone(),
        "yaw_class": torch.full((1, 12, 64, 64), -20.0),
    }
    y, x = torch.nonzero(targets["weight"][0, 0], as_tuple=True)
    cls = targets["yaw_class"][0, y, x]
    output["yaw_class"][0, cls, y, x] = 20.0
    total, parts = native_detection_loss(output, boxes, _Head(), _config())
    assert total.item() < 1e-5
    assert set(parts) == {"center_heatmap", "box_wh", "box_offset", "yaw_class",
                          "yaw_residual", "velocity"}
    assert all(value.item() < 1e-5 for value in parts.values())


class _FreezeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = nn.Module()
        self._model.transformer1 = nn.Linear(2, 2)
        self._model.image_encoder = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
        self._model.up_conv5 = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)
        self.decoder = nn.GRUCell(2, 2)


def test_freeze_prefixes_and_gradients_do_not_reach_frozen_parameters():
    model = _FreezeModel()
    trainable, total = set_trainable(model)
    expected = {name for name, _ in model.named_parameters()
                if name.startswith(TF_TRAINABLE_PREFIXES)}
    assert {name for name, p in model.named_parameters() if p.requires_grad} == expected
    assert trainable == sum(p.numel() for name, p in model.named_parameters() if name in expected)
    assert total == sum(p.numel() for p in model.parameters())
    train_mode(model)
    assert model._model.transformer1.training and model.head.training
    assert not model._model.image_encoder[0].training
    sum((parameter.square().sum() for parameter in model.parameters()
         if parameter.requires_grad)).backward()
    assert all((parameter.grad is None) == (not parameter.requires_grad)
               for parameter in model.parameters())


def test_gate_only_optimizer_step_leaves_fusion_weights_unchanged():
    torch.manual_seed(2)
    model = _FreezeModel()
    set_trainable(model, ())
    before = copy.deepcopy(model.state_dict())
    estimator = ReliabilityEstimator(3, hidden=4, layers=1)
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=0.1)
    loss = estimator(torch.randn(2, 2, 3), torch.ones(2, 4)).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_b3a_and_p_steps_differ_only_by_quality_gradient():
    assert "B3A" in PRETRAINED_ROUTER_METHODS and "B3A" not in GATED_METHODS
    q_b3a = torch.tensor([[0.2, 0.8]], requires_grad=True)
    q_p = q_b3a.detach().clone().requires_grad_()
    q_star = torch.tensor([[0.9, 0.1]])
    foreground = torch.tensor([True])
    b3a, _ = fusion_objective(q_b3a.square().sum(), q_b3a, None,
                              foreground, 0.7, False)
    p, quality = fusion_objective(q_p.square().sum(), q_p, q_star,
                                  foreground, 0.7, True)
    quality_gradient = torch.autograd.grad(quality, q_p, retain_graph=True)[0]
    b3a.backward()
    p.backward()
    assert torch.allclose(q_p.grad - q_b3a.grad, 0.7 * quality_gradient)


def test_tf_schedule_draw_equals_interfuser_draw():
    config = load_config("configs/journal_weak.yaml")["corruption"]
    seed, sample_id, epoch = 17, "Town03_ep00_seed4700/0042", 3
    draw_seed = corruption_seed(seed, sample_id, "schedule", epoch, 0)
    expected = sample_condition(np.random.default_rng(draw_seed), config)
    assert scheduled_condition(config, seed, sample_id, epoch) == expected


def test_three_step_p_smoke_on_trial_episode(monkeypatch, tmp_path):
    import rrf.tf_train as tf_train

    episode = "Town03_ep00_seed4700"
    ids = [f"{episode}/{frame:04d}" for frame in (18, 19, 20, 21)]
    split = {
        "train": {"probe_fit_groups": ids[:1], "reliability_fit_groups": ids[1:3]},
        "validation": {"model_select_groups": ids[3:], "calibration_groups": []},
        "test": {"locked_groups": ids[3:]},
    }
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps(split))
    config = load_config("configs/journal.yaml")
    config["paths"]["dataset_root"] = str(Path("dataset_trial_dual").resolve())
    config["paths"]["artifacts_root"] = str(tmp_path)
    config["training"].update(batch_size=1, workers=0, fusion_steps=3,
                              evaluation_interval_steps=3)
    checkpoint_dir = tmp_path / "tf_checkpoints" / "transfuser"
    checkpoint_dir.mkdir(parents=True)
    estimator = ReliabilityEstimator(152, config["reliability"]["estimator_hidden_dim"],
                                     config["reliability"]["estimator_hidden_layers"])
    torch.save({"estimator": estimator.state_dict()}, checkpoint_dir / "reliability_seed0.pt")
    saved = []
    monkeypatch.setattr(tf_train, "validation_suite", lambda: {"clean": {}})
    monkeypatch.setattr(tf_train.torch, "save", lambda _state, path: saved.append(Path(path)))
    args = Namespace(split_file=str(split_file), seed=0, variant="transfuser",
                     seed_file=None, method="P", stages=(0, 1, 2, 3),
                     keep_all_checkpoints=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage_fusion(config, args, device)
    run = tmp_path / "tf_runs" / "transfuser" / "fusion_P_seed0"
    assert (run / "setup.json").is_file() and (run / "final.json").is_file()
    frames = [json.loads(line) for line in (run / "_val" / "frames.jsonl").read_text().splitlines()]
    assert frames and all(frame["q"] is not None and len(frame["q"]) == 2 for frame in frames)
    assert any(path.name == "fusion_P_seed0.pt" for path in saved)
