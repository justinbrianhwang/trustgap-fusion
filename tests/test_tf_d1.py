import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

if "rrf_tf" not in Path(sys.executable).parent.name:
    pytest.skip("TransFuser tests require the rrf_tf interpreter", allow_module_level=True)

import torch
from torch import nn

from rrf import tf_diagnose
from rrf.diag_common import GateTemporalAccumulator, cross_episode_permutation
from rrf.tf_corruptions import corrupt_tf_sample
from rrf.tf_diagnose import matched_center_count_1m, removal_outputs
from rrf.tf_evaluate import center_distance_ap, save_predictions
from rrf.tf_gated import GatedTransFuser
from rrf.tf_probes import heatmap_reference_error, render_tf_heatmap


class _Attention(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.key = nn.Linear(channels, channels)
        self.query = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)
        self.attn_drop = self.resid_drop = nn.Identity()
        self.n_head = 2


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(4)
        self.ln2 = nn.LayerNorm(4)
        self.attn = _Attention()
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


class _GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_embd = 4
        self.pos_emb = nn.Parameter(torch.randn(1, 8, 4) * 0.01)
        self.use_velocity = False
        self.drop = nn.Identity()
        self.blocks = nn.ModuleList([_Block()])
        self.ln_f = nn.LayerNorm(4)


class _Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.act1 = nn.ReLU()
        self.maxpool = nn.Identity()
        self.layer1 = self.layer2 = self.layer3 = self.layer4 = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)


class _Head(nn.Module):
    def forward(self, features):
        x = features[0]
        heat = x.mean(1, keepdim=True).sigmoid()
        zeros1 = torch.zeros_like(heat)
        zeros2 = torch.zeros((len(x), 2, *x.shape[2:]), device=x.device)
        zeros12 = torch.zeros((len(x), 12, *x.shape[2:]), device=x.device)
        return ([heat], [zeros2], [zeros2], [zeros12], [zeros1], [zeros1], [zeros2])


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = SimpleNamespace(normalize=False, features=_Features())
        self.add_module("image_features", self.image_encoder.features)
        self.lidar_encoder = SimpleNamespace(_model=_Features())
        self.add_module("lidar_features", self.lidar_encoder._model)
        self.avgpool_img = self.avgpool_lidar = nn.AdaptiveAvgPool2d((2, 2))
        self.transformer1 = _GPT()
        self.transformer2 = _GPT()
        self.transformer3 = _GPT()
        self.transformer4 = _GPT()
        self.change_channel_conv_image = self.change_channel_conv_lidar = nn.Identity()

    @staticmethod
    def top_down(x):
        return x, x, x, x


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = "transFuser"
        self._model = _Backbone()
        self.use_target_point_image = False
        self.head = _Head()
        self.pred_bev = nn.Conv2d(4, 3, 1)
        self.config = SimpleNamespace(center_net_max_pooling_kernel=3, top_k_center_keypoints=4,
                                      lidar_resolution_width=4, lidar_resolution_height=4,
                                      bev_resolution_width=4, bev_resolution_height=4,
                                      num_dir_bins=12)

    @staticmethod
    def forward_gru(fused, target):
        return fused[:, None, :2].expand(-1, 4, -1), None, None, None, None


def _batch():
    torch.manual_seed(4)
    return {"rgb": torch.randn(2, 4, 9, 12), "lidar": torch.randn(2, 4, 9, 12),
            "target_point": torch.zeros(2, 2), "target_point_image": torch.zeros(2, 1, 9, 12),
            "velocity": torch.zeros(2, 1)}


def test_q_one_identity_all_mechanisms_and_stage_subsets():
    model, batch = _Model().eval(), _batch()
    expected = GatedTransFuser(model, stages=()).eval()(batch)["center_heatmap"]
    for mechanism, stages in (("log_prior", (0,)), ("token_scaling", (1, 3)),
                              ("key_scaling", (0, 1, 2, 3))):
        actual = GatedTransFuser(model, mechanism=mechanism, stages=stages).eval()(
            batch, q_override=torch.ones(2, 2))["center_heatmap"]
        assert torch.allclose(actual, expected, atol=1e-5)


def test_mask_and_log_prior_attention():
    gate, batch = GatedTransFuser(_Model().eval(), stages=(0,)).eval(), _batch()
    neutral = gate(batch, q_override=torch.ones(2, 2), return_attention=True)
    low = gate(batch, q_override=torch.tensor([[0.1, 1.0]]).expand(2, -1), return_attention=True)
    assert torch.all(low["attention"]["stage_0"]["layer_0"][:, 0] <
                     neutral["attention"]["stage_0"]["layer_0"][:, 0])
    for layers in neutral["attention"].values():
        for mass in layers.values():
            assert torch.allclose(mass.sum(-1), torch.ones_like(mass[:, 0]), atol=1e-6)
    masked = gate(batch, modality_mask="camera")
    assert not torch.allclose(masked["center_heatmap"], neutral["center_heatmap"])
    with pytest.raises(ValueError, match="both"):
        gate(batch, modality_mask=("camera", "lidar"))


def test_gate_can_default_off_while_still_reporting_quality():
    class Estimator(nn.Module):
        def forward(self, stats, availability):
            return stats.new_tensor([0.1, 0.9]).expand(stats.shape[0], -1)

    model, batch = _Model().eval(), _batch()
    expected = GatedTransFuser(model).eval()(batch, gate_enabled=False)["center_heatmap"]
    output = GatedTransFuser(model, Estimator(), gate_enabled_by_default=False).eval()(batch)
    assert torch.allclose(output["center_heatmap"], expected, atol=1e-5)
    assert torch.allclose(output["q"], torch.tensor([[0.1, 0.9], [0.1, 0.9]]))


def test_tf_corruptions_are_deterministic_and_monotone():
    raw = {f"rgb_{name}": np.full((24, 32, 3), 200, np.uint8)
           for name in ("left", "front", "right")}
    raw.update(sample_id="episode/0001", lidar=np.ones((100, 4), np.float32))
    first, _ = corrupt_tf_sample(raw, {"camera": ("gaussian_noise", 3)}, 7)
    second, _ = corrupt_tf_sample(raw, {"camera": ("gaussian_noise", 3)}, 7)
    assert np.array_equal(first["rgb_front"], second["rgb_front"])
    means = [corrupt_tf_sample(raw, {"camera": ("darkening", severity)}, 7)[0]["rgb_front"].mean()
             for severity in range(1, 6)]
    assert all(a > b for a, b in zip(means, means[1:]))


def test_perfect_heatmap_reference_error_is_zero():
    config = SimpleNamespace(lidar_resolution_width=256, lidar_resolution_height=256)
    boxes = np.array([[128, 128, 16, 32, 0, 0, 0]], np.float32)
    target = render_tf_heatmap(boxes, config)[None, None]
    perfect = (target >= 0.99).float()  # CenterNet convention: 1 at the peak, 0 elsewhere (tails ignored)
    assert heatmap_reference_error(perfect, target).item() == 0.0
    tails_on = torch.where(target >= 0.99, perfect, target)  # tail values are ignored, not penalised
    assert heatmap_reference_error(tails_on, target).item() < 1e-4
    assert heatmap_reference_error(torch.zeros_like(target), target).item() > 0.4  # missing the peak is penalised


def test_center_distance_ap_known_answer():
    target = np.array([[10, 10, 1, 1, 0, 0, 0]], np.float32)
    prediction = np.array([[30, 30, 1, 1, 0, 0, 0, 0.9],
                           [10, 10, 1, 1, 0, 0, 0, 0.8]], np.float32)
    ap, info = center_distance_ap([prediction], [target], pixels_per_meter=1, radius=1)
    assert ap == 0.5 and info["n_gt"] == 1 and info["n_det"] == 2


def test_tf_predictions_are_saved_padded_with_counts(tmp_path):
    heatmaps = [np.ones((64, 64), np.float32), np.zeros((64, 64), np.float32)]
    boxes = [np.ones((2, 8), np.float32), np.full((1, 8), 2, np.float32)]
    path = tmp_path / "pred_clean.npz"
    save_predictions(path, ["ep/0", "ep/1"], heatmaps, boxes)
    with np.load(path) as saved:
        assert saved["center_heatmap"].dtype == np.float16
        assert saved["center_heatmap"].shape == (2, 64, 64)
        assert saved["boxes"].shape == (2, 2, 8)
        assert saved["box_counts"].tolist() == [2, 1]
        assert np.all(saved["boxes"][1, 1] == 0)


def test_diagnostic_utility_interventions_keep_the_same_weights():
    gate, batch = GatedTransFuser(_Model().eval()).eval(), _batch()
    before = {name: value.clone() for name, value in gate.state_dict().items()}
    means = {stage: {modality: torch.zeros(4) for modality in ("camera", "lidar")}
             for stage in range(4)}
    outputs = removal_outputs(gate, batch, means)
    assert set(outputs) == {
        (variant, level, modality)
        for variant, level in (("sensor-level removal", "sensor"),
                               ("stage-level removal (all stages)", "stage"),
                               ("stage-level mean replacement (all stages)", "stage"))
        for modality in ("camera", "lidar")}
    assert all(torch.equal(value, gate.state_dict()[name]) for name, value in before.items())


def test_sensor_removal_replaces_inputs_before_the_encoders():
    gate, batch = GatedTransFuser(_Model().eval()).eval(), _batch()
    seen = {}
    camera_handle = gate.model._model.image_encoder.features.conv1.register_forward_pre_hook(
        lambda _module, inputs: seen.__setitem__("camera", inputs[0].detach().clone()))
    lidar_handle = gate.model._model.lidar_encoder._model.conv1.register_forward_pre_hook(
        lambda _module, inputs: seen.__setitem__("lidar", inputs[0].detach().clone()))
    try:
        gate(batch, sensor_remove="camera")
        assert torch.count_nonzero(seen["camera"]) == 0
        gate(batch, sensor_remove="lidar")
        assert torch.count_nonzero(seen["lidar"]) == 0
    finally:
        camera_handle.remove()
        lidar_handle.remove()
    assert torch.count_nonzero(batch["rgb"]) > 0 and torch.count_nonzero(batch["lidar"]) > 0


def test_diagnostic_shuffle_crosses_episodes():
    episodes = ["a", "a", "b", "b", "c", "c"]
    permutation = cross_episode_permutation(episodes, seed=7)
    assert sorted(permutation.tolist()) == list(range(len(episodes)))
    assert all(episodes[source] != episodes[destination]
               for destination, source in enumerate(permutation))


def test_matched_center_count_is_score_ordered_and_one_to_one():
    target = np.array([[10, 10, 1, 1, 0, 0, 0]], np.float32)
    prediction = np.array([[10, 10, 1, 1, 0, 0, 0, 0.9],
                           [10.2, 10, 1, 1, 0, 0, 0, 0.8]], np.float32)
    assert matched_center_count_1m(prediction, target, pixels_per_meter=1) == 1


def test_transfuser_temporal_analysis_runs_from_synthetic_outputs(monkeypatch):
    ids = ["ep0/0000", "ep0/0002", "ep1/0000", "ep1/0002"]
    heatmaps = np.zeros((len(ids), 4, 4), np.float32)
    for index in range(len(ids)):
        heatmaps[index, index // 2, index % 2] = 1
    rng = np.random.default_rng(6)
    scores = rng.random((16, *heatmaps.shape), dtype=np.float32)
    scores[0] = heatmaps * 0.9 + 0.05
    boxes = [np.array([[index % 2, index // 2, 1, 1, 0, 0, 0]], np.float32)
             for index in range(len(ids))]
    candidates = [(a, b) for a in tf_diagnose.GRID for b in tf_diagnose.GRID]
    sample_index = {sample_id: index for index, sample_id in enumerate(ids)}

    class FakeGate:
        def __call__(self, batch, q_override=None):
            selected = 15 if q_override is None else min(
                range(len(candidates)),
                key=lambda value: sum(abs(a - b) for a, b in zip(candidates[value], q_override[0].tolist())))
            indices = [sample_index[sample_id] for sample_id in batch["sample_id"]]
            predicted_boxes = np.stack([
                np.concatenate([boxes[index], np.full((1, 1), 0.9, np.float32)], axis=1)
                for index in indices])
            return {"center_heatmap": torch.from_numpy(scores[selected, indices])[:, None],
                    "boxes": torch.from_numpy(predicted_boxes)}

    def fake_loader(_root, sample_ids, _spec, _seed, _config, _batch_size):
        for offset in range(0, len(sample_ids), 2):
            batch_ids = sample_ids[offset:offset + 2]
            yield batch_ids, {"sample_id": batch_ids}

    def fake_truth(batch_ids, _targets, _config, _device):
        indices = [sample_index[sample_id] for sample_id in batch_ids]
        return torch.from_numpy(heatmaps[indices]), [boxes[index] for index in indices]

    monkeypatch.setattr(tf_diagnose, "_loader", fake_loader)
    monkeypatch.setattr(tf_diagnose, "_truth", fake_truth)
    names = [f"{a:g},{b:g}" for a, b in candidates] + ["learned_q"]
    accumulator = GateTemporalAccumulator(names, names.index("1,1"), "transfuser")
    summary = tf_diagnose.run_oracle_gate_temporal(
        FakeGate(), {"dataset_root": "unused", "corruption": {"base_seed": 0}}, ids,
        {"clean": {}}, dict(zip(ids, boxes)), "cpu", 2,
        SimpleNamespace(pixels_per_meter=1), accumulator)
    assert summary["best_fixed_candidate"] == names[0]
    assert summary["in_sample_oracle_learned_loss_violations"] == 0
    assert set(summary["ap"]["temporal_oracle"]["clean"]) == {
        "heatmap_peak_ap", "vehicle_center_ap_1m"}


def test_transfuser_partial_analysis_scales_and_clips_learned_q(monkeypatch):
    ids = ["ep0/0000", "ep1/0000"]
    learned_q = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    boxes = [np.ones((1, 7), np.float32) for _ in ids]
    captured = {}

    class FakeGate:
        eps = 0.1

        def __call__(self, batch, q_override=None):
            q = learned_q if q_override is None else q_override
            return {"q": q, "center_heatmap": q.sum(1)[:, None, None, None],
                    "boxes": torch.zeros(len(q), 1, 8)}

    monkeypatch.setattr(tf_diagnose, "_loader",
                        lambda *_args: iter([(ids, {"sample_id": ids})]))
    monkeypatch.setattr(tf_diagnose, "_truth", lambda *_args: (
        torch.ones(2, 1, 1), boxes))
    monkeypatch.setattr(tf_diagnose, "q_targets", lambda *_args: (
        torch.ones(2, 2), torch.tensor([[0.1, 0.2], [0.3, 0.4]])))
    monkeypatch.setattr(tf_diagnose, "prediction_losses", lambda output, *_args: (
        output["q"].sum(1), (output["q"][:, 0] * 10).long()))
    monkeypatch.setattr(tf_diagnose, "write_csv",
                        lambda _path, rows: captured.setdefault("rows", rows))
    project = {"dataset_root": "unused", "corruption": {"base_seed": 0},
               "reliability": {"tau_quality": 0.2}}
    config = SimpleNamespace(pixels_per_meter=1)
    summary = tf_diagnose.run_partial(
        FakeGate(), object(), project, ids, {"clean": {}}, {}, Path("unused"),
        "cpu", 2, 0, config)
    assert len(captured["rows"]) == len(ids) * 2 * 4
    assert min(row["intervened_q"] for row in captured["rows"]) == pytest.approx(0.1)
    assert max(row["intervened_q"] for row in captured["rows"]) == pytest.approx(1.0)
    assert "clean" in summary["foreground_by_kind"]
