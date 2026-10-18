"""Small CPU checks for the diagnostic interventions and attention capture."""
import numpy as np
import torch
import pytest

from rrf import upstream
from rrf.config import load_config
from rrf.diag_common import GateTemporalAccumulator, GateXfitAccumulator, partial_summary
from rrf.gated import GatedInterFuser
from rrf.tools import diagnose
from rrf.tools.diagnose import attention_masses, cross_episode_permutation, per_sample_native_score
from rrf.train import GATED_METHODS, PRETRAINED_ROUTER_METHODS, fusion_objective


@pytest.fixture(scope="module")
def gate():
    cfg = load_config("configs/pilot.yaml")
    model, _ = upstream.load_upstream_model(cfg["paths"]["upstream_checkpoint"], "cpu")
    return GatedInterFuser(model, estimator=None).eval()


def _batch(seed=4):
    generator = torch.Generator().manual_seed(seed)
    return {
        "rgb": torch.randn(1, 3, 224, 224, generator=generator) * 0.3,
        "rgb_left": torch.randn(1, 3, 128, 128, generator=generator) * 0.3,
        "rgb_right": torch.randn(1, 3, 128, 128, generator=generator) * 0.3,
        "rgb_center": torch.randn(1, 3, 128, 128, generator=generator) * 0.3,
        "lidar": torch.rand(1, 3, 224, 224, generator=generator),
        "measurements": torch.tensor([[0, 0, 0, 1, 0, 0, 3.0]]),
        "target_point": torch.tensor([[0.0, -8.0]]),
    }


def test_hardmask_utility_uses_same_weights_and_changes_output(gate):
    batch = _batch()
    before = {name: value.clone() for name, value in gate.state_dict().items()}
    target = torch.zeros(1, 400, 7)
    with torch.no_grad():
        full = gate(batch, gate_enabled=False)["traffic"]
        removed = gate(batch, availability=torch.tensor([[0.0, 0.0, 0.0, 1.0]]), gate_enabled=False)["traffic"]
    utility = per_sample_native_score(removed, target) - per_sample_native_score(full, target)
    assert utility.shape == (1,) and not torch.allclose(full, removed, atol=1e-3)
    assert all(torch.equal(value, gate.state_dict()[name]) for name, value in before.items())


def test_neutral_override_reproduces_ungated(gate):
    batch = _batch()
    with torch.no_grad():
        ungated = gate(batch, gate_enabled=False)["traffic"]
        neutral = gate(batch, q_override=torch.ones(1, 2))["traffic"]
    assert torch.allclose(ungated, neutral, atol=1e-5, rtol=1e-4)


def test_attention_capture_mass_and_log_prior_response(gate):
    batch = _batch()
    one, mass_one = attention_masses(gate, batch, torch.ones(1, 2))
    _, mass_low = attention_masses(gate, batch, torch.tensor([[0.05, 1.0]]))
    expected = len(gate.model.encoder.layers) + len(gate.model.decoder.layers)
    assert len(mass_one) == expected and one["traffic"].shape == (1, 400, 7)
    for name, mass in mass_one.items():
        assert torch.allclose(mass.sum(-1), torch.ones_like(mass[..., 0]), atol=1e-5), name
        assert mass_low[name][..., 0].mean() < mass[..., 0].mean(), name


def test_within_condition_shuffle_crosses_episodes():
    episodes = ["a", "a", "b", "b", "c", "c"]
    permutation = cross_episode_permutation(episodes, seed=7)
    assert sorted(permutation.tolist()) == list(range(len(episodes)))
    assert all(episodes[source] != episodes[destination] for destination, source in enumerate(permutation))


def test_cross_fitted_oracle_removes_noise_selection_gain():
    rng = np.random.default_rng(22)
    labels = (rng.random((500, 400)) < 0.08).astype(np.float32)
    candidates = rng.random((16, 500, 400), dtype=np.float32)
    accumulator = GateXfitAccumulator(400, [str(i) for i in range(16)], 15, 7, "transfuser")
    accumulator.add("noise", candidates, candidates[15], labels)
    for fold, index in enumerate(accumulator.best_indices):
        accumulator.add_fixed("noise", fold, candidates[index], labels)
    ap = accumulator.summary()["ap"]
    neutral = ap["neutral"]["noise"]
    assert abs(ap["cross_fitted_oracle"]["noise"] - neutral) < 0.01
    assert ap["in_sample_oracle"]["noise"] - neutral > 0.02


def test_learned_q_is_an_oracle_candidate_and_best_fixed_stays_fixed():
    rng = np.random.default_rng(8)
    labels = (rng.random((20, 40)) < 0.1).astype(np.float32)
    fixed = rng.random((16, *labels.shape), dtype=np.float32)
    learned = labels * 0.99 + 0.005
    candidates = np.concatenate((fixed, learned[None]))
    names = [str(index) for index in range(16)] + ["learned_q"]
    accumulator = GateXfitAccumulator(40, names, 15, 2, "transfuser")
    accumulator.add("synthetic", candidates, learned, labels)
    assert all(index < 16 for index in accumulator.best_indices)
    for fold, index in enumerate(accumulator.best_indices):
        accumulator.add_fixed("synthetic", fold, candidates[index], labels)
    summary = accumulator.summary()
    assert summary["in_sample_oracle_learned_loss_violations"] == 0
    assert summary["ap"]["in_sample_oracle"]["synthetic"] > 0.99


def test_partial_summary_and_aligned_router_gradient_contract():
    rows = []
    for index, q in enumerate((0.1, 0.3, 0.7, 0.9)):
        for factor in (0.5, 0.8, 1.25, 2.0):
            rows.append({"episode_id": f"ep{index // 2}", "fg_present": True,
                         "kind": "seen", "modality": "camera", "factor": factor,
                         "q": q, "E": 1 - q,
                         "delta_loss": (q - 0.5) * (1 if factor > 1 else -1)})
            rows.append({**rows[-1], "modality": "lidar"})
    summary = partial_summary(rows, ("loss",), seed=3, repeats=20)
    camera = summary["seen"]["camera"]["loss"]
    assert camera["fraction_lower_f0.5"]["estimate"] == 0.5
    assert camera["q_vs_neg_delta_f0.5"]["estimate"] == pytest.approx(1.0)

    assert "B3A" in PRETRAINED_ROUTER_METHODS and "B3A" not in GATED_METHODS
    q_b3a = torch.tensor([[0.2, 0.8]], requires_grad=True)
    q_p = q_b3a.detach().clone().requires_grad_()
    target = torch.tensor([[0.9, 0.1]])
    foreground = torch.tensor([True])
    task_b3a = q_b3a.square().sum()
    task_p = q_p.square().sum()
    b3a, _ = fusion_objective(task_b3a, q_b3a, None, foreground, 0.7, False)
    p, quality = fusion_objective(task_p, q_p, target, foreground, 0.7, True)
    quality_grad = torch.autograd.grad(quality, q_p, retain_graph=True)[0]
    b3a.backward()
    p.backward()
    assert torch.allclose(q_p.grad - q_b3a.grad, 0.7 * quality_grad)


def _temporal_summary(sample_ids, candidates, labels, neutral_index):
    losses = ((candidates - labels[None]) ** 2).mean(2)
    accumulator = GateTemporalAccumulator(
        [str(index) for index in range(len(candidates))], neutral_index, "transfuser")
    accumulator.add("synthetic", sample_ids, candidates, losses,
                    candidates[neutral_index], labels)
    accumulator.add_fixed("synthetic", candidates[accumulator.best_index])
    return accumulator, accumulator.summary()


def test_temporal_oracle_has_no_expected_gain_for_iid_candidate_quality():
    rng = np.random.default_rng(31)
    labels = (rng.random((1000, 100)) < 0.08).astype(np.float32)
    candidates = rng.random((4, *labels.shape), dtype=np.float32)
    _, summary = _temporal_summary(
        [f"episode/{index:04d}" for index in range(len(labels))], candidates, labels, 3)
    ap = summary["ap"]
    assert abs(ap["temporal_oracle"]["synthetic"] - ap["neutral"]["synthetic"]) < 0.02


def test_temporal_oracle_recovers_stable_episode_choice_and_resets_to_neutral():
    rng = np.random.default_rng(9)
    frames_per_episode, episodes, cells = 20, 20, 64
    labels = (rng.random((frames_per_episode * episodes, cells)) < 0.1).astype(np.float32)
    candidates = np.stack([
        np.clip(0.05 + 0.9 * labels + rng.normal(0, 0.02, labels.shape), 0, 1),
        rng.random(labels.shape),
        rng.random(labels.shape),
    ]).astype(np.float32)
    sample_ids = [f"ep{episode}/{frame:04d}"
                  for episode in range(episodes) for frame in range(frames_per_episode)]
    accumulator, summary = _temporal_summary(sample_ids, candidates, labels, 2)
    ap = summary["ap"]
    neutral, temporal = ap["neutral"]["synthetic"], ap["temporal_oracle"]["synthetic"]
    in_sample = ap["in_sample_oracle"]["synthetic"]
    assert temporal - neutral > 0.9 * (in_sample - neutral)
    temporal_scores = np.concatenate(accumulator.scores["temporal_oracle"]["synthetic"])
    first = np.arange(0, len(sample_ids), frames_per_episode)
    assert np.array_equal(temporal_scores[first], candidates[2, first])


def test_interfuser_temporal_analysis_runs_from_synthetic_outputs(monkeypatch):
    ids = ["ep0/0000", "ep0/0002", "ep1/0000", "ep1/0002"]
    labels = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                       [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    rng = np.random.default_rng(5)
    scores = rng.random((16, len(ids), 4), dtype=np.float32)
    scores[0] = labels * 0.9 + 0.05
    candidates = [(a, b) for a in diagnose.GRID for b in diagnose.GRID]
    index = {sample_id: offset for offset, sample_id in enumerate(ids)}

    class FakeGate:
        def __call__(self, batch, q_override=None):
            selected = 15 if q_override is None else min(
                range(len(candidates)),
                key=lambda value: sum(abs(a - b) for a, b in zip(candidates[value], q_override[0].tolist())))
            batch_scores = torch.from_numpy(scores[selected, [index[sid] for sid in batch["sample_id"]]])
            traffic = torch.zeros(len(batch_scores), 4, 7)
            traffic[..., 0] = batch_scores
            return {"traffic": traffic}

    def fake_loader(_root, sample_ids, _spec, _seed, _batch_size):
        for offset in range(0, len(sample_ids), 2):
            yield {"sample_id": sample_ids[offset:offset + 2]}

    def fake_truth(batch_ids, _targets, _device):
        target = torch.zeros(len(batch_ids), 4, 7)
        target[..., 0] = torch.from_numpy(labels[[index[sid] for sid in batch_ids]])
        return target, target[..., 0] > 0.01

    monkeypatch.setattr(diagnose, "_loader", fake_loader)
    monkeypatch.setattr(diagnose, "_model_batch", lambda item, _device: item)
    monkeypatch.setattr(diagnose, "_truth", fake_truth)
    names = [f"{a:g},{b:g}" for a, b in candidates] + ["learned_q"]
    accumulator = GateTemporalAccumulator(names, names.index("1,1"), "interfuser")
    summary = diagnose.run_oracle_gate_temporal(
        FakeGate(), {"paths": {"dataset_root": "unused"}, "corruption": {"base_seed": 0}},
        ids, {"clean": {}}, {}, "cpu", 2, accumulator)
    assert set(summary["ap"]) == {"learned", "neutral", "best_fixed",
                                  "in_sample_oracle", "temporal_oracle"}
    assert summary["best_fixed_candidate"] == names[0]
    assert summary["in_sample_oracle_learned_loss_violations"] == 0


def test_interfuser_partial_analysis_scales_and_clips_learned_q(monkeypatch):
    ids = ["ep0/0000", "ep1/0000"]
    learned_q = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    target = torch.zeros(2, 4, 7)
    target[:, 0, 0] = 1
    captured = {}

    class FakeGate:
        eps = 0.1

        def __call__(self, batch, q_override=None):
            q = learned_q if q_override is None else q_override
            traffic = torch.zeros(len(q), 4, 7)
            traffic[..., 0] = q.sum(1, keepdim=True)
            return {"q": q, "traffic": traffic}

    monkeypatch.setattr(diagnose, "_loader", lambda *_args: iter([{"sample_id": ids}]))
    monkeypatch.setattr(diagnose, "_model_batch", lambda item, _device: item)
    monkeypatch.setattr(diagnose, "_truth", lambda *_args: (target, target[..., 0] > 0.01))
    monkeypatch.setattr(diagnose, "q_targets", lambda *_args: (
        torch.ones(2, 2), torch.tensor([[0.1, 0.2], [0.3, 0.4]])))
    monkeypatch.setattr(diagnose, "_write_csv",
                        lambda _path, rows: captured.setdefault("rows", rows))
    cfg = {"paths": {"dataset_root": "unused"},
           "corruption": {"base_seed": 0}, "reliability": {"tau_quality": 0.2}}
    summary = diagnose.run_partial(FakeGate(), object(), cfg, ids, {"clean": {}}, {},
                                   diagnose.Path("unused"), "cpu", 2, 0)
    assert len(captured["rows"]) == len(ids) * 2 * 4
    camera_half = next(row for row in captured["rows"]
                       if row["sample_id"] == ids[0] and row["modality"] == "camera"
                       and row["factor"] == 0.5)
    lidar_double = next(row for row in captured["rows"]
                        if row["sample_id"] == ids[0] and row["modality"] == "lidar"
                        and row["factor"] == 2.0)
    assert camera_half["intervened_q"] == pytest.approx(0.1)
    assert lidar_double["intervened_q"] == pytest.approx(1.0)
    assert "clean" in summary["foreground_by_kind"]
