"""Gate invariance tests (md §9.8). Run on CPU float32 with atol=1e-6/rtol=1e-5 where the
same kernel path is used, looser where a masked vs. unmasked attention kernel differs."""
import copy

import numpy as np
import pytest
import torch

from rrf import upstream
from rrf.config import load_config
from rrf.gated import (FUSION_PREFIXES, GatedInterFuser, ReliabilityEstimator, StaticGate, assemble,
                       modality_statistics, set_trainable, tokenize, train_mode)

torch.manual_seed(0)


@pytest.fixture(scope="module")
def model():
    cfg = load_config("configs/pilot.yaml")
    m, _ = upstream.load_upstream_model(cfg["paths"]["upstream_checkpoint"], "cpu")
    return m


def _batch(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "rgb": torch.randn(B, 3, 224, 224, generator=g) * 0.3,
        "rgb_left": torch.randn(B, 3, 128, 128, generator=g) * 0.3,
        "rgb_right": torch.randn(B, 3, 128, 128, generator=g) * 0.3,
        "rgb_center": torch.randn(B, 3, 128, 128, generator=g) * 0.3,
        "lidar": torch.rand(B, 3, 224, 224, generator=g).clamp(0, 1),
        "measurements": torch.tensor([[0, 0, 0, 1, 0, 0, 3.0]] * B),
        "target_point": torch.tensor([[0.0, -8.0]] * B),
    }


def test_assemble_matches_upstream_forward_features(model):
    x = _batch()
    with torch.no_grad():
        ref = model.forward_features(x["rgb"], x["rgb_left"], x["rgb_right"], x["rgb_center"], x["lidar"], x["measurements"])
        feat, layout = assemble(tokenize(model, x))
    assert torch.equal(ref, feat)
    assert len(layout) == ref.shape[0] and layout.count("lidar") == 50 and layout.count("front") == 50


def test_gate_disabled_and_all_q_one_match_upstream(model):
    x = _batch()
    g = GatedInterFuser(model, estimator=None).eval()
    with torch.no_grad():
        ref = upstream.predict(model, x)
        off = g(x, gate_enabled=False)
        one = g(x, q_override=torch.ones(2, 2))
    assert torch.allclose(ref["traffic"], off["traffic"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(ref["waypoints"], off["waypoints"], atol=1e-6, rtol=1e-5)
    # a zero float mask goes through a masked attention kernel: allow tiny numeric drift
    assert torch.allclose(ref["traffic"], one["traffic"], atol=1e-5, rtol=1e-4)


def test_q_changes_output_and_batch_isolation(model):
    x = _batch(B=2)
    g = GatedInterFuser(model, estimator=None).eval()
    with torch.no_grad():
        base = g(x, q_override=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))["traffic"]
        low = g(x, q_override=torch.tensor([[1.0, 1.0], [0.05, 1.0]]))["traffic"]
    assert torch.allclose(base[0], low[0], atol=1e-5, rtol=1e-4)   # sample 0 untouched
    assert not torch.allclose(base[1], low[1], atol=1e-3)           # sample 1 changed


def test_constant_scaling_of_q_is_invariant_without_clamp(model):
    x = _batch(B=1)
    g = GatedInterFuser(model, estimator=None).eval()
    with torch.no_grad():
        a = g(x, q_override=torch.tensor([[0.8, 0.2]]))["traffic"]
        b = g(x, q_override=torch.tensor([[0.4, 0.1]]))["traffic"]  # same ratio -> same softmax
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_bq_ungated_output_ignores_any_q(model, monkeypatch):
    from rrf import train

    cfg = load_config("configs/pilot.yaml")
    estimator = ReliabilityEstimator(2 * model.embed_dim + 2 + 6)
    monkeypatch.setattr(train.upstream, "load_upstream_model", lambda *_args, **_kwargs: (model, {}))
    monkeypatch.setattr(train.torch, "load", lambda *_args, **_kwargs: {"estimator": estimator.state_dict()})
    monkeypatch.setattr(train, "load_reference", lambda *_args, **_kwargs: object())
    x = _batch(B=1)
    bq, _ = train.build_method(cfg, "BQ", "cpu", seed=0)
    bq.eval()
    with torch.no_grad():
        ungated = bq(x, gate_enabled=False)["traffic"]
        low_camera = bq(x, q_override=torch.tensor([[0.01, 1.0]]))["traffic"]
        low_lidar = bq(x, q_override=torch.tensor([[1.0, 0.01]]))["traffic"]
    assert bq.estimator is not None and bq.gate_enabled_by_default is False
    assert torch.equal(ungated, low_camera)
    assert torch.equal(ungated, low_lidar)


def test_unavailable_tokens_do_not_influence_output(model):
    x = _batch(B=1)
    g = GatedInterFuser(model, estimator=None).eval()
    avail = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # lidar unavailable
    x2 = dict(x)
    x2["lidar"] = torch.rand(1, 3, 224, 224) * 5  # wildly different lidar
    with torch.no_grad():
        a = g(x, availability=avail, q_override=torch.ones(1, 2))
        b = g(x2, availability=avail, q_override=torch.ones(1, 2))
        c = g(x2, q_override=torch.ones(1, 2))  # lidar available -> should differ
    assert torch.allclose(a["traffic"], b["traffic"], atol=1e-5, rtol=1e-4)
    assert torch.equal(a["stats"][:, 1], b["stats"][:, 1]) and float(a["stats"][:, 1].abs().sum()) == 0.0
    assert not torch.allclose(a["traffic"], c["traffic"], atol=1e-3)
    # front unavailable must mask both front and centre views
    kb = g.key_bias(torch.ones(1, 2), a["layout"], torch.tensor([[0.0, 1.0, 1.0, 1.0]]))
    masked = [v for v, val in zip(a["layout"], kb[0].tolist()) if val < -1e3]
    assert set(masked) == {"front", "center"} and len(masked) == 50 + 17


def test_all_unavailable_is_invalid_not_empty_road(model):
    x = _batch(B=1)
    g = GatedInterFuser(model, estimator=None).eval()
    with torch.no_grad():
        out = g(x, availability=torch.zeros(1, 4), q_override=torch.ones(1, 2))
    assert out["status"] == "ALL_SENSORS_UNAVAILABLE" and "traffic" not in out and not bool(out["prediction_valid"][0])


def test_task_loss_gradient_reaches_estimator(model):
    x = _batch(B=2)
    est = ReliabilityEstimator(stat_dim=2 * 256 + 2 + 6)
    g = GatedInterFuser(model, estimator=est)
    set_trainable(model)
    train_mode(model)
    out = g(x)
    loss = out["traffic"][..., 0].mean()
    loss.backward()
    grads = [p.grad for p in est.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(gr).all() for gr in grads)
    assert any(float(gr.abs().sum()) > 0 for gr in grads)
    assert all(p.grad is None for n, p in model.named_parameters() if n.startswith("rgb_backbone"))


def test_frozen_tokenizer_and_bn_stats_unchanged_after_step(model):
    x = _batch(B=2)
    m = copy.deepcopy(model)
    n_tr, n_tot = set_trainable(m)
    assert 0 < n_tr < n_tot
    before = {k: v.clone() for k, v in m.state_dict().items() if "running" in k or k.startswith("rgb_backbone")}
    g = GatedInterFuser(m, estimator=StaticGate())
    train_mode(m)
    opt = torch.optim.AdamW([p for p in list(m.parameters()) + list(g.estimator.parameters()) if p.requires_grad], lr=1e-3)
    out = g(x)
    out["traffic"].mean().backward()
    opt.step()
    after = m.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), k
    assert not torch.equal(after["encoder.layers.0.linear1.weight"], model.state_dict()["encoder.layers.0.linear1.weight"])


def test_metadata_never_reaches_model(model):
    """Changing privileged metadata keys with identical sensor inputs cannot change the output."""
    x = _batch(B=1)
    g = GatedInterFuser(model, estimator=ReliabilityEstimator(2 * 256 + 2 + 6)).eval()
    with torch.no_grad():
        a = g({k: x[k] for k in upstream.MODEL_INPUT_KEYS})
        polluted = dict(x, corruption="blur", severity=5, weather=3, target=torch.zeros(1, 400, 7))
        b = g({k: polluted[k] for k in upstream.MODEL_INPUT_KEYS})
    assert torch.equal(a["traffic"], b["traffic"]) and torch.equal(a["q"], b["q"])
    with pytest.raises(KeyError):
        upstream.collate([{k: x[k][0] for k in upstream.MODEL_INPUT_KEYS if k != "lidar"}])


def test_token_scaling_mechanism(model):
    x = _batch(B=2)
    g = GatedInterFuser(model, estimator=None, mechanism="token_scaling").eval()
    with torch.no_grad():
        ref = upstream.predict(model, x)["traffic"]
        one = g(x, q_override=torch.ones(2, 2))["traffic"]
        low = g(x, q_override=torch.tensor([[1.0, 1.0], [1.0, 0.3]]))["traffic"]
    assert torch.allclose(ref, one, atol=1e-5, rtol=1e-4)          # q = 1 is the identity
    assert torch.allclose(one[0], low[0], atol=1e-5, rtol=1e-4)    # batch isolation
    assert not torch.allclose(one[1], low[1], atol=1e-3)           # scaling the LiDAR tokens changes the output
    a = g(x, q_override=torch.tensor([[0.8, 0.2]] * 2))["traffic"]
    b = g(x, q_override=torch.tensor([[0.4, 0.1]] * 2))["traffic"]
    assert not torch.allclose(a, b, atol=1e-3)                     # unlike the log-prior, not ratio-invariant


def test_key_scaling_mechanism(model):
    x = _batch(B=2)
    g = GatedInterFuser(model, estimator=None, mechanism="key_scaling").eval()
    assert g.key_gamma.shape == (2,)
    with torch.no_grad():
        ref = upstream.predict(model, x)["traffic"]
        one = g(x, q_override=torch.ones(2, 2))["traffic"]
        low = g(x, q_override=torch.tensor([[1.0, 1.0], [1.0, 0.3]]))["traffic"]
        g.key_gamma.fill_(2.0)
        one2 = g(x, q_override=torch.ones(2, 2))["traffic"]
        low2 = g(x, q_override=torch.tensor([[1.0, 1.0], [1.0, 0.3]]))["traffic"]
        off = g(x, gate_enabled=False)["traffic"]
    assert torch.allclose(ref, one, atol=1e-5, rtol=1e-4) and torch.allclose(ref, one2, atol=1e-5, rtol=1e-4)  # identity for any gamma
    assert torch.allclose(ref, off, atol=1e-6, rtol=1e-5)          # patched attention is inert when unused
    assert torch.allclose(one[0], low[0], atol=1e-5, rtol=1e-4)    # batch isolation
    assert not torch.allclose(one[1], low[1], atol=1e-3)           # scaling LiDAR keys changes the output
    assert not torch.allclose(low[1], low2[1], atol=1e-3)          # gamma changes how strongly q acts
    assert "encoder.layers.0.self_attn.in_proj_weight" in model.state_dict()  # patched instances keep their state_dict keys
