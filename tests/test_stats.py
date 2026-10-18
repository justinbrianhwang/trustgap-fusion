import numpy as np
import pytest

from rrf.stats import (crossed_bootstrap_delta, episode_bootstrap_ci, holm,
                       holm_on_cis, paired_episode_bootstrap, tost, verdict)


def test_episode_bootstrap_known_constant_statistic():
    rows = [{"episode_id": ep, "value": value} for ep, value in (("a", 2.0), ("b", 2.0), ("c", 2.0))]
    result = episode_bootstrap_ci(rows, lambda rs: np.mean([r["value"] for r in rs]), n=100, seed=4)
    assert result["estimate"] == result["ci_low"] == result["ci_high"] == 2.0
    assert result["n_episodes"] == 3 and result["repeats"] == 100


def test_paired_episode_bootstrap_known_difference():
    a = [{"episode_id": ep, "value": value} for ep, value in (("a", 1.0), ("b", 3.0))]
    b = [{"episode_id": ep, "value": value + 0.5} for ep, value in (("a", 1.0), ("b", 3.0))]
    result = paired_episode_bootstrap(a, b, lambda rs: np.mean([r["value"] for r in rs]), n=100, seed=2)
    assert result["estimate"] == pytest.approx(0.5)
    assert result["ci_low"] == pytest.approx(0.5) and result["ci_high"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        paired_episode_bootstrap(a, b[:1], lambda rs: 0.0, n=2)
    mismatch = [{"episode_id": "a", "sample_id": "a/2", "value": 1.0},
                {"episode_id": "b", "sample_id": "b/1", "value": 3.0}]
    paired = [{"episode_id": "a", "sample_id": "a/1", "value": 1.0},
              {"episode_id": "b", "sample_id": "b/1", "value": 3.0}]
    with pytest.raises(ValueError):
        paired_episode_bootstrap(paired, mismatch, lambda rs: 0.0, n=2)


def test_tost_and_holm_known_answers():
    assert tost(-0.09, 0.08, 0.1)
    assert not tost(-0.1, 0.08, 0.1)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    with pytest.raises(ValueError):
        holm([1.1])


def _crossed(values):
    ids = ["ep0/0", "ep0/1", "ep1/0", "ep1/1"]
    return {
        seed: {"condition_a": {"sample_ids": ids,
                               "predictions": np.asarray(row, dtype=float),
                               "labels": np.zeros(4)}}
        for seed, row in values.items()
    }


def test_crossed_bootstrap_known_answers_and_verdicts():
    metric = lambda predictions, _labels: np.mean(predictions)
    same = _crossed({0: [1, 2, 3, 4], 1: [2, 3, 4, 5]})
    identical = crossed_bootstrap_delta(same, same, metric, n=100, seed=3)
    assert identical["delta"] == 0
    assert identical["ci90"] == [0, 0] and identical["ci95"] == [0, 0]
    assert verdict(identical["ci90"], identical["ci95"], 0.1) == "equivalent"

    shifted = _crossed({0: [2, 3, 4, 5], 1: [3, 4, 5, 6]})
    improved = crossed_bootstrap_delta(same, shifted, metric, n=100, seed=3)
    assert improved["delta"] == pytest.approx(1)
    assert verdict(improved["ci90"], improved["ci95"], 0.1) == "improves"
    assert holm_on_cis([identical, improved]) == pytest.approx([1, 0])


def test_crossed_seed_resampling_includes_seed_variance():
    metric = lambda predictions, _labels: np.mean(predictions)
    baseline = _crossed({0: [0, 0, 0, 0]})  # B0 is broadcast
    trained = _crossed({0: [-10] * 4, 1: [0] * 4, 2: [10] * 4})
    crossed = crossed_bootstrap_delta(baseline, trained, metric, n=1000, seed=8)
    # Averaging the three seeds first (incorrectly holding them fixed) leaves zero in every episode.
    fixed = paired_episode_bootstrap(
        [{"episode_id": ep, "value": 0.0} for ep in ("ep0", "ep1")],
        [{"episode_id": ep, "value": np.mean([-10, 0, 10])} for ep in ("ep0", "ep1")],
        lambda rows: np.mean([row["value"] for row in rows]), n=1000, seed=8)
    fixed_seed_ci_width = fixed["ci_high"] - fixed["ci_low"]
    assert crossed["ci95"][1] - crossed["ci95"][0] > fixed_seed_ci_width
