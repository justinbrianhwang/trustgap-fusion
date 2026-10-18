import numpy as np
import pytest

from rrf.stats import episode_bootstrap_ci, holm, paired_episode_bootstrap, tost


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
