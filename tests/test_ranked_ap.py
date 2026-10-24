import numpy as np
import pytest

from rrf.metrics import grid_fg_ap
from rrf.ranked_ap import ranked_ap, ranked_ap_reference, ranked_ap_table
from rrf.stats import crossed_bootstrap_delta, crossed_bootstrap_delta_tables
from rrf.tools.journal_summary import _ranked_table


@pytest.mark.parametrize("ties", [False, True])
def test_ranked_ap_matches_reference(ties):
    rng = np.random.default_rng(12)
    scores = rng.random(300)
    if ties:
        scores = scores.astype(np.float16)
    hits = rng.random(300) < 0.2
    episode = np.repeat(np.arange(5), 60)
    denom = np.bincount(episode[hits], minlength=5)
    table = ranked_ap_table(scores, hits, episode, denom, 5)
    actual = ranked_ap(table, np.ones(5, dtype=int))
    assert actual == pytest.approx(ranked_ap_reference(scores, hits, hits.sum()))
    if not ties:
        assert actual == pytest.approx(grid_fg_ap(scores, hits))


def test_ranked_ap_integer_weights_match_duplicated_episodes():
    scores = np.array([0.8, 0.4, 0.4, 0.9, 0.4, 0.1])
    hits = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    episode = np.array([0, 0, 0, 1, 1, 1])
    weights = np.array([2, 3])
    denom = np.bincount(episode[hits], minlength=2)
    table = ranked_ap_table(scores, hits, episode, denom, 2)
    indices = np.concatenate([np.flatnonzero(episode == value).repeat(weights[value])
                              for value in range(2)])
    expected = ranked_ap_reference(scores[indices], hits[indices], denom @ weights)
    assert ranked_ap(table, weights) == pytest.approx(expected)


def _crossed(values):
    sample_ids = ["ep0/0", "ep0/1", "ep1/0", "ep1/1"]
    labels = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=bool)
    raw, tables = {}, {}
    episode = np.repeat([0, 1], 6)
    denom = np.bincount(episode[labels.ravel()], minlength=2)
    for seed, rows in values.items():
        raw[seed], tables[seed] = {}, {}
        for condition, offset in (("a", 0.0), ("b", 0.125)):
            predictions = (np.asarray(rows) + offset).astype(np.float64)
            raw[seed][condition] = {"sample_ids": sample_ids,
                                    "predictions": predictions, "labels": labels}
            table = ranked_ap_table(predictions, labels, episode, denom, 2)
            table["episodes"] = ("ep0", "ep1")
            tables[seed][condition] = table
    return raw, tables


def _tie_metric(scores, labels):
    return ranked_ap_reference(scores, labels, np.asarray(labels).sum())


@pytest.mark.parametrize("a_seeds,b_seeds", [
    ({0: [[0.9, 0.7, 0.7]] * 4, 1: [[0.8, 0.6, 0.4]] * 4},
     {0: [[0.9, 0.8, 0.7]] * 4, 1: [[0.8, 0.7, 0.5]] * 4}),
    ({0: [[0.9, 0.7, 0.7]] * 4},
     {0: [[0.9, 0.8, 0.7]] * 4, 1: [[0.8, 0.7, 0.5]] * 4,
      2: [[0.7, 0.6, 0.3]] * 4}),
])
def test_crossed_table_bootstrap_matches_raw(a_seeds, b_seeds):
    raw_a, tables_a = _crossed(a_seeds)
    raw_b, tables_b = _crossed(b_seeds)
    raw = crossed_bootstrap_delta(raw_a, raw_b, _tie_metric, n=100, seed=17)
    table = crossed_bootstrap_delta_tables(tables_a, tables_b, n=100, seed=17)
    assert table["delta"] == pytest.approx(raw["delta"], abs=1e-12)
    assert table["ci90"] == pytest.approx(raw["ci90"], abs=1e-12)
    assert table["ci95"] == pytest.approx(raw["ci95"], abs=1e-12)
    assert table["bootstrap_deltas"] == pytest.approx(raw["bootstrap_deltas"], abs=1e-12)


def test_common_ap_table_uses_matched_detections_and_episode_gt_counts():
    sample_ids = ["ep0/0", "ep1/0"]
    predictions = np.empty(2, dtype=object)
    predictions[:] = [[(0.9, 0.0, 0.0), (0.4, 4.0, 4.0)],
                      [(0.8, 1.0, 1.0)]]
    targets = np.empty(2, dtype=object)
    targets[:] = [[(0.1, 0.0)], [(1.1, 1.0), (8.0, 8.0)]]
    table = _ranked_table(
        "common_vehicle_center_ap_1m", predictions, targets, sample_ids)
    assert table["D"].tolist() == [1, 2]
    assert table["hit_episode"].tolist() == [0, 1]
    expected = ranked_ap_reference([0.9, 0.4, 0.8], [True, False, True], 3)
    assert ranked_ap(table, [1, 1]) == pytest.approx(expected)
