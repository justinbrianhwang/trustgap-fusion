"""Episode-level resampling and small multiple-testing helpers."""
from collections import defaultdict
from collections import Counter

import numpy as np

from rrf.ranked_ap import ranked_ap


def _by_episode(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["episode_id"]].append(row)
    if not groups:
        raise ValueError("rows must contain at least one episode")
    return groups


def episode_bootstrap_ci(rows, stat_fn, n=2000, seed=0):
    """Bootstrap ``stat_fn(rows)`` by resampling whole episodes with replacement."""
    groups = _by_episode(rows)
    episodes = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=np.float64)
    cache = {}
    for i in range(n):
        sampled = tuple(rng.choice(episodes, len(episodes), replace=True))
        if sampled not in cache:
            cache[sampled] = stat_fn([row for episode in sampled for row in groups[episode]])
        values[i] = cache[sampled]
    estimate = float(stat_fn(rows))
    finite = values[np.isfinite(values)]
    return {
        "estimate": estimate if np.isfinite(estimate) else None,
        "ci_low": float(np.percentile(finite, 2.5)) if len(finite) else None,
        "ci_high": float(np.percentile(finite, 97.5)) if len(finite) else None,
        "n_episodes": len(episodes),
        "repeats": int(n),
    }


def paired_episode_bootstrap(a_rows, b_rows, stat_fn, n=2000, seed=0):
    """CI for ``stat_fn(b) - stat_fn(a)`` using paired episode resamples."""
    a_rows, b_rows = list(a_rows), list(b_rows)
    a_groups, b_groups = _by_episode(a_rows), _by_episode(b_rows)
    if set(a_groups) != set(b_groups):
        raise ValueError("paired rows must contain identical episodes")
    identity_fields = [key for key in ("sample_id", "condition", "variant")
                       if all(key in row for row in a_rows + b_rows)]
    if identity_fields:
        identify = lambda row: tuple(row[key] for key in identity_fields)
        if Counter(map(identify, a_rows)) != Counter(map(identify, b_rows)):
            raise ValueError("paired rows must contain identical samples")
    episodes = sorted(a_groups)
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=np.float64)
    cache = {}
    for i in range(n):
        sampled = tuple(rng.choice(episodes, len(episodes), replace=True))
        if sampled not in cache:
            aa = [row for episode in sampled for row in a_groups[episode]]
            bb = [row for episode in sampled for row in b_groups[episode]]
            cache[sampled] = stat_fn(bb) - stat_fn(aa)
        values[i] = cache[sampled]
    estimate = float(stat_fn(b_rows) - stat_fn(a_rows))
    finite = values[np.isfinite(values)]
    return {
        "estimate": estimate if np.isfinite(estimate) else None,
        "ci_low": float(np.percentile(finite, 2.5)) if len(finite) else None,
        "ci_high": float(np.percentile(finite, 97.5)) if len(finite) else None,
        "n_episodes": len(episodes),
        "repeats": int(n),
    }


def tost(ci_low, ci_high, delta):
    """Return whether a confidence interval lies strictly inside (-delta, +delta)."""
    if delta < 0:
        raise ValueError("delta must be non-negative")
    return bool(ci_low > -delta and ci_high < delta)


def holm(pvalues):
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be a one-dimensional sequence in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def _seed_pairs(pred_a, pred_b):
    """Pair seed dictionaries, broadcasting a one-seed baseline across trained seeds."""
    a_keys, b_keys = sorted(pred_a), sorted(pred_b)
    if a_keys == b_keys:
        return [(key, key) for key in a_keys]
    if len(a_keys) == 1:
        return [(a_keys[0], key) for key in b_keys]
    if len(b_keys) == 1:
        return [(key, b_keys[0]) for key in a_keys]
    common = sorted(set(a_keys) & set(b_keys))
    if common:
        return [(key, key) for key in common]
    raise ValueError("methods must have paired seed indices unless one is a one-seed baseline")


def _validate_crossed(pred_a, pred_b):
    pairs = _seed_pairs(pred_a, pred_b)
    first = pred_a[pairs[0][0]]
    conditions = sorted(first)
    if not conditions:
        raise ValueError("at least one condition is required")
    sample_ids = list(first[conditions[0]]["sample_ids"])
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be non-empty and unique")
    for method in (pred_a, pred_b):
        for seed_data in method.values():
            if sorted(seed_data) != conditions:
                raise ValueError("all seeds and methods must contain identical conditions")
            for condition in conditions:
                payload = seed_data[condition]
                if list(payload["sample_ids"]) != sample_ids:
                    raise ValueError("all conditions, seeds, and methods must use identical sample order")
                if len(payload["predictions"]) != len(sample_ids) or len(payload["labels"]) != len(sample_ids):
                    raise ValueError("predictions and labels must align with sample_ids")
    episodes = sorted({sample_id.split("/", 1)[0] for sample_id in sample_ids})
    episode_indices = {
        episode: np.asarray([i for i, sample_id in enumerate(sample_ids)
                             if sample_id.split("/", 1)[0] == episode], dtype=np.int64)
        for episode in episodes
    }
    return pairs, conditions, episodes, episode_indices


def _take(values, indices):
    if isinstance(values, np.ndarray):
        return values[indices]
    return [values[int(index)] for index in indices]


def _crossed_stat(method, seed_keys, conditions, indices, metric_fn):
    """Condition macro-average within seed, followed by an equal-weight seed average."""
    seed_values = []
    for seed in seed_keys:
        condition_values = []
        for condition in conditions:
            payload = method[seed][condition]
            value = metric_fn(_take(payload["predictions"], indices),
                              _take(payload["labels"], indices))
            if value is None or not np.isfinite(value):
                raise ValueError(f"metric is undefined for condition {condition!r}")
            condition_values.append(float(value))
        seed_values.append(float(np.mean(condition_values)))
    return float(np.mean(seed_values))


def crossed_bootstrap_delta(pred_a, pred_b, metric_fn, n=2000, seed=0):
    """Crossed seed x episode bootstrap of ``metric(B) - metric(A)``.

    Inputs are ``seed -> condition -> {sample_ids, predictions, labels}``. A method with one
    seed is broadcast across the other method's seeds (the released-model/B0 case); otherwise only
    shared seed indices are paired. Each replicate
    independently resamples training seeds and source episodes, using the same draws for both
    methods and every corruption condition. Aggregation is fixed: pool the selected episodes within
    each seed, compute one metric per condition, equally average conditions, then average seeds.
    There is no prediction ensembling across seeds.
    """
    if n < 1:
        raise ValueError("n must be positive")
    pairs, conditions, episodes, episode_indices = _validate_crossed(pred_a, pred_b)
    original = np.arange(sum(len(episode_indices[episode]) for episode in episodes))
    a_keys, b_keys = zip(*pairs)
    delta = (_crossed_stat(pred_b, b_keys, conditions, original, metric_fn) -
             _crossed_stat(pred_a, a_keys, conditions, original, metric_fn))
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=np.float64)
    for repeat in range(n):
        selected_pairs = rng.integers(0, len(pairs), size=len(pairs))
        selected_episodes = rng.integers(0, len(episodes), size=len(episodes))
        indices = np.concatenate([episode_indices[episodes[index]] for index in selected_episodes])
        aa = [a_keys[index] for index in selected_pairs]
        bb = [b_keys[index] for index in selected_pairs]
        values[repeat] = (_crossed_stat(pred_b, bb, conditions, indices, metric_fn) -
                          _crossed_stat(pred_a, aa, conditions, indices, metric_fn))
    return {
        "delta": float(delta),
        "ci95": [float(x) for x in np.percentile(values, (2.5, 97.5))],
        "ci90": [float(x) for x in np.percentile(values, (5.0, 95.0))],
        "n_seeds": len(pairs), "n_episodes": len(episodes), "repeats": int(n),
        "bootstrap_deltas": values.tolist(),
    }


def _validate_crossed_tables(tables_a, tables_b):
    pairs = _seed_pairs(tables_a, tables_b)
    first = tables_a[pairs[0][0]]
    conditions = sorted(first)
    if not conditions:
        raise ValueError("at least one condition is required")
    reference = first[conditions[0]]
    n_episodes = reference["n_episodes"]
    episodes = tuple(reference.get("episodes", range(n_episodes)))
    sample_ids = reference.get("sample_ids")
    if len(episodes) != n_episodes:
        raise ValueError("episode list must have n_episodes entries")
    for method in (tables_a, tables_b):
        for seed_tables in method.values():
            if sorted(seed_tables) != conditions:
                raise ValueError("all seeds and methods must contain identical conditions")
            for condition in conditions:
                table = seed_tables[condition]
                table_episodes = tuple(table.get("episodes", range(table["n_episodes"])))
                if table["n_episodes"] != n_episodes or table_episodes != episodes:
                    raise ValueError("all conditions, seeds, and methods must use identical episodes")
                if sample_ids is not None and table.get("sample_ids") != sample_ids:
                    raise ValueError("all conditions, seeds, and methods must use identical samples")
    return pairs, conditions, episodes


def _crossed_table_stat(method, seed_keys, conditions, weights):
    seed_values = []
    for seed in seed_keys:
        condition_values = []
        for condition in conditions:
            value = ranked_ap(method[seed][condition], weights)
            if value is None or not np.isfinite(value):
                raise ValueError(f"metric is undefined for condition {condition!r}")
            condition_values.append(float(value))
        seed_values.append(float(np.mean(condition_values)))
    return float(np.mean(seed_values))


def crossed_bootstrap_delta_tables(tables_a, tables_b, n=2000, seed=0):
    """Crossed seed x episode bootstrap using precomputed ranked-AP tables."""
    if n < 1:
        raise ValueError("n must be positive")
    pairs, conditions, episodes = _validate_crossed_tables(tables_a, tables_b)
    a_keys, b_keys = zip(*pairs)
    original = np.ones(len(episodes), dtype=np.int64)
    delta = (_crossed_table_stat(tables_b, b_keys, conditions, original) -
             _crossed_table_stat(tables_a, a_keys, conditions, original))
    rng = np.random.default_rng(seed)
    values = np.empty(n, dtype=np.float64)
    for repeat in range(n):
        selected_pairs = rng.integers(0, len(pairs), size=len(pairs))
        selected_episodes = rng.integers(0, len(episodes), size=len(episodes))
        weights = np.bincount(selected_episodes, minlength=len(episodes))
        aa = [a_keys[index] for index in selected_pairs]
        bb = [b_keys[index] for index in selected_pairs]
        values[repeat] = (_crossed_table_stat(tables_b, bb, conditions, weights) -
                          _crossed_table_stat(tables_a, aa, conditions, weights))
    return {
        "delta": float(delta),
        "ci95": [float(x) for x in np.percentile(values, (2.5, 97.5))],
        "ci90": [float(x) for x in np.percentile(values, (5.0, 95.0))],
        "n_seeds": len(pairs), "n_episodes": len(episodes), "repeats": int(n),
        "bootstrap_deltas": values.tolist(),
    }


def verdict(ci90, ci95, delta_eq):
    """Apply the pre-registered superiority, harm, and equivalence decision rules."""
    if delta_eq < 0:
        raise ValueError("delta_eq must be non-negative")
    if ci95[0] > 0:
        return "improves"
    if ci95[1] < 0:
        return "worse"
    if ci90[0] > -delta_eq and ci90[1] < delta_eq:
        return "equivalent"
    if ci90[1] < delta_eq:
        return "rules_out_delta"
    return "inconclusive"


def holm_on_cis(results):
    """Holm-adjust two-sided bootstrap p-values recovered from crossed-bootstrap draws.

    For a positive point estimate, the one-sided tail is the fraction of bootstrap deltas <= 0;
    for a negative estimate it is the fraction >= 0. The inverted two-sided p-value is twice that
    fraction (clipped at one). A zero estimate has p=1. Returned values follow input order.
    """
    pvalues = []
    for result in results:
        values = np.asarray(result["bootstrap_deltas"], dtype=np.float64)
        if values.ndim != 1 or not len(values):
            raise ValueError("each result must contain one-dimensional bootstrap_deltas")
        if result["delta"] > 0:
            tail = np.mean(values <= 0)
        elif result["delta"] < 0:
            tail = np.mean(values >= 0)
        else:
            tail = 0.5
        pvalues.append(min(1.0, 2.0 * float(tail)))
    return holm(pvalues)
