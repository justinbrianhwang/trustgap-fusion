"""Episode-level resampling and small multiple-testing helpers."""
from collections import defaultdict
from collections import Counter

import numpy as np


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
