"""Compact episode tables for tie-grouped ranked average precision."""
import numpy as np


def ranked_ap_table(scores, hits, episode_index, denom_per_episode, n_episodes):
    """Precompute threshold counts needed for episode-weighted ranked AP."""
    scores = np.asarray(scores).ravel()
    hits = np.asarray(hits, dtype=bool).ravel()
    episode_index = np.asarray(episode_index, dtype=np.int64).ravel()
    denom = np.asarray(denom_per_episode, dtype=np.int64).ravel()
    if len(scores) != len(hits) or len(scores) != len(episode_index):
        raise ValueError("scores, hits, and episode_index must have equal lengths")
    if len(denom) != n_episodes:
        raise ValueError("denom_per_episode must have n_episodes entries")
    if np.any((episode_index < 0) | (episode_index >= n_episodes)):
        raise ValueError("episode_index is out of range")

    hit_scores = scores[hits]
    hit_episode = episode_index[hits]
    h = np.empty((len(hit_scores), n_episodes), dtype=np.int32)  # per-episode counts fit
    a = np.empty_like(h)
    for episode in range(n_episodes):
        selected = episode_index == episode
        episode_scores = np.sort(scores[selected])
        episode_hits = np.sort(scores[selected & hits])
        a[:, episode] = len(episode_scores) - np.searchsorted(
            episode_scores, hit_scores, side="left")
        h[:, episode] = len(episode_hits) - np.searchsorted(
            episode_hits, hit_scores, side="left")
    return {"H": h, "A": a, "hit_episode": hit_episode, "D": denom,
            "n_episodes": int(n_episodes)}


def ranked_ap(table, weights):
    """Evaluate a ranked-AP table with integer episode multiplicities."""
    weights = np.asarray(weights, dtype=np.int64).ravel()
    if len(weights) != table["n_episodes"]:
        raise ValueError("weights must have n_episodes entries")
    denominator = int(table["D"] @ weights)
    if denominator == 0:
        return None
    selected = weights[table["hit_episode"]] > 0
    if not np.any(selected):
        return 0.0
    tp = table["H"][selected] @ weights
    total = table["A"][selected] @ weights
    contribution = weights[table["hit_episode"][selected]] * tp / total
    return float(np.sum(contribution, dtype=np.float64) / denominator)


def ranked_ap_reference(scores, hits, denom):
    """Direct tie-grouped ranked AP used as a small-data reference."""
    scores = np.asarray(scores).ravel()
    hits = np.asarray(hits, dtype=bool).ravel()
    if len(scores) != len(hits):
        raise ValueError("scores and hits must have equal lengths")
    if denom == 0:
        return None
    if not np.any(hits):
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_hits = hits[order]
    ends = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    end_indices = np.flatnonzero(ends)
    starts = np.r_[0, end_indices[:-1] + 1]
    group_hits = np.add.reduceat(ordered_hits, starts)
    precision = np.cumsum(ordered_hits)[end_indices] / (end_indices + 1)
    return float(np.sum(group_hits * precision, dtype=np.float64) / denom)
