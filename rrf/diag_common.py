"""Model-independent helpers shared by diagnostic command-line tools."""
import csv
import json

import numpy as np

from rrf.evaluate import full_suite
from rrf.metrics import grid_fg_ap


REDUCED_NAMES = (
    "clean",
    "camera_gaussian_blur_s3",
    "camera_gaussian_blur_s5",
    "camera_darkening_s5",
    "lidar_point_dropout_s3",
    "lidar_point_dropout_s5",
    "camera_rectangular_occlusion_s5",
    "lidar_azimuth_sector_dropout_s5",
    "mixed_gaussian_blur_point_dropout_s3",
    "camera_silent_blackout",
    "lidar_silent_empty",
)


def reduced_suite(cfg_corr):
    full = full_suite(cfg_corr)
    missing = [name for name in REDUCED_NAMES if name not in full]
    if missing:
        raise RuntimeError(f"reduced-suite conditions missing from full_suite: {missing}")
    return {name: full[name] for name in REDUCED_NAMES}


def cross_episode_permutation(episode_ids, seed):
    """Return a seeded permutation whose source episode always differs from its destination."""
    episodes = np.asarray(episode_ids)
    if len(set(episodes)) < 2:
        raise ValueError("within-condition shuffle needs at least two episodes")
    rng = np.random.default_rng(seed)
    names, counts = np.unique(episodes, return_counts=True)
    shift = int(counts.max())
    if 2 * shift > len(episodes):
        raise ValueError("one episode holds more than half of the samples; no cross-episode permutation exists")
    order = np.concatenate([rng.permutation(np.flatnonzero(episodes == name))
                            for name in rng.permutation(names)])
    permutation = np.empty(len(episodes), dtype=int)
    permutation[order] = order[(np.arange(len(order)) + shift) % len(order)]
    return permutation


def write_csv(path, rows):
    """Write the union of row fields in first-seen order."""
    if not rows:
        raise RuntimeError(f"no rows produced for {path.stem}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_summary(path, summary):
    """Keep completed analyses when a diagnostic directory is reused."""
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        summary["analyses"] = {**previous.get("analyses", {}), **summary.get("analyses", {})}
    return summary


class GateXfitAccumulator:
    """Accumulate cell-split gate-oracle scores without retaining every candidate grid."""

    def __init__(self, n_cells, candidates, neutral_index, seed, mode):
        if mode not in ("interfuser", "transfuser"):
            raise ValueError("mode must be 'interfuser' or 'transfuser'")
        order = np.random.default_rng(seed).permutation(n_cells)
        self.halves = (np.sort(order[:n_cells // 2]), np.sort(order[n_cells // 2:]))
        self.candidates = list(candidates)
        self.neutral_index = neutral_index
        self.seed = seed
        self.mode = mode
        self.loss_sums = np.zeros((2, len(candidates)), dtype=np.float64)
        self.loss_counts = np.zeros(2, dtype=np.int64)
        self.scores = {name: {} for name in ("learned", "neutral", "in_sample_oracle",
                                              "cross_fitted_oracle", "best_fixed")}
        self.labels = {}

    def _loss(self, scores, target, cells):
        scores = np.asarray(scores)[..., cells]
        target = np.asarray(target)[..., cells]
        if self.mode == "interfuser":
            positive, negative = target >= 0.01, target <= 0.01
            error = np.abs(scores - target)
        else:
            positive, negative = target >= 0.99, target < 0.01
            error = (scores - positive) ** 2
        positive_error = (error * positive).sum(-1) / np.maximum(positive.sum(-1), 1)
        negative_error = (error * negative).sum(-1) / np.maximum(negative.sum(-1), 1)
        both = positive.any(-1) & negative.any(-1)
        return np.where(both, 0.5 * (positive_error + negative_error),
                        np.where(positive.any(-1), positive_error, negative_error))

    def _valid(self, target, cells):
        target = np.asarray(target)[..., cells]
        if self.mode == "interfuser":
            return np.ones_like(target, dtype=bool), target > 0.01
        valid = (target >= 0.99) | (target < 0.01)
        return valid, target >= 0.99

    @staticmethod
    def _append(store, condition, fold, values):
        store.setdefault(condition, [[], []])[fold].append(np.asarray(values).ravel())

    def add(self, condition, candidate_scores, learned_scores, target):
        """Add one batch shaped candidates x samples x cells and samples x cells."""
        candidates = np.asarray(candidate_scores).reshape(len(self.candidates), len(target), -1)
        learned = np.asarray(learned_scores).reshape(len(target), -1)
        target = np.asarray(target).reshape(len(target), -1)
        losses = [self._loss(candidates, target, cells) for cells in self.halves]
        for selection_fold, evaluation_fold in ((0, 1), (1, 0)):
            evaluation_cells = self.halves[evaluation_fold]
            selected = losses[selection_fold].argmin(0)
            in_sample = losses[evaluation_fold].argmin(0)
            sample = np.arange(len(target))
            valid, labels = self._valid(target, evaluation_cells)
            self.loss_sums[selection_fold] += losses[selection_fold].sum(1)
            self.loss_counts[selection_fold] += len(target)
            self._append(self.labels, condition, selection_fold, labels[valid])
            values = {
                "learned": learned[:, evaluation_cells],
                "neutral": candidates[self.neutral_index][:, evaluation_cells],
                "in_sample_oracle": candidates[in_sample, sample][:, evaluation_cells],
                "cross_fitted_oracle": candidates[selected, sample][:, evaluation_cells],
            }
            for name, scores in values.items():
                self._append(self.scores[name], condition, selection_fold, scores[valid])

    @property
    def best_indices(self):
        if np.any(self.loss_counts == 0):
            raise RuntimeError("no candidate batches were accumulated")
        return self.loss_sums.argmin(1)

    def add_fixed(self, condition, fold, scores, target):
        """Add scores for the fixed candidate selected on ``fold``'s selection cells."""
        evaluation_cells = self.halves[1 - fold]
        scores = np.asarray(scores).reshape(len(target), -1)[:, evaluation_cells]
        target = np.asarray(target).reshape(len(target), -1)
        valid, _ = self._valid(target, evaluation_cells)
        self._append(self.scores["best_fixed"], condition, fold, scores[valid])

    def summary(self):
        output = {"partition_seed": self.seed,
                  "half_sizes": {"A": len(self.halves[0]), "B": len(self.halves[1])},
                  "best_fixed_candidates": {
                      "A_to_B": self.candidates[int(self.best_indices[0])],
                      "B_to_A": self.candidates[int(self.best_indices[1])],
                  }, "ap": {}, "selection_bias": {}}
        conditions = sorted(self.labels)
        for name, condition_scores in self.scores.items():
            output["ap"][name] = {}
            for condition in conditions:
                fold_ap = []
                for fold in range(2):
                    labels = np.concatenate(self.labels[condition][fold])
                    scores = np.concatenate(condition_scores[condition][fold])
                    fold_ap.append(grid_fg_ap(scores, labels))
                finite = [value for value in fold_ap if value is not None]
                output["ap"][name][condition] = float(np.mean(finite)) if finite else None
        for condition in conditions:
            in_sample = output["ap"]["in_sample_oracle"][condition]
            cross_fitted = output["ap"]["cross_fitted_oracle"][condition]
            output["selection_bias"][condition] = (None if in_sample is None or cross_fitted is None
                                                     else in_sample - cross_fitted)
        return output


class GateTemporalAccumulator:
    """Accumulate a gate oracle whose current choice comes from the prior frame."""

    def __init__(self, candidates, neutral_index, mode):
        if mode not in ("interfuser", "transfuser"):
            raise ValueError("mode must be 'interfuser' or 'transfuser'")
        self.candidates = list(candidates)
        self.neutral_index = neutral_index
        self.mode = mode
        self.loss_sums = np.zeros(len(candidates), dtype=np.float64)
        self.loss_count = 0
        self.scores = {name: {} for name in ("learned", "neutral", "best_fixed",
                                              "in_sample_oracle", "temporal_oracle")}
        self.labels = {}
        self.payloads = {name: {} for name in self.scores}
        self.target_payloads = {}
        self._previous = {}
        self._agreement = {}

    @property
    def has_data(self):
        return self.loss_count > 0

    @property
    def best_index(self):
        if not self.has_data:
            raise RuntimeError("no candidate batches were accumulated")
        return int(self.loss_sums.argmin())

    @property
    def fixed_complete(self):
        return bool(self.labels) and all(
            sum(len(value) for value in self.scores["best_fixed"].get(condition, [])) ==
            sum(len(value) for value in batches)
            for condition, batches in self.labels.items())

    @staticmethod
    def _append(store, condition, values):
        store.setdefault(condition, []).append(np.asarray(values))

    def add(self, condition, sample_ids, candidate_scores, candidate_losses,
            learned_scores, target, candidate_payloads=None, learned_payloads=None,
            target_payloads=None):
        """Add an ordered batch; each episode's first choice is always neutral."""
        scores = np.asarray(candidate_scores)
        losses = np.asarray(candidate_losses)
        learned = np.asarray(learned_scores)
        target = np.asarray(target)
        sample_ids = list(sample_ids)
        if scores.shape[:2] != losses.shape or losses.shape != (len(self.candidates), len(sample_ids)):
            raise ValueError("candidate scores/losses must be candidates x samples")
        if learned.shape[0] != len(sample_ids) or target.shape[0] != len(sample_ids):
            raise ValueError("learned scores and targets must match sample_ids")

        in_sample = losses.argmin(0)
        previous = self._previous.setdefault(condition, {})
        temporal = np.empty(len(sample_ids), dtype=np.int64)
        for index, sample_id in enumerate(sample_ids):
            episode = sample_id.split("/", 1)[0]
            temporal[index] = previous.get(episode, self.neutral_index)
            previous[episode] = int(in_sample[index])
        sample = np.arange(len(sample_ids))
        variants = {
            "learned": learned,
            "neutral": scores[self.neutral_index],
            "in_sample_oracle": scores[in_sample, sample],
            "temporal_oracle": scores[temporal, sample],
        }
        for name, values in variants.items():
            self._append(self.scores[name], condition, values)
        self._append(self.labels, condition, target)
        self.loss_sums += losses.sum(1)
        self.loss_count += len(sample_ids)
        matched, count = self._agreement.get(condition, (0, 0))
        self._agreement[condition] = (matched + int((temporal == in_sample).sum()),
                                      count + len(sample_ids))

        if candidate_payloads is not None:
            payloads = np.asarray(candidate_payloads)
            selected = {
                "neutral": payloads[self.neutral_index],
                "in_sample_oracle": payloads[in_sample, sample],
                "temporal_oracle": payloads[temporal, sample],
            }
            for name, values in selected.items():
                self.payloads[name].setdefault(condition, []).extend(values)
        if learned_payloads is not None:
            self.payloads["learned"].setdefault(condition, []).extend(learned_payloads)
        if target_payloads is not None:
            self.target_payloads.setdefault(condition, []).extend(target_payloads)

    def add_fixed(self, condition, scores, payloads=None):
        """Add the globally best fixed candidate's outputs for one batch."""
        self._append(self.scores["best_fixed"], condition, scores)
        if payloads is not None:
            self.payloads["best_fixed"].setdefault(condition, []).extend(payloads)

    def summary(self):
        if not self.fixed_complete:
            raise RuntimeError("best-fixed outputs are incomplete")
        output = {
            "best_fixed_candidate": self.candidates[self.best_index],
            "best_fixed_mean_loss": self.loss_sums[self.best_index] / self.loss_count,
            "ap": {},
            "temporal_choice_equals_in_sample_fraction": {
                condition: matched / count
                for condition, (matched, count) in self._agreement.items()
            },
        }
        for name, conditions in self.scores.items():
            output["ap"][name] = {}
            for condition, batches in conditions.items():
                scores = np.concatenate(batches).reshape(-1)
                target = np.concatenate(self.labels[condition]).reshape(-1)
                if self.mode == "interfuser":
                    valid, labels = np.ones_like(target, dtype=bool), target > 0.01
                else:
                    valid = (target >= 0.99) | (target < 0.01)
                    labels = target >= 0.99
                output["ap"][name][condition] = grid_fg_ap(scores[valid], labels[valid])
        return output


__all__ = ["GateTemporalAccumulator", "GateXfitAccumulator", "REDUCED_NAMES", "cross_episode_permutation",
           "merge_summary", "reduced_suite", "write_csv"]
