"""Paired clean/corrupted evaluation of a model on stored samples.

For every (sample, condition) we run the model on the corrupted raw observation, keep the raw
traffic output (before tracker/post-processing), and compute grid-foreground metrics against
the upstream soft target. All conditions of a suite share the same samples and corruption
seeds so different methods are directly comparable.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from rrf import corruptions as C
from rrf import data, metrics, upstream

SMOKE_SUITE = {
    "clean": {},
    "camera_blur_s3": {"camera": ("gaussian_blur", 3)},
    "lidar_dropout_s3": {"lidar": ("point_dropout", 3)},
    "camera_blur_s3__lidar_dropout_s3": {"camera": ("gaussian_blur", 3), "lidar": ("point_dropout", 3)},
    "camera_silent_blackout": {"camera": ("silent_blackout", 1)},
    "lidar_silent_empty": {"lidar": ("silent_empty", 1)},
}


def full_suite(cfg_corr):
    suite = {"clean": {}}
    for mod, fams in (("camera", cfg_corr["train_families"]["camera"] + cfg_corr["heldout_families"]["camera"]),
                      ("lidar", cfg_corr["train_families"]["lidar"] + cfg_corr["heldout_families"]["lidar"])):
        for fam in fams:
            for s in cfg_corr["test_severities"]:
                suite[f"{mod}_{fam}_s{s}"] = {mod: (fam, s)}
    for a, b, s in (("gaussian_blur", "point_dropout", 3), ("gaussian_noise", "xyz_jitter", 3),
                    ("rectangular_occlusion", "azimuth_sector_dropout", 3), ("gaussian_blur", "point_dropout", 5)):
        suite[f"mixed_{a}_{b}_s{s}"] = {"camera": (a, s), "lidar": (b, s)}
    suite["camera_silent_blackout"] = {"camera": ("silent_blackout", 1)}
    suite["lidar_silent_empty"] = {"lidar": ("silent_empty", 1)}
    return suite


def condition_kind(spec, cfg_corr):
    cam, lid = spec.get("camera"), spec.get("lidar")
    if not cam and not lid:
        return "clean"
    if cam and lid:
        return "mixed"
    fam, sev = cam or lid
    if fam in C.FULL_LOSS:
        return "full_loss"
    seen_fam = fam in cfg_corr["train_families"]["camera"] + cfg_corr["train_families"]["lidar"]
    seen_sev = sev in cfg_corr["train_severities"]
    return "seen" if (seen_fam and seen_sev) else ("unseen_severity" if seen_fam else "unseen_family")


class _ConditionSamples(torch.utils.data.Dataset):
    """(sample, fixed corruption spec) -> preprocessed model inputs + audit; used with DataLoader workers."""

    def __init__(self, dataset_root, sample_ids, spec, experiment_seed):
        self.root, self.ids, self.spec, self.seed = dataset_root, list(sample_ids), spec, experiment_seed

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        cor, audit = C.corrupt_sample(data.load_raw(self.root, sid), self.spec, self.seed)
        x = upstream.preprocess(cor)
        x["sample_id"] = sid
        x["audit"] = {m: audit[m] and {k: v for k, v in audit[m].items() if k != "views"} for m in ("camera", "lidar")}
        return x


def _collate(items):
    out = {k: torch.stack([it[k] for it in items]) for k in upstream.MODEL_INPUT_KEYS}
    out["sample_id"] = [it["sample_id"] for it in items]
    out["audit"] = [it["audit"] for it in items]
    return out


@torch.no_grad()
def run_suite(model_fn, dataset_root, sample_ids, suite, experiment_seed, out_dir, batch_size=8, device="cuda",
              save_predictions=True, log_every=200, workers=4):
    """model_fn(batch_dict) -> dict with 'traffic' [B,400,7] (+ optional 'q' [B,2], 'diagnostics').
    Writes per-condition prediction .npz and a frames.jsonl; returns per-condition metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gts = {sid: data.load_gt(dataset_root, sid) for sid in sample_ids}
    labels = {sid: upstream.foreground_label(gts[sid]) for sid in sample_ids}
    results = {}
    frames_f = open(out_dir / "frames.jsonl", "w")
    for cond, spec in suite.items():
        t0 = time.time()
        preds, qs, audits, lat = {}, {}, {}, []
        loader = torch.utils.data.DataLoader(_ConditionSamples(dataset_root, sample_ids, spec, experiment_seed), batch_size=batch_size,
                                             shuffle=False, num_workers=workers, collate_fn=_collate)
        for b0, item in enumerate(loader):
            ids = item["sample_id"]
            for sid, a in zip(ids, item["audit"]):
                audits[sid] = a
            batch = {k: item[k].to(device) for k in upstream.MODEL_INPUT_KEYS}
            b0 = b0 * batch_size
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            out = model_fn(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            lat.append((time.time() - t1) / len(ids))
            traffic = out["traffic"].float().cpu().numpy()
            for k, sid in enumerate(ids):
                preds[sid] = traffic[k]
                if "q" in out:
                    qs[sid] = out["q"][k].float().cpu().numpy().tolist()
            if (b0 // batch_size) % log_every == 0:
                print(f"  [{cond}] {b0 + len(ids)}/{len(sample_ids)}", flush=True)
        scores = np.concatenate([preds[s][:, 0] for s in sample_ids])
        lab = np.concatenate([labels[s] for s in sample_ids])
        losses = [metrics.native_traffic_losses(preds[s], gts[s]) for s in sample_ids]
        per_episode = {}
        for s in sample_ids:
            per_episode.setdefault(s.split("/")[0], []).append(s)
        ep_ap = {ep: metrics.grid_fg_ap(np.concatenate([preds[s][:, 0] for s in ss]), np.concatenate([labels[s] for s in ss]))
                 for ep, ss in per_episode.items()}
        fg_present = [s for s in sample_ids if labels[s].any()]
        results[cond] = {
            "spec": spec, "n_samples": len(sample_ids), "n_fg_cells": int(lab.sum()), "n_cells": int(lab.size),
            "grid_fg_ap": metrics.grid_fg_ap(scores, lab),
            "grid_fg_ap_fg_present": metrics.grid_fg_ap(np.concatenate([preds[s][:, 0] for s in fg_present]),
                                                        np.concatenate([labels[s] for s in fg_present])) if fg_present else None,
            "per_episode_ap": ep_ap,
            "native_loss": {k: float(np.mean([l[k] for l in losses])) for k in losses[0]},
            "latency_ms_per_sample": {"median": float(np.median(lat) * 1e3), "p95": float(np.percentile(lat, 95) * 1e3)},
            "q_mean": (np.mean(list(qs.values()), axis=0).tolist() if qs else None),
            "audit_example": next(iter(audits.values())),
            "elapsed_s": round(time.time() - t0, 1),
        }
        if save_predictions:
            np.savez_compressed(out_dir / f"pred_{cond}.npz", sample_ids=np.array(sample_ids),
                                traffic=np.stack([preds[s] for s in sample_ids]),
                                q=np.stack([qs[s] for s in sample_ids]) if qs else np.zeros((0, 2)))
        for s in sample_ids:
            frames_f.write(json.dumps({"condition": cond, "sample_id": s, "episode_id": s.split("/")[0],
                                       "q": qs.get(s), "fg_cells": int(labels[s].sum()),
                                       "native_loss": metrics.native_traffic_losses(preds[s], gts[s]),
                                       "corruption": audits[s]}) + "\n")
        print(f"[{cond}] AP={results[cond]['grid_fg_ap']} fg_present_AP={results[cond]['grid_fg_ap_fg_present']} "
              f"score_l1={results[cond]['native_loss']['score']:.4f} ({results[cond]['elapsed_s']}s)", flush=True)
    frames_f.close()
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def operating_point(pred_dir_calib, pred_dir_test, target_precision, dataset_root):
    """Threshold from the clean *calibration* predictions, applied unchanged to every test condition."""
    cal = np.load(Path(pred_dir_calib) / "pred_clean.npz")
    lab = np.concatenate([upstream.foreground_label(data.load_gt(dataset_root, s)) for s in cal["sample_ids"]])
    thr = metrics.calibrate_threshold(cal["traffic"][:, :, 0].ravel(), lab, target_precision)
    out = {"threshold": thr, "target_precision": target_precision, "conditions": {}}
    if thr is None:
        out["status"] = "operating_point_unavailable"
        return out
    for f in sorted(Path(pred_dir_test).glob("pred_*.npz")):
        p = np.load(f)
        lab = np.concatenate([upstream.foreground_label(data.load_gt(dataset_root, s)) for s in p["sample_ids"]])
        out["conditions"][f.stem[5:]] = metrics.confusion(p["traffic"][:, :, 0].ravel(), lab, thr)
    return out
