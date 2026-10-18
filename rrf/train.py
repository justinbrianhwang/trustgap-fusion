"""Training stages: reference check -> reliability estimator -> fusion methods.

All fusion methods share the initial checkpoint, split, corruption schedule (seeded by
experiment seed + sample id + step, never by method), freeze policy, optimizer budget and
checkpoint-selection rule. q targets are computed on the fly from the frozen masked-upstream reference on the
*same corrupted input* the fusion model sees (no cache to go stale).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rrf import corruptions as C
from rrf import data, evaluate, metrics, upstream
from rrf.config import load_config
from rrf.gated import FUSION_PREFIXES, RAW_STAT_DIM, GatedInterFuser, ReliabilityEstimator, StaticGate, set_trainable, train_mode
from rrf.probes import q_targets, q_targets_relative

FULL_LOSS = {"camera": ("silent_blackout", 1), "lidar": ("silent_empty", 1)}


# ------------------------------------------------------------------ corruption schedule
def sample_condition(rng, cfg_corr, clean_only=False):
    if clean_only:
        return {}
    probs = cfg_corr["train_condition_probabilities"]
    kinds = ["clean", "camera_only_corrupted", "lidar_only_corrupted", "both_corrupted"]
    kind = rng.choice(kinds, p=[probs[k] for k in kinds])
    spec = {}
    single = kind in ("camera_only_corrupted", "lidar_only_corrupted")
    for mod in ("camera", "lidar"):
        if kind == "both_corrupted" or kind == f"{mod}_only_corrupted":
            fam = rng.choice(cfg_corr["train_families"][mod])
            sev = int(rng.choice(cfg_corr["train_severities"]))
            spec[mod] = (str(fam), sev)
    if single and rng.random() < cfg_corr["silent_full_loss_fraction_of_single_corruption"]:
        mod = next(iter(spec))
        spec[mod] = FULL_LOSS[mod]
    return spec


class TrainSamples(Dataset):
    """Raw sample + scheduled corruption -> model inputs, GT, foreground label, meta."""

    def __init__(self, root, sample_ids, cfg_corr, experiment_seed, epoch=0, clean_only=False, with_clean=False):
        self.root, self.ids, self.cfg_corr = root, list(sample_ids), cfg_corr
        self.seed, self.epoch, self.clean_only, self.with_clean = experiment_seed, epoch, clean_only, with_clean

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        rng = np.random.default_rng(C.corruption_seed(self.seed, sid, "schedule", self.epoch, 0))
        spec = sample_condition(rng, self.cfg_corr, self.clean_only)
        raw = data.load_raw(self.root, sid)
        cor, _ = C.corrupt_sample(raw, spec, self.seed, repeat=self.epoch)
        x = upstream.preprocess(cor)
        if self.with_clean:  # clean counterpart, used only to build the relative quality target
            for k, v in upstream.preprocess(raw).items():
                x["clean_" + k] = v
        gt = torch.from_numpy(data.load_gt(self.root, sid))
        x["target"] = gt
        x["fg"] = torch.from_numpy(upstream.foreground_label(gt.numpy()))
        x["sample_id"] = sid
        x["condition"] = C.condition_name(spec)
        return x


def collate(items):
    keys = list(upstream.MODEL_INPUT_KEYS) + ["target", "fg"] + [k for k in items[0] if k.startswith("clean_")]
    out = {k: torch.stack([it[k] for it in items]) for k in keys}
    out["sample_id"] = [it["sample_id"] for it in items]
    out["condition"] = [it["condition"] for it in items]
    return out


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def infinite_loader(make_dataset, batch_size, workers, seed):
    epoch = 0
    while True:
        ds = make_dataset(epoch)
        g = torch.Generator().manual_seed(seed + epoch)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate,
                        generator=g, drop_last=True, persistent_workers=False)
        for b in dl:
            yield b
        epoch += 1


def native_traffic_loss(output, target):
    """Upstream MVTL1Loss (L1) combined as train.py does: 0.5*traffic + 0.05*velocity (aux heads off)."""
    l1 = torch.nn.L1Loss()
    m1, m0 = target[..., 0].ge(0.01), target[..., 0].le(0.01)
    zero = output.new_zeros(())
    lp1 = l1(output[..., 0][m1], target[..., 0][m1]) if m1.any() else zero
    lp0 = l1(output[..., 0][m0], target[..., 0][m0]) if m0.any() else zero
    geo = l1(output[m1][:, 1:6], target[m1][:, 1:6]) if m1.any() else zero
    vel = l1(output[m1][:, 6], target[m1][:, 6]) if m1.any() else zero
    traffic = 0.5 * (0.5 * lp0 + 0.5 * lp1) + 0.5 * geo
    return 0.5 * traffic + 0.05 * vel, {"score": float(0.5 * lp0 + 0.5 * lp1), "geometry": float(geo), "speed": float(vel)}


# ------------------------------------------------------------------ helpers
def load_split(cfg):
    p = Path(cfg["paths"]["artifacts_root"]) / "manifests" / "split.json"
    return json.loads(p.read_text())


def ckpt_dir(cfg):
    d = Path(cfg["paths"]["artifacts_root"]) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_reference(cfg, device):
    """Frozen upstream model wrapped for masked (camera-only / lidar-only) reference forwards."""
    model, _ = upstream.load_upstream_model(cfg["paths"]["upstream_checkpoint"], device)
    for p in model.parameters():
        p.requires_grad = False
    return GatedInterFuser(model, None).to(device).eval()


def clean_batch(batch):
    return {k: batch["clean_" + k] for k in upstream.MODEL_INPUT_KEYS}


def quality_targets(reference, batch, fg, cfg, target):
    """q* for the batch: 'absolute' = exp(-E_corrupt/tau) (md 8.5); 'relative' = exp(-max(E_corrupt-E_clean,0)/tau_relative)."""
    if target == "relative":
        q, e_rel, _, _ = q_targets_relative(reference, batch, clean_batch(batch), fg, cfg["reliability"]["tau_relative"])
        return q, e_rel
    return q_targets(reference, batch, fg, cfg["reliability"]["tau_quality"])


GATED_METHODS = ("P", "BQ", "PG", "PR", "PS", "PGS", "PT", "PGT")  # trained reliability + quality loss; BQ keeps its gate disabled
PRETRAINED_ROUTER_METHODS = GATED_METHODS + ("B3A",)
GATE_ONLY = ("PG", "PGS", "PGT")
MECHANISM = {"PS": "token_scaling", "PGS": "token_scaling", "PT": "key_scaling", "PGT": "key_scaling"}


def gate_state(gate):
    return {"key_gamma": gate.key_gamma.detach().cpu().clone()} if gate.key_gamma is not None else None


def load_gate_state(gate, state):
    if state and gate.key_gamma is not None:
        gate.key_gamma.data.copy_(state["key_gamma"].to(gate.key_gamma.device))


def rel_ckpt_name(seed, target):
    return f"reliability_seed{seed}.pt" if target == "absolute" else f"reliability_rel_seed{seed}.pt"


def log_line(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def fusion_objective(task, q, q_star, foreground, quality_weight, use_quality):
    """Return task plus optional foreground quality loss and the unweighted quality term."""
    if not use_quality or quality_weight <= 0 or not foreground.any():
        return task, None
    quality = torch.nn.functional.smooth_l1_loss(
        q[foreground], q_star[foreground].detach())
    return task + quality_weight * quality, quality


# ------------------------------------------------------------------ stage: reference
@torch.no_grad()
def stage_reference(cfg, args, device):
    """Report the masked-upstream reference quality (AP/Brier per modality) on validation
    under clean and seen corruptions, and its invariance to the other modality's corruption."""
    split = load_split(cfg)
    root = cfg["paths"]["dataset_root"]
    reference = load_reference(cfg, device)
    val_ids = data.split_samples(root, split, "validation", "model_select_groups", max_frames=300, seed=1)
    suite = {"clean": {}, "camera_gaussian_blur_s3": {"camera": ("gaussian_blur", 3)}, "camera_darkening_s3": {"camera": ("darkening", 3)},
             "lidar_point_dropout_s3": {"lidar": ("point_dropout", 3)}, "lidar_xyz_jitter_s3": {"lidar": ("xyz_jitter", 3)},
             "camera_silent_blackout": {"camera": ("silent_blackout", 1)}, "lidar_silent_empty": {"lidar": ("silent_empty", 1)}}
    run_dir = Path(cfg["paths"]["artifacts_root"]) / "runs" / "reference"
    report = {}
    for name, avail in (("camera_only", [1, 1, 1, 0]), ("lidar_only", [0, 0, 0, 1])):
        a = torch.tensor([avail], dtype=torch.float32, device=device)
        res = evaluate.run_suite(lambda b: reference(b, availability=a.expand(b["rgb"].shape[0], 4), gate_enabled=False), root, val_ids, suite,
                                 args.seed + 999, run_dir / name, batch_size=16, device=device, save_predictions=False, log_every=10**9)
        report[name] = {c: {"ap": r["grid_fg_ap"], "ap_fg_present": r["grid_fg_ap_fg_present"], "score_l1": r["native_loss"]["score"]} for c, r in res.items()}
    report["n_samples"] = len(val_ids)
    report["note"] = "reference = frozen upstream with the other modality hard-masked; scores are upstream sigmoid outputs, not calibrated probabilities"
    (run_dir / "eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# ------------------------------------------------------------------ stage: reliability
def stage_reliability(cfg, args, device):
    split = load_split(cfg)
    root = cfg["paths"]["dataset_root"]
    rc = cfg["reliability"]
    torch.manual_seed(args.seed)
    model, _ = upstream.load_upstream_model(cfg["paths"]["upstream_checkpoint"], device)
    for p in model.parameters():
        p.requires_grad = False
    reference = load_reference(cfg, device)
    gate = GatedInterFuser(model, estimator=None, add_raw_statistics=rc["add_raw_statistics"]).to(device)
    train_ids = data.split_samples(root, split, "train")  # reference is frozen/untrained: no probe holdout needed
    val_ids = data.split_samples(root, split, "validation", "model_select_groups", max_frames=300, seed=1)
    # 1) fit the input normalizer on a pass over training statistics (train split only)
    est = ReliabilityEstimator(stat_dim=2 * model.embed_dim + 2 + RAW_STAT_DIM, hidden=rc["estimator_hidden_dim"], layers=rc["estimator_hidden_layers"]).to(device)
    ds0 = TrainSamples(root, train_ids, cfg["corruption"], args.seed, epoch=0, with_clean=(args.target == "relative"))
    dl0 = DataLoader(ds0, batch_size=16, shuffle=False, num_workers=cfg["training"]["workers"], collate_fn=collate)
    stats = []
    with torch.no_grad():
        for i, batch in enumerate(dl0):
            batch = to_device(batch, device)
            stats.append(gate(batch, gate_enabled=False)["stats"].cpu())
            if i * 16 >= 800:
                break
    est.fit_normalizer(torch.cat(stats).to(device))
    opt = torch.optim.AdamW(est.parameters(), lr=rc["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    loader = infinite_loader(lambda ep: TrainSamples(root, train_ids, cfg["corruption"], args.seed, ep, with_clean=(args.target == "relative")), rc["batch_size"], cfg["training"]["workers"], args.seed)
    run_dir = Path(cfg["paths"]["artifacts_root"]) / "runs" / rel_ckpt_name(args.seed, args.target)[:-3]
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "log.jsonl"
    best = math.inf
    t0 = time.time()
    for step in range(1, rc["steps"] + 1):
        batch = to_device(next(loader), device)
        est.train()
        with torch.no_grad():
            avail = torch.ones(batch["rgb"].shape[0], 4, device=device)
            st = gate(batch, gate_enabled=False)["stats"]
            q_star, _ = quality_targets(reference, batch, batch["fg"], cfg, args.target)
        q_hat = est(st, avail)
        fg_present = batch["fg"].any(1)  # q* is only informative where the task has foreground
        if not fg_present.any():
            continue
        loss = torch.nn.functional.smooth_l1_loss(q_hat[fg_present], q_star[fg_present].detach())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(est.parameters(), cfg["training"]["gradient_clip_norm"])
        opt.step()
        if step % 50 == 0:
            rec = {"step": step, "loss": float(loss), "q_hat_mean": q_hat.mean(0).tolist(), "q_star_mean": q_star.mean(0).tolist(), "elapsed_s": round(time.time() - t0)}
            log_line(log, rec)
            print(rec, flush=True)
        if step % cfg["training"]["evaluation_interval_steps"] == 0 or step == rc["steps"]:
            ev = eval_reliability(gate, reference, est, root, val_ids, cfg, args.seed, device, args.target)
            ev["step"] = step
            log_line(log, {"eval": ev})
            print({"eval": ev}, flush=True)
            crit = ev["foreground_present"].get("mae", ev["overall"]["mae"])  # select on informative frames
            if crit < best:
                best = crit
                torch.save({"estimator": est.state_dict(), "meta": {"step": step, "eval": ev, "seed": args.seed, "target": args.target,
                                                                    "config_sha256": cfg["_meta"]["config_sha256"]}},
                           ckpt_dir(cfg) / rel_ckpt_name(args.seed, args.target))


@torch.no_grad()
def eval_reliability(gate, reference, est, root, val_ids, cfg, seed, device, target="absolute"):
    from scipy.stats import spearmanr
    est.eval()
    ds = TrainSamples(root, val_ids, cfg["corruption"], seed + 999, epoch=0, with_clean=(target == "relative"))
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=cfg["training"]["workers"], collate_fn=collate)
    qh, qs, es, fg_present, conds = [], [], [], [], []
    for batch in dl:
        batch = to_device(batch, device)
        avail = torch.ones(batch["rgb"].shape[0], 4, device=device)
        st = gate(batch, gate_enabled=False)["stats"]
        qh.append(est(st, avail).cpu())
        q_star, e = quality_targets(reference, batch, batch["fg"], cfg, target)
        es.append(e.cpu())
        qs.append(q_star.cpu())
        fg_present.append(batch["fg"].any(1).cpu())
        conds += batch["condition"]
    qh, qs, es, fgp = torch.cat(qh).numpy(), torch.cat(qs).numpy(), torch.cat(es).numpy(), torch.cat(fg_present).numpy()

    def block(mask):
        if mask.sum() < 3:
            return {"n": int(mask.sum())}
        gap = np.abs(qs[mask, 0] - qs[mask, 1]) >= 0.05
        rank_ok = ((qh[mask, 0] > qh[mask, 1]) == (qs[mask, 0] > qs[mask, 1]))[gap]
        return {"n": int(mask.sum()), "mae": float(np.abs(qh[mask] - qs[mask]).mean()),
                "mae_camera": float(np.abs(qh[mask, 0] - qs[mask, 0]).mean()), "mae_lidar": float(np.abs(qh[mask, 1] - qs[mask, 1]).mean()),
                "spearman_camera": float(spearmanr(qh[mask, 0], -es[mask, 0]).correlation),
                "spearman_lidar": float(spearmanr(qh[mask, 1], -es[mask, 1]).correlation),
                "ranking_accuracy": float(rank_ok.mean()) if len(rank_ok) else None, "ranking_pairs": int(len(rank_ok)),
                "q_hat_mean": qh[mask].mean(0).tolist(), "q_star_mean": qs[mask].mean(0).tolist(), "q_hat_std": qh[mask].std(0).tolist()}
    out = {"overall": block(np.ones(len(qh), bool)), "foreground_present": block(fgp), "foreground_absent": block(~fgp)}
    by_cond = {}
    for c in sorted(set(conds)):
        m = np.array([x == c for x in conds])
        by_cond[c] = {"n": int(m.sum()), "q_hat_mean": qh[m].mean(0).tolist(), "q_star_mean": qs[m].mean(0).tolist()}
    out["by_condition"] = by_cond
    return out


# ------------------------------------------------------------------ stage: fusion
def build_method(cfg, method, device, seed):
    model, _ = upstream.load_upstream_model(cfg["paths"]["upstream_checkpoint"], device)
    fc = cfg["fusion"]
    stat_dim = 2 * model.embed_dim + 2 + RAW_STAT_DIM
    est, reference = None, None
    if method in PRETRAINED_ROUTER_METHODS:  # B3A matches P except that its quality loss is disabled
        ck = torch.load(ckpt_dir(cfg) / rel_ckpt_name(seed, "relative" if method == "PR" else "absolute"), map_location=device, weights_only=False)
        est = ReliabilityEstimator(stat_dim, cfg["reliability"]["estimator_hidden_dim"], cfg["reliability"]["estimator_hidden_layers"]).to(device)
        est.load_state_dict(ck["estimator"])
        reference = load_reference(cfg, device)
    elif method == "B3":
        est = ReliabilityEstimator(stat_dim, cfg["reliability"]["estimator_hidden_dim"], cfg["reliability"]["estimator_hidden_layers"]).to(device)
        ck = torch.load(ckpt_dir(cfg) / f"reliability_seed{seed}.pt", map_location=device, weights_only=False)
        est.mu.copy_(ck["estimator"]["mu"]); est.sigma.copy_(ck["estimator"]["sigma"])  # same input normalization, random MLP
    elif method == "B4":
        est = StaticGate().to(device)
    gate = GatedInterFuser(model, est, alpha=fc["alpha"], eps=fc["epsilon"], apply_encoder=fc["apply_encoder_self_attention"],
                           apply_decoder=fc["apply_decoder_cross_attention"], add_raw_statistics=cfg["reliability"]["add_raw_statistics"],
                           mechanism=MECHANISM.get(method, "log_prior"), gate_enabled_by_default=(method != "BQ")).to(device)
    return gate, reference


def stage_fusion(cfg, args, device):
    split = load_split(cfg)
    root = cfg["paths"]["dataset_root"]
    tc = cfg["training"]
    method = args.method
    torch.manual_seed(args.seed)
    gate, reference = build_method(cfg, method, device, args.seed)
    n_tr, n_tot = set_trainable(gate.model, train_prefixes=() if method in GATE_ONLY else FUSION_PREFIXES)
    params = [{"params": [p for p in gate.model.parameters() if p.requires_grad], "lr": tc["learning_rate"]}] if n_tr else []
    if gate.estimator is not None:
        params.append({"params": list(gate.estimator.parameters()), "lr": cfg["reliability"]["learning_rate"]})
    if gate.key_gamma is not None:
        params.append({"params": [gate.key_gamma], "lr": cfg["reliability"]["learning_rate"]})
    opt = torch.optim.AdamW(params, weight_decay=tc["weight_decay"])
    train_ids = data.split_samples(root, split, "train")
    val_ids = data.split_samples(root, split, "validation", "model_select_groups", max_frames=200, seed=1)
    target_kind = "relative" if method == "PR" else "absolute"
    loader = infinite_loader(lambda ep: TrainSamples(root, train_ids, cfg["corruption"], args.seed, ep, clean_only=(method == "B1"), with_clean=(method == "PR")),
                             tc["batch_size"], tc["workers"], args.seed)
    run_dir = Path(cfg["paths"]["artifacts_root"]) / "runs" / f"fusion_{method}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "log.jsonl"
    (run_dir / "setup.json").write_text(json.dumps({"method": method, "seed": args.seed, "trainable_params": n_tr, "total_params": n_tot,
                                                     "estimator_params": sum(p.numel() for p in gate.estimator.parameters()) if gate.estimator else 0,
                                                     "effective_batch": tc["batch_size"] * tc["gradient_accumulation_steps"],
                                                     "optimizer_updates": tc["fusion_steps"], "config_sha256": cfg["_meta"]["config_sha256"]}, indent=2))
    # reference for the clean-drop rule: the initial upstream model (gate disabled = B0) on the
    # same validation samples, identical for every method
    ref = eval_fusion(gate, root, val_ids, cfg, args.seed, device, gate_enabled=False)
    log_line(log, {"eval": {"step": 0, **ref}})
    print({"eval_step0": ref["summary"]}, flush=True)
    clean_floor = ref["summary"]["clean_ap"] - cfg["metrics"]["max_clean_ap_drop"]
    best, best_rec = -math.inf, None
    fallback = {"clean_ap": -math.inf, "state": None, "rec": None}  # used only if no checkpoint meets the clean-drop rule
    accum = tc["gradient_accumulation_steps"]
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    for step in range(1, tc["fusion_steps"] + 1):
        train_mode(gate.model)
        if gate.estimator is not None:
            gate.estimator.train()
        rec = {"step": step, "loss": 0.0, "task": 0.0, "quality": 0.0}
        opt.zero_grad()
        for _ in range(accum):
            batch = to_device(next(loader), device)
            out = gate(batch)
            task, parts = native_traffic_loss(out["traffic"], batch["target"])
            q_star = None
            if method in GATED_METHODS and tc["lambda_quality"] > 0:
                q_star, _ = quality_targets(reference, batch, batch["fg"], cfg, target_kind)
            loss, lq = fusion_objective(
                task, out["q"], q_star, batch["fg"].any(1), tc["lambda_quality"],
                method in GATED_METHODS)
            if lq is not None:
                rec["quality"] += float(lq) / accum
            (loss / accum).backward()
            rec["loss"] += float(loss) / accum
            rec["task"] += float(task) / accum
        torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], tc["gradient_clip_norm"])
        opt.step()
        if step % 25 == 0:
            rec["elapsed_s"] = round(time.time() - t0)
            if out.get("q") is not None:
                rec["q_mean"] = out["q"].mean(0).tolist()
            if gate.key_gamma is not None:
                rec["key_gamma"] = gate.key_gamma.detach().cpu().tolist()
            log_line(log, rec)
            print(rec, flush=True)
        if step % tc["evaluation_interval_steps"] == 0 or step == tc["fusion_steps"]:
            ev = eval_fusion(gate, root, val_ids, cfg, args.seed, device)
            ev["step"] = step
            ev["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None
            log_line(log, {"eval": ev})
            print({"eval": {"step": step, **ev["summary"]}}, flush=True)
            if args.keep_all_checkpoints:
                trajectory = ckpt_dir(cfg) / "trajectory"
                trajectory.mkdir(parents=True, exist_ok=True)
                torch.save({"model": gate.model.state_dict(), "estimator": gate.estimator.state_dict() if gate.estimator else None, "gate": gate_state(gate),
                            "meta": {"method": method, "seed": args.seed, "step": step, "eval": ev["summary"], "config_sha256": cfg["_meta"]["config_sha256"],
                                     "key_gamma": gate.key_gamma.detach().cpu().tolist() if gate.key_gamma is not None else None}},
                           trajectory / f"fusion_{method}_seed{args.seed}_step{step}.pt")
            eligible = ev["summary"]["clean_ap"] is not None and ev["summary"]["clean_ap"] >= clean_floor
            score = ev["summary"]["corruption_macro_ap"]
            if ev["summary"]["clean_ap"] is not None and ev["summary"]["clean_ap"] > fallback["clean_ap"]:
                fallback = {"clean_ap": ev["summary"]["clean_ap"], "rec": {"step": step, **ev["summary"]},
                            "state": ({k: v.detach().cpu().clone() for k, v in gate.model.state_dict().items()},
                                      {k: v.detach().cpu().clone() for k, v in gate.estimator.state_dict().items()} if gate.estimator else None,
                                      gate_state(gate))}
            if eligible and score is not None and score > best:
                best, best_rec = score, {"step": step, **ev["summary"]}
                torch.save({"model": gate.model.state_dict(), "estimator": gate.estimator.state_dict() if gate.estimator else None, "gate": gate_state(gate),
                            "meta": {"method": method, "seed": args.seed, "step": step, "eval": ev["summary"], "config_sha256": cfg["_meta"]["config_sha256"],
                                     "key_gamma": gate.key_gamma.detach().cpu().tolist() if gate.key_gamma is not None else None}},
                           ckpt_dir(cfg) / f"fusion_{method}_seed{args.seed}.pt")
    final = {"method": method, "seed": args.seed, "selected": best_rec, "clean_floor": clean_floor, "step0": ref["summary"],
             "selection_rule": "max validation corruption macro-AP among checkpoints with clean AP >= step0 clean AP - max_clean_ap_drop",
             "wall_seconds": round(time.time() - t0), "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None}
    if best_rec is None:
        final["status"] = "no checkpoint satisfied the clean-drop rule; saved the highest-clean-AP checkpoint flagged ineligible"
        final["fallback"] = fallback["rec"]
        torch.save({"model": fallback["state"][0], "estimator": fallback["state"][1], "gate": fallback["state"][2],
                    "meta": {"method": method, "seed": args.seed, "step": fallback["rec"]["step"], "eval": fallback["rec"], "eligible": False,
                             "key_gamma": fallback["state"][2]["key_gamma"].tolist() if fallback["state"][2] else None,
                             "config_sha256": cfg["_meta"]["config_sha256"]}}, ckpt_dir(cfg) / f"fusion_{method}_seed{args.seed}.pt")
    (run_dir / "final.json").write_text(json.dumps(final, indent=2))
    print(final, flush=True)


VAL_SUITE_FAMILIES = {"camera": ["gaussian_noise", "gaussian_blur", "darkening"], "lidar": ["point_dropout", "xyz_jitter"]}


def validation_suite():
    suite = {"clean": {}}
    for mod, fams in VAL_SUITE_FAMILIES.items():
        for f in fams:
            suite[f"{mod}_{f}_s2"] = {mod: (f, 2)}
    return suite


@torch.no_grad()
def eval_fusion(gate, root, val_ids, cfg, seed, device, gate_enabled=True):
    gate.model.eval()
    if gate.estimator is not None:
        gate.estimator.eval()
    tmp = Path(cfg["paths"]["artifacts_root"]) / "runs" / "_val_tmp"
    res = evaluate.run_suite(lambda b: gate(b, gate_enabled=gate_enabled), root, val_ids, validation_suite(), seed + 777, tmp,
                             batch_size=16, device=device, save_predictions=False, log_every=10**9)
    aps = {c: r["grid_fg_ap"] for c, r in res.items()}
    corr = [v for c, v in aps.items() if c != "clean" and v is not None]
    return {"summary": {"clean_ap": aps["clean"], "corruption_macro_ap": float(np.mean(corr)) if corr else None, "per_condition_ap": aps,
                        "q_mean": {c: r["q_mean"] for c, r in res.items()}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--stage", required=True, choices=["reference", "reliability", "fusion"])
    ap.add_argument("--method", default="P", choices=["B1", "B2", "B3", "B3A", "B4", "P", "BQ", "PG", "PR", "PS", "PGS", "PT", "PGT"])
    ap.add_argument("--target", default="absolute", choices=["absolute", "relative"], help="reliability stage: quality target definition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="override the stage's step count (smoke runs)")
    ap.add_argument("--keep-all-checkpoints", action="store_true", help="save every fusion evaluation checkpoint under checkpoints/trajectory")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.steps:
        cfg["reliability"]["steps"] = cfg["training"]["fusion_steps"] = args.steps
        cfg["training"]["evaluation_interval_steps"] = min(cfg["training"]["evaluation_interval_steps"], args.steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    {"reference": stage_reference, "reliability": stage_reliability, "fusion": stage_fusion}[args.stage](cfg, args, device)


if __name__ == "__main__":
    main()
