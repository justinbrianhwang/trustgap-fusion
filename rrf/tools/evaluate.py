"""Evaluate a method on a split with a corruption suite (smoke / full).

  python -m rrf.tools.evaluate --method B0 --suite smoke --split validation
Outputs artifacts/metrics/<method>_seed<seed>_<suite>_<split>/{metrics.json, frames.jsonl, pred_*.npz}
"""
import argparse
import json
from pathlib import Path

import torch

from rrf import data, evaluate, upstream
from rrf.config import load_config
from rrf.gated import GatedInterFuser


def load_method(cfg, method, seed, device):
    """B0 = upstream checkpoint; others = fusion_<method>_seed<seed>.pt from the training stage."""
    from rrf.train import build_method, ckpt_dir, load_gate_state
    if method == "B0":
        model, _ = upstream.load_upstream_model(cfg["paths"]["upstream_checkpoint"], device)
        gate = GatedInterFuser(model, None).to(device)
        return gate, {"checkpoint": cfg["paths"]["upstream_checkpoint"], "checkpoint_sha256": upstream.sha256_file(cfg["paths"]["upstream_checkpoint"])}
    gate, _ = build_method(cfg, method, device, seed)
    path = ckpt_dir(cfg) / f"fusion_{method}_seed{seed}.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    gate.model.load_state_dict(ck["model"])
    if gate.estimator is not None and ck["estimator"] is not None:
        gate.estimator.load_state_dict(ck["estimator"])
    load_gate_state(gate, ck.get("gate"))
    return gate, {"checkpoint": str(path), "checkpoint_sha256": upstream.sha256_file(path), "meta": ck["meta"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--method", default="B0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suite", default="smoke", choices=["smoke", "full", "validation", "clean"])
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--subgroup", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = cfg["paths"]["dataset_root"]
    split = json.loads((Path(cfg["paths"]["artifacts_root"]) / "manifests" / "split.json").read_text())
    ids = data.split_samples(root, split, args.split, args.subgroup, max_frames=args.max_samples, seed=1)
    if args.suite == "smoke":
        suite = evaluate.SMOKE_SUITE
    elif args.suite == "clean":
        suite = {"clean": {}}
    elif args.suite == "validation":
        from rrf.train import validation_suite
        suite = validation_suite()
    else:
        suite = evaluate.full_suite(cfg["corruption"])
    gate, prov = load_method(cfg, args.method, args.seed, device)
    gate.model.eval()
    if gate.estimator is not None:
        gate.estimator.eval()
    tag = f"{args.split}" + (f"-{args.subgroup}" if args.subgroup else "")
    out_dir = Path(cfg["paths"]["artifacts_root"]) / "metrics" / f"{args.method}_seed{args.seed}_{args.suite}_{tag}"
    print(f"evaluating {args.method} on {len(ids)} samples x {len(suite)} conditions -> {out_dir}")
    res = evaluate.run_suite(lambda b: gate(b), root, ids, suite, cfg["corruption"]["base_seed"], out_dir, batch_size=args.batch_size, device=device)
    for c in res:
        res[c]["kind"] = evaluate.condition_kind(res[c]["spec"], cfg["corruption"])
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    summary = {"method": args.method, "seed": args.seed, "suite": args.suite, "split": args.split, "n_samples": len(ids),
               "episodes": sorted({s.split('/')[0] for s in ids}), "provenance": prov, "config_sha256": cfg["_meta"]["config_sha256"],
               "split_sha256": upstream.sha256_file(Path(cfg["paths"]["artifacts_root"]) / "manifests" / "split.json"),
               "ap": {c: r["grid_fg_ap"] for c, r in res.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["ap"], indent=2))


if __name__ == "__main__":
    main()
