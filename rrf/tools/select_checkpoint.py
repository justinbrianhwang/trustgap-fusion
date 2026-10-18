"""Select a saved fusion trajectory checkpoint without rerunning training."""
import argparse
import json
from pathlib import Path

from rrf.config import load_config


def read_evaluations(log_path):
    evaluations = []
    for line in Path(log_path).read_text().splitlines():
        record = json.loads(line)
        evaluation = record.get("eval")
        if isinstance(evaluation, dict) and "step" in evaluation and "summary" in evaluation:
            evaluations.append(evaluation)
    if not evaluations:
        raise ValueError(f"no evaluation summaries in {log_path}")
    return evaluations


def load_test_ap(path):
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and "per_step" in raw:
        raw = raw["per_step"]
    if isinstance(raw, dict):
        return {int(step): float(ap) for step, ap in raw.items()}
    return {int(row["step"]): float(row["test_ap"]) for row in raw}


def select_from_log(log_path, rule, max_clean_ap_drop, fixed_step=None, test_ap=None):
    evaluations = read_evaluations(log_path)
    step0 = next((e for e in evaluations if e["step"] == 0), None)
    candidates = [e for e in evaluations if e["step"] > 0]
    if not candidates:
        raise ValueError("trajectory has no post-training evaluation steps")

    result = {"rule": rule}
    if rule == "early":
        chosen = min(candidates, key=lambda e: e["step"])
    elif rule == "fixed_step":
        chosen = next((e for e in candidates if e["step"] == fixed_step), None)
        if chosen is None:
            raise ValueError(f"step {fixed_step} is not an evaluation step")
    elif rule == "oracle_test":
        if not test_ap:
            raise ValueError("oracle_test requires per-step test AP")
        available = [e for e in candidates if e["step"] in test_ap]
        if not available:
            raise ValueError("test AP JSON has no trajectory evaluation steps")
        chosen = max(available, key=lambda e: test_ap[e["step"]])
        result["test_ap"] = test_ap[chosen["step"]]
    elif rule == "val":
        if step0 is None:
            raise ValueError("val selection requires the step-0 evaluation")
        clean_floor = step0["summary"]["clean_ap"] - max_clean_ap_drop
        eligible = [e for e in candidates if e["summary"]["clean_ap"] is not None and
                    e["summary"]["clean_ap"] >= clean_floor and e["summary"]["corruption_macro_ap"] is not None]
        if eligible:
            chosen = max(eligible, key=lambda e: e["summary"]["corruption_macro_ap"])
            result["eligible"] = True
        else:
            clean = [e for e in candidates if e["summary"]["clean_ap"] is not None]
            if not clean:
                raise ValueError("trajectory has no clean AP values")
            chosen = max(clean, key=lambda e: e["summary"]["clean_ap"])
            result["eligible"] = False
        result["clean_floor"] = clean_floor
    else:
        raise ValueError(f"unknown rule: {rule}")
    return {**result, "step": chosen["step"], "summary": chosen["summary"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/journal.yaml")
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", required=True, choices=["val", "fixed_step", "early", "oracle_test"])
    ap.add_argument("--step", type=int, default=None, help="evaluation step for fixed_step")
    ap.add_argument("--test-ap-json", default=None, help="JSON mapping step to test AP for oracle_test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.rule == "fixed_step" and args.step is None:
        ap.error("--step is required for fixed_step")
    if args.rule == "oracle_test" and args.test_ap_json is None:
        ap.error("--test-ap-json is required for oracle_test")

    cfg = load_config(args.config)
    artifacts = Path(cfg["paths"]["artifacts_root"])
    run_dir = artifacts / "runs" / f"fusion_{args.method}_seed{args.seed}"
    result = select_from_log(run_dir / "log.jsonl", args.rule, cfg["metrics"]["max_clean_ap_drop"],
                             fixed_step=args.step, test_ap=load_test_ap(args.test_ap_json) if args.test_ap_json else None)
    checkpoint = artifacts / "checkpoints" / "trajectory" / f"fusion_{args.method}_seed{args.seed}_step{result['step']}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"selected trajectory checkpoint does not exist: {checkpoint}")
    result.update({"method": args.method, "seed": args.seed, "checkpoint": str(checkpoint)})
    out = Path(args.out) if args.out else run_dir / f"checkpoint_selection_{args.rule}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
