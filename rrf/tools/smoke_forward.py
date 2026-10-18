"""Synthetic-tensor smoke test: strict checkpoint load, forward, token layout, contract dump.

This is NOT a perception validation (inputs are synthetic). It verifies the vendored code
runs in the rrf env and records artifacts/tensor_contract.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rrf import ROOT
from rrf.config import load_config
from rrf.upstream import collate, load_upstream_model, predict, preprocess, token_layout


def synthetic_raw(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-30, 30, size=(20000, 4)).astype(np.float32)
    pts[:, 2] = rng.uniform(-2.4, 1.0, size=20000)
    return {
        "rgb_front": rng.integers(0, 255, size=(600, 800, 3), dtype=np.uint8),
        "rgb_left": rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8),
        "rgb_right": rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8),
        "lidar": pts,
        "speed": 3.0,
        "command": 4,
        "target_point": np.array([0.0, -10.0], dtype=np.float32),
    }


def tensor_stats(t):
    t = t.detach().float()
    return {"shape": list(t.shape), "dtype": "float32", "min": float(t.min()), "max": float(t.max()),
            "nan": int(torch.isnan(t).sum()), "inf": int(torch.isinf(t).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    model, report = load_upstream_model(cfg["paths"]["upstream_checkpoint"], device)
    batch = collate([preprocess(synthetic_raw(0)), preprocess(synthetic_raw(1))], device)
    outs = predict(model, batch)
    layout = token_layout(model, batch)
    memory = model.encoder(model.forward_features(batch["rgb"], batch["rgb_left"], batch["rgb_right"],
                                                  batch["rgb_center"], batch["lidar"], batch["measurements"]))
    assert memory.shape[0] == len(layout), (memory.shape, len(layout))

    contract = {
        "device": device,
        "load_report": {k: v for k, v in report.items() if k != "missing_keys" and k != "unexpected_keys"} |
                       {"missing": len(report["missing_keys"]), "unexpected": len(report["unexpected_keys"])},
        "inputs": {k: tensor_stats(v) for k, v in batch.items()},
        "outputs": {k: tensor_stats(v) for k, v in outs.items()},
        "token_sequence": {
            "axis_order": "[sequence, batch, channel]",
            "length": int(memory.shape[0]), "embed_dim": int(memory.shape[2]),
            "per_view": {str(v): {"spatial": sum(1 for t in layout if t["view"] == v and not t["is_global"]),
                                  "global": sum(1 for t in layout if t["view"] == v and t["is_global"])}
                         for v in range(5)},
            "modality_counts": {m: sum(1 for t in layout if t["modality"] == m) for m in ("camera", "lidar")},
        },
        "traffic_channels": ["soft_prob(dist)", "dx/3.5", "dy/3.5", "yaw/2", "box_x/3.5", "box_y/2", "speed/8"],
        "traffic_head_activation": "sigmoid (already applied inside the model)",
        "elapsed_s": round(time.time() - t0, 2),
        "note": "synthetic inputs; not a perception validation",
    }
    out = Path(cfg["paths"]["artifacts_root"]) / "tensor_contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, indent=2))
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
