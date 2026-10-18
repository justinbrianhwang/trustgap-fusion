"""Visual sanity check: front RGB, LiDAR BEV histogram with the GT traffic grid drawn on it,
and (optionally) the B0 prediction grid. Used to verify the coordinate conventions.

BEV histogram axes (upstream lidar_to_histogram_features after the y flip):
  dim0 (rows)  = 224 bins of sensor x over [-28, 29] m  (sensor +x = ego left, LiDAR yaw -90) -> 0.254 m/bin
  dim1 (cols)  = 224 bins of -sensor y over [-28, 0] m  (sensor +y = ego forward) -> col 0 = 28 m ahead, 0.125 m/bin
Traffic grid cell (i, j): lateral x = j - 9.5 (right +), forward y = 17.5 - i.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from rrf import data, upstream
from rrf.config import load_config

ROW_PER_M = 224 / 57.0   # dim0: x in [-28, 29]
COL_PER_M = 8.0          # dim1: -y in [-28, 0]


def grid_cell_to_bev_px(i, j):
    """Cell (i, j) -> (row, col) of its centre in the 224x224 BEV histogram."""
    lateral = j - 9.5           # right positive
    forward = 17.5 - i
    row = (28.0 - lateral) * ROW_PER_M  # sensor x = -lateral (left positive)
    col = (28.0 - forward) * COL_PER_M
    return row, col


def draw_grid(bev_total, grid_scores, threshold, color):
    img = Image.fromarray((np.clip(bev_total, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    for i in range(20):
        for j in range(20):
            if grid_scores[i, j] > threshold:
                r, c = grid_cell_to_bev_px(i, j)
                d.rectangle([c - COL_PER_M / 2, r - ROW_PER_M / 2, c + COL_PER_M / 2, r + ROW_PER_M / 2], outline=color)
    # ego position: lateral 0, forward 0 -> (row 224, col 224); mark ROI box
    r0, c0 = grid_cell_to_bev_px(19.5, -0.5)
    r1, c1 = grid_cell_to_bev_px(-0.5, 19.5)
    d.rectangle([c1, r1, c0, r0], outline=(0, 128, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/baseline_smoke/overlay")
    ap.add_argument("--no-model", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = args.dataset_root or cfg["paths"]["dataset_root"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = None
    if not args.no_model:
        model, _ = upstream.load_upstream_model(cfg["paths"]["upstream_checkpoint"], "cuda" if torch.cuda.is_available() else "cpu")
    for sid in args.samples:
        raw = data.load_raw(root, sid)
        gt = data.load_gt(root, sid).reshape(20, 20, 7)
        bev = upstream.lidar_histogram(raw["lidar"])[2]
        panels = [Image.fromarray(raw["rgb_front"]).resize((448, 336)), draw_grid(bev, gt[..., 0], 0.01, (255, 0, 0)).resize((336, 336), Image.NEAREST)]
        info = {"sample": sid, "gt_fg_cells": int((gt[..., 0] > 0.01).sum()), "lidar_points": int(len(raw["lidar"]))}
        if model is not None:
            x = upstream.collate([upstream.preprocess(raw)], next(model.parameters()).device)
            pred = upstream.predict(model, x)["traffic"][0].cpu().numpy().reshape(20, 20, 7)
            panels.append(draw_grid(bev, pred[..., 0], 0.3, (0, 255, 0)).resize((336, 336), Image.NEAREST))
            info["pred_cells_gt0.3"] = int((pred[..., 0] > 0.3).sum())
            info["pred_max"] = float(pred[..., 0].max())
        w = sum(p.width for p in panels)
        canvas = Image.new("RGB", (w, 336), (0, 0, 0))
        xoff = 0
        for p in panels:
            canvas.paste(p, (xoff, 0))
            xoff += p.width
        name = sid.replace("/", "__")
        canvas.save(out / f"{name}.png")
        (out / f"{name}.json").write_text(json.dumps(info, indent=2))
        print(info)


if __name__ == "__main__":
    main()
