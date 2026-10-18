"""Figures from stored metrics: AP vs severity per method, q vs severity, q-quantile vs
reference error, and paired examples (B2 missed / P detected and the reverse)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rrf import data, upstream
from rrf.config import load_config

FAMILIES = ["camera_gaussian_noise", "camera_gaussian_blur", "camera_darkening", "lidar_point_dropout", "lidar_xyz_jitter",
            "camera_rectangular_occlusion", "lidar_azimuth_sector_dropout"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--methods", nargs="+", default=["B0", "B2", "P"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    art = Path(cfg["paths"]["artifacts_root"])
    fig_dir = art / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = {m: json.loads((art / "metrics" / f"{m}_seed{args.seed}_full_test" / "metrics.json").read_text())
            for m in args.methods if (art / "metrics" / f"{m}_seed{args.seed}_full_test" / "metrics.json").exists()}
    if not runs:
        print("no full test metrics found"); return
    # 1) AP vs severity
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for ax, fam in zip(axes.ravel(), FAMILIES):
        for m, r in runs.items():
            ys = [r.get(f"{fam}_s{s}", {}).get("grid_fg_ap") for s in range(1, 6)]
            ax.plot(range(1, 6), [np.nan if y is None else y for y in ys], marker="o", label=m)
            ax.axhline(r["clean"]["grid_fg_ap"], ls=":", lw=0.8, color=ax.lines[-1].get_color())
        ax.set_title(fam, fontsize=9); ax.set_xlabel("severity"); ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("grid_fg_ap"); axes[0, 0].legend()
    axes[1, 3].axis("off")
    fig.suptitle(f"grid foreground AP vs severity (test split, seed {args.seed}); dotted = clean AP")
    fig.tight_layout(); fig.savefig(fig_dir / f"ap_vs_severity_seed{args.seed}.png", dpi=120); plt.close(fig)
    # 2) q vs severity (from reliability report) and quantiles
    rel = art / "metrics" / f"reliability_P_seed{args.seed}.json"
    if rel.exists():
        R = json.loads(rel.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for fam, d in R["q_by_family_severity"].items():
            sev = sorted(int(s) for s in d)
            for i, name in enumerate(("q_camera", "q_lidar")):
                axes[i].errorbar(sev, [d[str(s)]["q_mean"][i] for s in sev], yerr=[d[str(s)]["q_std"][i] for s in sev], marker="o", capsize=2, label=fam)
        for i, name in enumerate(("q_camera", "q_lidar")):
            axes[i].axhline(R["q_clean"]["q_mean"][i], ls=":", color="k", label="clean")
            axes[i].set_title(f"{name} vs severity (test)"); axes[i].set_xlabel("severity"); axes[i].grid(alpha=0.3)
        axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(fig_dir / f"q_vs_severity_seed{args.seed}.png", dpi=120); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for i, name in enumerate(("camera", "lidar")):
            tbl = R["overall"][f"quantiles_{name}"]
            axes[i].bar(range(len(tbl)), [t["mean_E"] or 0 for t in tbl], tick_label=[f"{t['q_range'][0]:.2f}-{t['q_range'][1]:.2f}" for t in tbl])
            axes[i].set_title(f"reference probe error by q_{name} quantile"); axes[i].tick_params(axis="x", labelsize=7)
        fig.tight_layout(); fig.savefig(fig_dir / f"q_quantile_error_seed{args.seed}.png", dpi=120); plt.close(fig)
    # 3) paired examples B2 vs P on a corrupted condition
    if "B2" in runs and "P" in runs:
        from rrf.tools.overlay import draw_grid
        from PIL import Image
        from rrf import corruptions as C
        cond = "camera_gaussian_blur_s5"
        pb = np.load(art / "metrics" / f"B2_seed{args.seed}_full_test" / f"pred_{cond}.npz")
        pp = np.load(art / "metrics" / f"P_seed{args.seed}_full_test" / f"pred_{cond}.npz")
        ids = list(pb["sample_ids"])
        root = cfg["paths"]["dataset_root"]
        gains = []
        for k, sid in enumerate(ids):
            y = upstream.foreground_label(data.load_gt(root, sid))
            if not y.any():
                continue
            # recall at 0.3 threshold difference
            rb = (pb["traffic"][k, :, 0] > 0.3)[y].mean(); rp = (pp["traffic"][k, :, 0] > 0.3)[y].mean()
            gains.append((rp - rb, sid, k))
        gains.sort()
        picks = [("P_better", gains[-1]), ("P_worse", gains[0])] if gains else []
        for tag, (delta, sid, k) in picks:
            raw = data.load_raw(root, sid)
            cor, _ = C.corrupt_sample(raw, {"camera": ("gaussian_blur", 5)}, cfg["corruption"]["base_seed"])
            bev = upstream.lidar_histogram(cor["lidar"])[2]
            gt = data.load_gt(root, sid).reshape(20, 20, 7)
            panels = [Image.fromarray(cor["rgb_front"]).resize((448, 336)),
                      draw_grid(bev, gt[..., 0], 0.01, (255, 0, 0)).resize((336, 336), Image.NEAREST),
                      draw_grid(bev, pb["traffic"][k, :, 0].reshape(20, 20), 0.3, (255, 255, 0)).resize((336, 336), Image.NEAREST),
                      draw_grid(bev, pp["traffic"][k, :, 0].reshape(20, 20), 0.3, (0, 255, 0)).resize((336, 336), Image.NEAREST)]
            canvas = Image.new("RGB", (sum(p.width for p in panels), 336))
            x = 0
            for p in panels:
                canvas.paste(p, (x, 0)); x += p.width
            canvas.save(fig_dir / f"example_{tag}_{cond}_seed{args.seed}.png")
            (fig_dir / f"example_{tag}_{cond}_seed{args.seed}.json").write_text(json.dumps(
                {"sample_id": sid, "condition": cond, "recall@0.3_P_minus_B2": float(delta), "corruption_seed": cfg["corruption"]["base_seed"],
                 "panels": ["corrupted front RGB", "GT (red) on LiDAR BEV", "B2 pred>0.3 (yellow)", "P pred>0.3 (green)"]}, indent=2))
    print("figures written to", fig_dir)


if __name__ == "__main__":
    main()
