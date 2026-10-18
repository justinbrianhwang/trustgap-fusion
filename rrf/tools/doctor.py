"""Environment check -> artifacts/environment.json. Never hides failures: each probe records
available:false with the error text."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from rrf import ROOT
from rrf.config import load_config


def probe(fn):
    try:
        return {"available": True, **fn()}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "cpu": platform.processor(), "config_sha256": cfg["_meta"]["config_sha256"]}

    def system():
        import psutil
        return {"ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
                "disk_free_gb": round(shutil.disk_usage(ROOT).free / 2**30, 1)}
    report["system"] = probe(system)

    def gpu():
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True)
        return {"nvidia_smi": out.strip()}
    report["gpu"] = probe(gpu)

    def torch_probe():
        import torch, torchvision
        x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda") if torch.cuda.is_available() else None
        return {"torch": torch.__version__, "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_matmul_ok": bool(x is not None and torch.isfinite(x).all().item()),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    report["torch"] = probe(torch_probe)

    def libs():
        import numpy, cv2, PIL, skimage, scipy
        return {"numpy": numpy.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__,
                "scikit_image": skimage.__version__, "scipy": scipy.__version__}
    report["libs"] = probe(libs)

    def upstream_probe():
        from rrf import upstream
        upstream.bootstrap()
        import timm
        from timm.models import is_model
        commit = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(upstream.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(upstream.UPSTREAM), "diff", "--stat", "--ignore-all-space"], text=True).strip()
        return {"timm_file": timm.__file__, "model_registered": is_model(cfg["upstream"]["model_name"]),
                "commit": commit, "expected_commit": cfg["upstream"]["commit"], "commit_matches": commit == cfg["upstream"]["commit"],
                "diff_stat_ignoring_whitespace": diff or "(no content changes)"}
    report["upstream"] = probe(upstream_probe)

    def ckpt():
        from rrf.upstream import sha256_file
        p = Path(paths["upstream_checkpoint"])
        return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    report["checkpoint"] = probe(ckpt)

    def carla_probe():
        from rrf import upstream
        upstream.bootstrap()
        import carla
        info = {"carla_module": carla.__file__, "client_version": None, "server": None}
        client = carla.Client(cfg["simulation"]["host"], cfg["simulation"]["port"])
        client.set_timeout(5.0)
        info["client_version"] = client.get_client_version()
        try:
            info["server"] = {"version": client.get_server_version(), "maps": sorted(m.split("/")[-1] for m in client.get_available_maps())}
        except RuntimeError as e:
            info["server"] = {"available": False, "error": str(e)}
        info["carla_root_exists"] = Path(paths["carla_root"], "CarlaUE4.exe").exists()
        return info
    report["carla"] = probe(carla_probe)

    def dataset():
        root = Path(paths["dataset_root"])
        eps = sorted(p.parent.name for p in root.glob("*/episode.json"))
        return {"root": str(root), "episodes": len(eps), "writable": os.access(root if root.exists() else ROOT, os.W_OK)}
    report["dataset"] = probe(dataset)

    out = Path(paths["artifacts_root"]) / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
