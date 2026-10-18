"""Small closed-loop check (md §14, optional): the model drives the ego in CARLA 0.9.16 along the
same route/traffic seed as a stored test episode, with corruptions injected at sensor
acquisition (never twice), through the *unmodified* upstream tracker + controller.

Per run we log route completion, collision events, stuck timeouts, mean speed, and the raw
per-frame q / traffic outputs (prediction used for control == prediction logged). The same
termination and failure aggregation applies to every method. No claim about driving safety is
derived from these numbers; they are a sanity check that the perception change survives the
closed loop.
"""
import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from rrf import corruptions as C
from rrf import upstream
from rrf.collector import Episode, planner_pos, theta_from_yaw
from rrf.config import load_config
from rrf.tools.evaluate import load_method

log = logging.getLogger("rrf.closed_loop")
CONDITIONS = {"clean": {}, "camera_blur_s3": {"camera": ("gaussian_blur", 3)}, "lidar_dropout_s3": {"lidar": ("point_dropout", 3)},
              "camera_blur_s5": {"camera": ("gaussian_blur", 5)}, "lidar_dropout_s5": {"lidar": ("point_dropout", 5)}}


class DrivingEpisode(Episode):
    """Collector episode whose ego is driven by an RRF model instead of BasicAgent."""

    def __init__(self, carla, client, cfg, town, k, seed, out_dir, gate, spec, method, max_ticks, run_id, ignore_signals=False):
        super().__init__(carla, client, cfg, town, k, seed, out_dir)
        self.gate, self.spec, self.method, self.max_ticks, self.run_id = gate, spec, method, max_ticks, run_id
        self.ignore_signals = ignore_signals  # perception-only driving: traffic-light / stop-sign heads bypassed for every method
        upstream.bootstrap()
        from team_code.interfuser_config import GlobalConfig
        from team_code.interfuser_controller import InterfuserController
        from team_code.tracker import Tracker
        self.controller = InterfuserController(GlobalConfig())
        self.tracker = Tracker()
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = next(gate.model.parameters()).device

    def route_progress(self):
        loc = self.ego.get_location()
        pts = [wp.transform.location for wp, _ in self.route]
        d = [loc.distance(p) for p in pts]
        i = int(np.argmin(d))
        return i / max(len(pts) - 1, 1), float(d[i]), float(loc.distance(pts[-1]))

    @torch.no_grad()
    def run(self):
        d = self.out_dir
        d.mkdir(parents=True, exist_ok=True)
        capture_every = int(self.sim["capture_every_n_ticks"])
        tick, last_infer, n_infer, stuck_ticks, rear = 0, -10, 0, 0, 0
        control = self.carla.VehicleControl()
        frames_log, speeds = [], []
        traffic_meta_avg = np.zeros((400, 7), np.float32)
        t0 = time.time()
        status = "max_ticks"
        while tick < self.max_ticks:
            frame = self.world.tick()
            snapshot = self.world.get_snapshot()
            sensors = self.buffer.read(frame)
            tick += 1
            self.ego.apply_control(control)
            v = self.ego.get_velocity()
            speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
            speeds.append(speed)
            if tick <= int(self.sim["warmup_ticks"]) // 2:
                continue
            lidar = sensors["lidar"]
            front_sweep = len(lidar) > 0 and float(np.mean(lidar[:, 1] > 0.0)) >= 0.5
            if not front_sweep or tick - last_infer < capture_every:
                rear += int(not front_sweep)
                continue
            last_infer = tick
            tf = self.ego.get_transform()
            pos = planner_pos(tf.location)
            near_wp, near_cmd = self.cmd_planner.run_step(pos)
            theta = theta_from_yaw(tf.rotation.yaw)
            R = np.array([[np.cos(theta + np.pi / 2), -np.sin(theta + np.pi / 2)], [np.sin(theta + np.pi / 2), np.cos(theta + np.pi / 2)]])
            target_point = R.T.dot(np.array([near_wp[0] - pos[0], near_wp[1] - pos[1]])).astype(np.float32)
            raw = {"sample_id": f"{self.run_id}/{n_infer:04d}", "rgb_front": sensors["rgb_front"], "rgb_left": sensors["rgb_left"],
                   "rgb_right": sensors["rgb_right"], "lidar": lidar, "speed": speed, "command": int(near_cmd), "target_point": target_point}
            cor, _ = C.corrupt_sample(raw, self.spec, self.cfg["corruption"]["base_seed"])  # injected once, at acquisition
            batch = upstream.collate([upstream.preprocess(cor)], self.device)
            out = self.gate(batch)
            traffic = out["traffic"][0].cpu().numpy()
            wps = out["waypoints"][0].cpu().numpy()
            junction = float(self.softmax(out["is_junction"])[0, 0])
            light = float(self.softmax(out["traffic_light_state"])[0, 0])
            stop = float(self.softmax(out["stop_sign"])[0, 0])
            # upstream agent post-processing: tracker every inference (it runs at 10 Hz there too), momentum 0
            tracked = self.tracker.update_and_predict(traffic.reshape(20, 20, -1).copy(), pos, theta, n_infer).reshape(400, -1)
            traffic_meta_avg = tracked.astype(np.float32)
            steer, throttle, brake, meta = self.controller.run_step(speed, wps, junction, 0.0 if self.ignore_signals else light,
                                                                    1.0 if self.ignore_signals else stop, traffic_meta_avg)
            if brake < 0.05:
                brake = 0.0
            if brake > 0.1:
                throttle = 0.0
            control = self.carla.VehicleControl(steer=float(np.clip(steer, -1, 1)), throttle=float(np.clip(throttle, 0, 1)), brake=float(np.clip(brake, 0, 1)))
            n_infer += 1
            prog, off_route, to_goal = self.route_progress()
            q = out["q"][0].cpu().numpy().tolist() if out.get("q") is not None else None
            frames_log.append({"tick": tick, "world_frame": int(frame), "t": float(snapshot.timestamp.elapsed_seconds), "speed": speed, "q": q,
                               "progress": prog, "off_route_m": off_route, "steer": control.steer, "throttle": control.throttle, "brake": control.brake,
                               "fg_cells_pred>0.3": int((traffic[:, 0] > 0.3).sum()), "collision": int(frame) in self.collisions,
                               "p_junction": junction, "p_red_light": light, "p_stop_sign": stop, "controller_meta": [str(m) for m in meta[:3]], "safe_distance": float(meta[3])})
            stuck_ticks = stuck_ticks + capture_every if speed < 0.1 and tick > 200 else 0
            if to_goal < 6.0:
                status = "route_completed"
                break
            if off_route > 15.0:
                status = "off_route"
                break
            if stuck_ticks >= 20 * 20:  # 20 s without motion
                status = "stuck_timeout"
                break
            if len(set(self.collisions)) >= 40:  # persistent collision (~2 s)
                status = "collision_stop"
                break
        prog, _, _ = self.route_progress()
        info = {"run_id": self.run_id, "method": self.method, "condition": C.condition_name(self.spec), "episode_seed": self.seed, "town": self.town,
                "ignore_signals": self.ignore_signals,
                "status": status, "route_completion": prog, "ticks": tick, "inferences": n_infer, "collision_events": len(self.collisions),
                "collision_frames_distinct": len(set(self.collisions)), "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
                "wall_seconds": round(time.time() - t0, 1), "rear_sweeps_skipped": rear,
                "q_mean": (np.mean([f["q"] for f in frames_log if f["q"]], 0).tolist() if any(f["q"] for f in frames_log) else None),
                "npc_vehicles": len(self.npcs), "npc_walkers": len(self.walkers)}
        (d / "run.json").write_text(json.dumps(info, indent=2))
        with open(d / "frames.jsonl", "w") as f:
            for r in frames_log:
                f.write(json.dumps(r) + "\n")
        return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--methods", nargs="+", default=["B0", "B2", "P"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", nargs="*", default=[], help="stored episode ids whose town/seed/route/traffic are replayed")
    ap.add_argument("--routes", nargs="*", default=[], help="synthetic route ids Town_epKK_seedS (new seeds, no stored data needed)")
    ap.add_argument("--search", type=int, default=0, help="route search: run B0 clean on N new-seed routes per town (seed offset 100) and report completion")
    ap.add_argument("--conditions", nargs="+", default=["clean", "camera_blur_s3", "lidar_dropout_s3"])
    ap.add_argument("--max-ticks", type=int, default=1600)
    ap.add_argument("--out", default="artifacts/closed_loop")
    ap.add_argument("--ignore-signals", action="store_true", help="bypass the traffic-light/stop-sign heads (perception-only driving), same for every method")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    sim = cfg["simulation"]
    upstream.bootstrap()
    import carla
    client = carla.Client(sim["host"], sim["port"])
    client.set_timeout(120.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out)
    rows = []
    if args.search:
        from rrf.collector import hash_town
        args.methods, args.conditions = ["B0"], ["clean"]
        args.episodes = [f"{t}_ep{k:02d}_seed{int(cfg['execution']['seed']) * 1000 + hash_town(t) * 100 + k + 100}" for t in sim["towns"] for k in range(args.search)]
    args.episodes = list(args.episodes) + list(args.routes)
    for method in args.methods:
        gate, _ = load_method(cfg, method, args.seed, device)
        gate.model.eval()
        if gate.estimator is not None:
            gate.estimator.eval()
        for eid in args.episodes:
            town, rest = eid.rsplit("_ep", 1)
            k, seed = rest.split("_seed")
            for cond in args.conditions:
                run_id = f"{method}_seed{args.seed}__{eid}__{cond}"
                out_dir = out_root / run_id
                if (out_dir / "run.json").exists():
                    rows.append(json.loads((out_dir / "run.json").read_text()))
                    log.info("skip existing %s", run_id)
                    continue
                ep = DrivingEpisode(carla, client, cfg, town, int(k), int(seed), out_dir, gate, CONDITIONS[cond], method, args.max_ticks, run_id, args.ignore_signals)
                try:
                    ep.setup_world(); ep.spawn_ego(); ep.build_route(); ep.spawn_traffic(); ep.attach_sensors()
                    for _ in range(5):
                        ep.world.tick(); ep.buffer.read(ep.world.get_snapshot().frame)
                    info = ep.run()
                    rows.append(info)
                    log.info("%s: %s completion=%.2f collisions=%d ticks=%d", run_id, info["status"], info["route_completion"], info["collision_events"], info["ticks"])
                except Exception:
                    log.exception("run %s failed", run_id)
                    rows.append({"run_id": run_id, "method": method, "condition": cond, "episode_seed": int(seed), "town": town, "status": "failed"})
                finally:
                    ep.cleanup()
    out_root.mkdir(parents=True, exist_ok=True)
    # aggregate every stored run in out_root (also runs of earlier invocations / other methods)
    rows = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/run.json"))]
    (out_root / "runs.json").write_text(json.dumps(rows, indent=2))
    csv_path = Path(cfg["paths"]["artifacts_root"]) / "csv" / f"{out_root.name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list({k: None for r in rows for k in r if k != "q_mean"}) + ["q_camera_mean", "q_lidar_mean"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            q = r.get("q_mean") or [None, None]
            w.writerow({**r, "q_camera_mean": q[0], "q_lidar_mean": q[1]})
    print("wrote", csv_path, len(rows))


if __name__ == "__main__":
    main()
