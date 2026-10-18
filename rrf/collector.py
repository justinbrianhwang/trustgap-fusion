"""CARLA 0.9.16 raw-observation collector for InterFuser/TransFuser episodes.

One episode = one BasicAgent drive along a GlobalRoutePlanner route in synchronous mode.
Every `capture_every_n_ticks` tick we store the raw sensors (PNG + npy), the measurements
JSON in the upstream schema, and actors_data JSON, so upstream GT generation applies as-is.

Sensor rig, blueprint attributes and preprocessing conventions follow
third_party/InterFuser/leaderboard (team_code/interfuser_agent.py, autoagents/agent_wrapper.py).
See docs/DECISIONS.md for the 0.9.16-specific coordinate conventions.
"""
import argparse
import json
import logging
import math
import random
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue

import numpy as np
from PIL import Image

from rrf import upstream
from rrf.config import load_config

log = logging.getLogger("rrf.collector")

SENSOR_SPECS = (
    {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "width": 800, "height": 600, "fov": 100, "id": "rgb_front"},
    {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": -60.0, "width": 400, "height": 300, "fov": 100, "id": "rgb_left"},
    {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 60.0, "width": 400, "height": 300, "fov": 100, "id": "rgb_right"},
    {"type": "sensor.lidar.ray_cast", "x": 1.3, "y": 0.0, "z": 2.5, "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "id": "lidar"},
    {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "id": "imu"},
)
TF_SENSOR_SPECS = (
    {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "width": 960, "height": 480, "fov": 120, "id": "rgb_front"},
    {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": -60.0, "width": 960, "height": 480, "fov": 120, "id": "rgb_left"},
    {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 60.0, "width": 960, "height": 480, "fov": 120, "id": "rgb_right"},
    {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.05, "id": "imu"},
    {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.01, "id": "gps"},
    {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
    {"type": "sensor.lidar.ray_cast", "x": 1.3, "y": 0.0, "z": 2.5, "roll": 0.0, "pitch": 0.0, "yaw": -90.0,
     "rotation_frequency": 20, "points_per_second": 1200000, "id": "lidar"},
)
# leaderboard/autoagents/agent_wrapper.py (SENSORS track)
CAMERA_ATTRS = {"lens_circle_multiplier": "3.0", "lens_circle_falloff": "3.0", "chromatic_aberration_intensity": "0.5", "chromatic_aberration_offset": "0"}
LIDAR_ATTRS = {"range": "85", "rotation_frequency": "10", "channels": "64", "upper_fov": "10", "lower_fov": "-30",
               "points_per_second": "600000", "atmosphere_attenuation_rate": "0.004", "dropoff_general_rate": "0.45",
               "dropoff_intensity_limit": "0.8", "dropoff_zero_intensity": "0.4"}
TF_LIDAR_ATTRS = {**LIDAR_ATTRS, "rotation_frequency": "20", "points_per_second": "1200000"}
SPARSE_SAMPLE_FACTOR = 50.0  # leaderboard route_manipulation.downsample_route
INTERFUSER_SUBDIRS = ("rgb_front", "rgb_left", "rgb_right", "lidar", "measurements", "actors_data")
TF_SUBDIRS = ("tf_rgb", "tf_lidar", "tf_measurements", "measurements", "actors_data")


def capture_subdirs(rig):
    if rig == "interfuser":
        return INTERFUSER_SUBDIRS
    if rig == "transfuser":
        return TF_SUBDIRS
    if rig == "both":
        return INTERFUSER_SUBDIRS + tuple(x for x in TF_SUBDIRS if x not in INTERFUSER_SUBDIRS)
    raise ValueError(f"unknown sensor rig: {rig}")


# ------------------------------------------------------------------ frames / conventions
def planner_pos(location):
    """0.9.10 leaderboard GPS-metric frame reconstructed from the world transform: (-y, x)."""
    return np.array([-location.y, location.x], dtype=np.float64)


def theta_from_yaw(yaw_deg):
    """Upstream compass convention: north = -y, so yaw 0 (facing +x) -> pi/2."""
    return (math.radians(yaw_deg) + math.pi / 2) % (2 * math.pi)


def downsample_route(route, sample_factor=SPARSE_SAMPLE_FACTOR):
    ids, prev_option, dist = [], None, 0.0
    for i, (wp, option) in enumerate(route):
        cur, prev = int(option), (None if prev_option is None else int(prev_option))
        if cur in (5, 6) or prev is None or (prev != cur and prev not in (5, 6)) or dist > sample_factor or i == len(route) - 1:
            ids.append(i)
            dist = 0.0
        else:
            dist += wp.transform.location.distance(route[i - 1][0].transform.location)
        prev_option = option
    return sorted(set(ids))


class SensorBuffer:
    def __init__(self, ids):
        self.q = {i: Queue() for i in ids}
        self.last_frames = {}

    def listen(self, sensor_id, convert):
        def cb(data):
            self.q[sensor_id].put((int(data.frame), convert(data)))
        return cb

    def read(self, frame, timeout=10.0):
        out = {}
        for sid, q in self.q.items():
            deadline = time.monotonic() + timeout
            while True:
                try:
                    f, payload = q.get(timeout=max(0.0, deadline - time.monotonic()))
                except Empty:
                    raise TimeoutError(f"sensor {sid} produced nothing for frame {frame}")
                if f < frame:
                    continue
                if f > frame:
                    raise RuntimeError(f"sensor {sid} skipped frame {frame} (got {f})")
                out[sid] = payload
                self.last_frames[sid] = f
                break
        return out


def _convert(spec):
    t = spec["type"]
    if t == "sensor.camera.rgb":
        if spec.get("_rig") == "transfuser":
            return lambda img: np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy()
        return lambda img: np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3][:, :, ::-1].copy()
    if t == "sensor.lidar.ray_cast":
        return lambda m: np.frombuffer(m.raw_data, np.float32).reshape(-1, 4).copy()
    if t == "sensor.other.imu":
        if spec.get("_rig") == "transfuser":
            return lambda m: [float(m.accelerometer.x), float(m.accelerometer.y), float(m.accelerometer.z),
                              float(m.gyroscope.x), float(m.gyroscope.y), float(m.gyroscope.z), float(m.compass)]
        return lambda m: float(m.compass)
    if t == "sensor.other.gnss":
        return lambda m: [float(m.latitude), float(m.longitude), float(m.altitude)]
    raise ValueError(t)


# ------------------------------------------------------------------ weather
WEATHER_ATTRS = ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
                 "fog_density", "fog_distance", "fog_falloff", "wetness", "scattering_intensity", "mie_scattering_scale", "rayleigh_scattering_scale", "dust_storm")
# research weather conditions: CARLA presets, plus a fog condition (no preset has fog) [design choice]
CUSTOM_WEATHER = {
    "fog": ("ClearNoon", {"fog_density": 70.0, "fog_distance": 10.0, "fog_falloff": 1.0, "cloudiness": 60.0}),
}


def make_weather(carla, name):
    if name in CUSTOM_WEATHER:
        base, overrides = CUSTOM_WEATHER[name]
        w = getattr(carla.WeatherParameters, base)
        for k, v in overrides.items():
            setattr(w, k, v)
        return w
    return getattr(carla.WeatherParameters, name)


def weather_dict(w):
    return {k: float(getattr(w, k)) for k in WEATHER_ATTRS if hasattr(w, k)}


# ------------------------------------------------------------------ episode
class Episode:
    def __init__(self, carla, client, cfg, town, episode_index, seed, out_dir, rig="interfuser"):
        self.carla, self.client, self.cfg, self.town = carla, client, cfg, town
        self.seed, self.out_dir, self.episode_index = seed, Path(out_dir), episode_index
        self.sim = cfg["simulation"]
        self.rig = rig
        self.actors, self.sensors, self.npcs, self.walkers, self.walker_ctrls = [], [], [], [], []
        self.rng = random.Random(seed)

    @property
    def has_interfuser(self):
        return self.rig in ("interfuser", "both")

    @property
    def has_transfuser(self):
        return self.rig in ("transfuser", "both")

    # -- world
    def setup_world(self):
        carla = self.carla
        # Always (re)load the map: a fresh world per episode, no leftover actors or settings.
        world = self.client.load_world(self.town)
        self.world = world
        self.tm = self.client.get_trafficmanager(self.sim["traffic_manager_port"])
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(self.seed)
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.sim["fixed_delta_seconds"]
        s.no_rendering_mode = bool(self.sim["no_rendering_mode"])
        world.apply_settings(s)
        self.weather_params = make_weather(carla, self.sim["weather"])
        world.set_weather(self.weather_params)
        self.map = world.get_map()
        self.bp = world.get_blueprint_library()
        self.spawn_points = self.map.get_spawn_points()
        world.tick()

    def spawn_ego(self):
        bps = list(self.bp.filter(self.sim["ego_blueprint"])) or list(self.bp.filter("vehicle.tesla.model3"))
        bp = bps[0]
        bp.set_attribute("role_name", "hero")
        order = list(range(len(self.spawn_points)))
        self.rng.shuffle(order)
        for idx in order:
            ego = self.world.try_spawn_actor(bp, self.spawn_points[idx])
            if ego is not None:
                self.ego, self.spawn_index = ego, idx
                self.actors.append(ego)
                self.world.tick()
                return
        raise RuntimeError("could not spawn ego")

    def build_route(self):
        from agents.navigation.basic_agent import BasicAgent
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        grp = GlobalRoutePlanner(self.map, 1.0)
        start = self.ego.get_location()
        candidates = [i for i in range(len(self.spawn_points)) if i != self.spawn_index]
        self.rng.shuffle(candidates)
        route = []
        for i in candidates[:20]:
            dest = self.spawn_points[i].location
            if dest.distance(start) < 100.0:
                continue
            r = grp.trace_route(start, dest)
            if len(r) >= 150:
                route, self.dest_index = r, i
                break
        if not route:
            raise RuntimeError("no route >= 150 m found")
        self.route = route
        self.agent = BasicAgent(self.ego, target_speed=self.sim["ego_target_speed_kmh"])
        self.agent.set_global_plan(route)
        self.agent.ignore_traffic_lights(False)
        # upstream planners in the reconstructed planner frame
        from team_code.planner import RoutePlanner
        self.cmd_planner = RoutePlanner(4.0, 50.0)     # sparse plan: command + target_point (as inference agent)
        self.wp_planner = RoutePlanner(4.0, 50.0)      # dense plan: future_waypoints (as expert)
        for planner, entries in ((self.cmd_planner, [route[i] for i in downsample_route(route)]), (self.wp_planner, route)):
            planner.route = deque((planner_pos(wp.transform.location), opt) for wp, opt in entries)
        if self.has_transfuser:
            self.tf_cmd_planner = RoutePlanner(7.5, 50.0)
            self.tf_cmd_planner.route = deque((planner_pos(route[i][0].transform.location), route[i][1])
                                              for i in downsample_route(route))

    def spawn_traffic(self):
        bps = sorted(self.bp.filter("vehicle.*"), key=lambda b: b.id)
        bps = [b for b in bps if int(b.get_attribute("number_of_wheels")) >= 2]
        cands = [p for i, p in enumerate(self.spawn_points) if i != self.spawn_index]
        self.rng.shuffle(cands)
        port = self.tm.get_port()
        for tf in cands:
            if len(self.npcs) >= self.sim["num_vehicles"]:
                break
            bp = self.rng.choice(bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", self.rng.choice(bp.get_attribute("color").recommended_values))
            bp.set_attribute("role_name", "autopilot")
            v = self.world.try_spawn_actor(bp, tf)
            if v is None:
                continue
            v.set_autopilot(True, port)
            self.npcs.append(v)
        walker_bps = list(self.bp.filter("walker.pedestrian.*"))
        ctrl_bp = self.bp.find("controller.ai.walker")
        for _ in range(self.sim["num_walkers"]):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            w = self.world.try_spawn_actor(self.rng.choice(walker_bps), self.carla.Transform(loc))
            if w is None:
                continue
            self.walkers.append(w)
        self.world.tick()
        for w in self.walkers:
            c = self.world.spawn_actor(ctrl_bp, self.carla.Transform(), attach_to=w)
            self.walker_ctrls.append(c)
        self.world.tick()
        for c in self.walker_ctrls:
            c.start()
            c.go_to_location(self.world.get_random_location_from_navigation())
            c.set_max_speed(1.0 + self.rng.random())
        self.actors += self.npcs + self.walkers + self.walker_ctrls

    def attach_sensors(self):
        specs = []
        if self.has_interfuser:
            specs.extend(({**s, "_buffer_id": s["id"], "_rig": "interfuser"} for s in SENSOR_SPECS))
        if self.has_transfuser:
            specs.extend(({**s, "_buffer_id": f"tf_{s['id']}", "_rig": "transfuser"}
                          for s in TF_SENSOR_SPECS if s["type"] != "sensor.speedometer"))
        self.buffer = SensorBuffer([s["_buffer_id"] for s in specs])
        self.sensor_attrs = {}
        self.tf_sensor_attrs = {}
        for spec in specs:
            bp = self.bp.find(spec["type"])
            attrs = {}
            if spec["type"] == "sensor.camera.rgb":
                attrs.update(image_size_x=str(spec["width"]), image_size_y=str(spec["height"]), fov=str(spec["fov"]))
                if spec["_rig"] == "interfuser":
                    attrs.update(CAMERA_ATTRS)
            elif spec["type"] == "sensor.lidar.ray_cast":
                attrs.update(TF_LIDAR_ATTRS if spec["_rig"] == "transfuser" else LIDAR_ATTRS)
            # sensor_tick from TransFuser's agent (0.05 / 0.01) is deliberately NOT applied: in synchronous
            # mode the sensors emit every tick anyway, and sensor_tick == fixed_delta_seconds made the IMU
            # skip a tick through float accumulation (live trial 2026-10-16, see DECISIONS).
            for k, v in attrs.items():
                bp.set_attribute(k, v)
            tf = self.carla.Transform(self.carla.Location(x=spec["x"], y=spec["y"], z=spec["z"]),
                                      self.carla.Rotation(roll=spec["roll"], pitch=spec["pitch"], yaw=spec["yaw"]))
            sensor = self.world.spawn_actor(bp, tf, attach_to=self.ego)
            sensor.listen(self.buffer.listen(spec["_buffer_id"], _convert(spec)))
            self.sensors.append(sensor)
            public_spec = {k: v for k, v in spec.items() if not k.startswith("_")}
            target = self.tf_sensor_attrs if spec["_rig"] == "transfuser" else self.sensor_attrs
            target[spec["id"]] = {"spec": public_spec, "attributes": attrs}
        if self.has_transfuser:
            speed_spec = next(s for s in TF_SENSOR_SPECS if s["type"] == "sensor.speedometer")
            self.tf_sensor_attrs["speed"] = {"spec": speed_spec, "attributes": {"reading_frequency": "20"}}
        self.collision = self.world.spawn_actor(self.bp.find("sensor.other.collision"), self.carla.Transform(), attach_to=self.ego)
        self.collisions = []
        self.collision.listen(lambda e: self.collisions.append(int(e.frame)))
        self.sensors.append(self.collision)

    # -- per-tick bookkeeping
    def actors_data(self):
        data = {}
        ego_loc = self.ego.get_location()
        for kind, tpe, radius in (("*vehicle*", 0, 50.0), ("*walker*", 1, 50.0)):
            for a in self.world.get_actors().filter(kind):
                if "controller" in a.type_id:
                    continue
                loc = a.get_location()
                if loc.distance(ego_loc) > radius:
                    continue
                ori = a.get_transform().rotation.get_forward_vector()
                box, vel = a.bounding_box.extent, a.get_velocity()
                data[str(a.id)] = {"loc": [loc.x, loc.y, loc.z], "ori": [ori.x, ori.y, ori.z], "box": [box.x, box.y],
                                   "vel": [vel.x, vel.y, vel.z], "tpe": tpe}
        return data

    def measurements(self, compass, speed, snapshot):
        tf = self.ego.get_transform()
        pos = planner_pos(tf.location)
        near_wp, near_cmd = self.cmd_planner.run_step(pos)
        self.wp_planner.run_step(pos)
        theta = theta_from_yaw(tf.rotation.yaw)
        light_state = self.ego.get_traffic_light_state()
        at_red = self.ego.is_at_traffic_light() and str(light_state) == "Red"
        light = self.ego.get_traffic_light()
        return {
            "gps_x": float(pos[0]), "gps_y": float(pos[1]),
            "x": float(tf.location.x), "y": float(tf.location.y), "z": float(tf.location.z),
            "theta": float(theta), "compass_imu": float(compass), "yaw_deg": float(tf.rotation.yaw),
            "speed": float(speed),
            "x_command": float(near_wp[0]), "y_command": float(near_wp[1]),
            "command": int(near_cmd), "gt_command": int(near_cmd),
            "is_junction": bool(self.map.get_waypoint(tf.location).is_junction),
            "is_red_light_present": [int(light.id)] if (at_red and light is not None) else [],
            "traffic_light_state": str(light_state),
            "is_vehicle_present": [], "is_bike_present": [], "is_lane_vehicle_present": [],
            "is_junction_vehicle_present": [], "is_pedestrian_present": [], "is_stop_sign_present": [],
            "affected_light_id": int(light.id) if light is not None else -1,
            "future_waypoints": [[float(x), float(y), int(c)] for x, y, c in self.wp_planner.get_future_waypoints(50)],
            "world_frame": int(snapshot.frame), "simulation_timestamp": float(snapshot.timestamp.elapsed_seconds),
            "collision": int(snapshot.frame) in self.collisions,
            "weather": self.sim["weather"],
        }

    def tf_measurements(self, sensors, source_frames, speed, snapshot, meas):
        """Scalars consumed by HybridAgent.tick/prepare_goal_location for this exact simulator frame."""
        pos = planner_pos(self.ego.get_location())
        next_wp, next_cmd = self.tf_cmd_planner.run_step(pos)
        theta = theta_from_yaw(self.ego.get_transform().rotation.yaw)
        rot = np.array([[np.cos(np.pi / 2 + theta), -np.sin(np.pi / 2 + theta)],
                        [np.sin(np.pi / 2 + theta), np.cos(np.pi / 2 + theta)]])
        target = rot.T.dot(np.asarray(next_wp) - pos)
        source_frames = {k[3:]: f for k, f in source_frames.items() if k.startswith("tf_")}
        assert source_frames and set(source_frames.values()) == {int(snapshot.frame)}
        return {"gps": sensors["tf_gps"], "imu": sensors["tf_imu"], "compass": float(sensors["tf_imu"][-1]),
                "speed": float(speed), "target_point": [float(target[0]), float(target[1])], "command": int(next_cmd),
                "world_frame": int(snapshot.frame), "simulation_timestamp": float(snapshot.timestamp.elapsed_seconds),
                "source_frames": source_frames, "measurement_world_frame": int(meas["world_frame"])}

    # -- main loop
    def run(self):
        d = self.out_dir
        for sub in capture_subdirs(self.rig):
            (d / sub).mkdir(parents=True, exist_ok=True)
        capture_every = int(self.sim["capture_every_n_ticks"])
        n_frames, tick, last_capture_tick, rear_sweeps = 0, 0, -10, 0
        compass_diffs = []
        t0 = time.time()
        while n_frames < int(self.sim["frames_per_episode"]):
            frame = self.world.tick()
            snapshot = self.world.get_snapshot()
            sensors = self.buffer.read(frame)
            source_frames = dict(self.buffer.last_frames)
            assert set(source_frames.values()) == {int(frame)} == {int(snapshot.frame)}
            control = self.agent.run_step()
            self.ego.apply_control(control)
            tick += 1
            if tick <= int(self.sim["warmup_ticks"]) or tick - last_capture_tick < capture_every:
                continue
            # The leaderboard LiDAR rotates at 10 Hz while the world ticks at 20 Hz, so each
            # measurement is a half sweep. Upstream's histogram keeps raw y in [0, 28] (the
            # front half, sensor +y = ego forward at yaw -90). Capture only front-half sweeps.
            if self.has_interfuser:
                lidar = sensors["lidar"]
                if len(lidar) == 0 or float(np.mean(lidar[:, 1] > 0.0)) < 0.5:
                    rear_sweeps += 1
                    continue
            last_capture_tick = tick
            if self.agent.done():
                log.info("route finished at frame %d", n_frames)
                break
            v = self.ego.get_velocity()
            speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
            compass = sensors["imu"] if self.has_interfuser else sensors["tf_imu"][-1]
            meas = self.measurements(compass, speed, snapshot)
            meas["source_frames"] = source_frames
            meas["control"] = {"steer": control.steer, "throttle": control.throttle, "brake": control.brake}
            compass_diffs.append(abs((meas["compass_imu"] - meas["theta"] + math.pi) % (2 * math.pi) - math.pi))
            name = "%04d" % n_frames
            if self.has_interfuser:
                for key in ("rgb_front", "rgb_left", "rgb_right"):
                    Image.fromarray(sensors[key]).save(d / key / f"{name}.png", compress_level=1)
                np.save(d / "lidar" / f"{name}.npy", sensors["lidar"])
            if self.has_transfuser:
                # Lossless strip of the three raw camera views, in the agent's left/front/right order.
                tf_bgr = np.concatenate([sensors["tf_rgb_left"], sensors["tf_rgb_front"], sensors["tf_rgb_right"]], axis=1)
                tf_rgb = tf_bgr[:, :, ::-1].copy()
                Image.fromarray(tf_rgb).save(d / "tf_rgb" / f"{name}.png", compress_level=1)
                np.save(d / "tf_lidar" / f"{name}.npy", sensors["tf_lidar"])
                tf_meas = self.tf_measurements(sensors, source_frames, speed, snapshot, meas)
                (d / "tf_measurements" / f"{name}.json").write_text(json.dumps(tf_meas))
            (d / "measurements" / f"{name}.json").write_text(json.dumps(meas))
            (d / "actors_data" / f"{name}.json").write_text(json.dumps(self.actors_data()))
            n_frames += 1
        elapsed = time.time() - t0
        info = {
            "episode_id": d.name, "town": self.town, "seed": self.seed, "episode_index": self.episode_index,
            "frames": n_frames, "ticks": tick, "wall_seconds": round(elapsed, 1), "rear_half_sweeps_skipped": rear_sweeps,
            "lidar_frame": "raw sensor frame at yaw -90: +x = ego left, +y = ego forward; only front-half sweeps stored",
            "spawn_index": self.spawn_index, "dest_index": self.dest_index, "route_points": len(self.route),
            "npc_vehicles": len(self.npcs), "npc_walkers": len(self.walkers),
            "collision_frames": sorted(set(self.collisions)),
            "compass_vs_theta_abs_diff_rad": {"mean": float(np.mean(compass_diffs)) if compass_diffs else None,
                                              "max": float(np.max(compass_diffs)) if compass_diffs else None},
            "settings": {"fixed_delta_seconds": self.sim["fixed_delta_seconds"], "capture_every_n_ticks": capture_every,
                         "synchronous_mode": True, "no_rendering_mode": bool(self.sim["no_rendering_mode"]),
                         "weather": self.sim["weather"], "weather_parameters": weather_dict(self.weather_params),
                         "ego_blueprint": self.ego.type_id},
            "sensors": self.sensor_attrs, "carla_version": self.client.get_server_version(),
            "planner_frame": "gps_x,gps_y = (-y_world, x_world); theta = yaw_rad + pi/2",
            "complete": n_frames >= int(self.sim["frames_per_episode"]) or self.agent.done(),
        }
        if self.has_transfuser:
            info["tf_sensors"] = self.tf_sensor_attrs
            info["tf_capture"] = {"rgb_layout": "one lossless RGB PNG strip: left, front, right; each 960x480 view reconstructs the agent's raw BGR input by channel reversal",
                                  "lidar_frame": "raw CARLA sensor frame at yaw -90; 20 Hz rotation at 20 Hz simulation gives one full sweep per stored tick",
                                  "same_simulator_frame_asserted": True}
        if not self.has_interfuser:
            info["lidar_frame"] = "TransFuser raw sensor frame at yaw -90; one full sweep per stored tick"
        (d / "episode.json").write_text(json.dumps(info, indent=2))
        return info

    def cleanup(self):
        for s in self.sensors:
            try:
                s.stop(); s.destroy()
            except RuntimeError:
                pass
        for c in self.walker_ctrls:
            try:
                c.stop()
            except RuntimeError:
                pass
        for a in reversed(self.actors):
            try:
                if a.is_alive:
                    a.destroy()
            except RuntimeError:
                pass
        try:
            self.world.tick()
            self.tm.set_synchronous_mode(False)
            s = self.world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            self.world.apply_settings(s)
        except RuntimeError:
            pass


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--towns", nargs="*", default=None)
    ap.add_argument("--episodes-per-town", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--dataset-root", default=None, help="override paths.dataset_root (e.g. a trial directory)")
    ap.add_argument("--weather", default=None, help="CARLA preset name or custom condition (fog); overrides simulation.weather")
    ap.add_argument("--replay", nargs="*", default=None, help="episode ids (Town_epKK_seedS) to re-run with the same town/seed, e.g. under another weather")
    ap.add_argument("--tag", default="", help="suffix appended to replayed episode ids")
    ap.add_argument("--rig", choices=("interfuser", "transfuser", "both"), default="interfuser")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    sim = cfg["simulation"]
    if args.frames:
        sim["frames_per_episode"] = args.frames
    if args.weather:
        sim["weather"] = args.weather
    towns = args.towns or sim["towns"]
    per_town = args.episodes_per_town or sim["episodes_per_town"]
    upstream.bootstrap()
    import carla
    client = carla.Client(sim["host"], sim["port"])
    client.set_timeout(120.0)
    root = Path(args.dataset_root or cfg["paths"]["dataset_root"])
    summary_path = root / "episodes.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    if args.replay:
        jobs = []
        for eid in args.replay:
            town, rest = eid.rsplit("_ep", 1)
            k, seed = rest.split("_seed")
            jobs.append((town, int(k), int(seed), f"{eid}{args.tag}"))
    else:
        jobs = [(town, k, int(cfg["execution"]["seed"]) * 1000 + hash_town(town) * 100 + k + args.seed_offset, None)
                for town in towns for k in range(per_town)]
        jobs = [(t, k, sd, f"{t}_ep{k:02d}_seed{sd}") for t, k, sd, _ in jobs]
    for town, k, seed, ep_id in jobs:
        if True:
            out_dir = root / ep_id
            if (out_dir / "episode.json").exists() and json.loads((out_dir / "episode.json").read_text()).get("complete"):
                log.info("skip complete episode %s", ep_id)
                continue
            log.info("=== episode %s ===", ep_id)
            ep = Episode(carla, client, cfg, town, k, seed, out_dir, rig=args.rig)
            try:
                ep.setup_world(); ep.spawn_ego(); ep.build_route(); ep.spawn_traffic(); ep.attach_sensors()
                for _ in range(5):
                    ep.world.tick()
                    ep.buffer.read(ep.world.get_snapshot().frame)
                info = ep.run()
                with open(summary_path, "a") as f:
                    f.write(json.dumps(info) + "\n")
                log.info("episode %s: %d frames in %.0fs, collisions=%d, compass diff mean=%s",
                         ep_id, info["frames"], info["wall_seconds"], len(info["collision_frames"]),
                         info["compass_vs_theta_abs_diff_rad"]["mean"])
            except Exception:
                log.exception("episode %s failed", ep_id)
            finally:
                ep.cleanup()


def hash_town(town):
    return sum(ord(c) for c in town) % 97


if __name__ == "__main__":
    main()
