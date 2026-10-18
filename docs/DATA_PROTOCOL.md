# Data protocol

## Collection (`python -m rrf.collector`)
- CARLA 0.9.16, synchronous mode, `fixed_delta_seconds=0.05`, `no_rendering_mode=false`, weather `ClearNoon`.
- Fresh `load_world(town)` per episode; ego `vehicle.lincoln.mkz_2017` driven by CARLA `BasicAgent`
  (25 km/h) along a `GlobalRoutePlanner` route (≥150 m) to a random spawn point; 40 Traffic-Manager
  vehicles and 20 walkers, all seeded (`seed = 1000*execution.seed + 100*hash(town) + episode`).
- Sensor rig and blueprint attributes = leaderboard SENSORS track (see `docs/UPSTREAM_AUDIT.md`).
- Capture every 2nd tick (10 Hz) after 40 warm-up ticks, phase-locked to LiDAR front-half sweeps,
  300 frames per episode (30 s). Every stored frame has identical `world_frame` for all sensors
  (the buffer raises on a mismatch instead of substituting the nearest reading).

## Stored per frame (`dataset/<episode>/`)
```
rgb_front/%04d.png   800x600 RGB, lossless
rgb_left/%04d.png    400x300
rgb_right/%04d.png   400x300
lidar/%04d.npy       (N,4) float32 raw sensor frame (+x left, +y forward, z up, intensity)
measurements/%04d.json  upstream schema + world_frame, simulation_timestamp, source_frames, yaw_deg,
                        compass_imu, traffic_light_state, control, collision
actors_data/%04d.json   {id: loc, ori, box, vel, tpe} vehicles (0) / walkers (1) within 50 m
episode.json            settings, sensor attributes, route, seeds, collision frames, CARLA version
```
`rgb_center` is never stored; it is derived from `rgb_front` by upstream preprocessing.

## Ground truth
`rrf.upstream.traffic_grid_target` calls upstream `generate_heatmap` + `generate_det_data` on
deep copies → `(400,7)` soft target, cached under `targets/` with a content hash.
`y_fg = target[:,0] > 0.01`.

## Manifest and splits (`python -m rrf.tools.make_splits`)
- `artifacts/manifests/dataset_manifest.jsonl`: one line per sample with paths and sha256.
- Episodes stuck in a collision for > 25 % of ticks are excluded (reason recorded in the split file).
- Episode-level split (global seeded shuffle of valid episodes; towns cannot be stratified with 2–5 episodes each): test/validation/train ≈ 20/20/60 %; train is divided into
  `probe_fit_groups` (40 %) and `reliability_fit_groups` (60 %) — kept in the split file, but since
  the reference is the frozen (never fine-tuned) upstream model, both groups are used for the
  reliability estimator (see DECISIONS); validation into `model_select_groups` and
  `calibration_groups` (50/50). The split file is never overwritten.
- `pretraining_overlap: unknown` — the released checkpoint's training towns are not documented, so
  no "unseen town" claim is made; claims are limited to new episodes and synthetic corruptions.
