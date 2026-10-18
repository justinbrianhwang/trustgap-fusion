# Decisions

Assumptions and choices made during implementation, with the reason. Dates are absolute.

## 2026-09-24

- **CARLA 0.9.16 + Python 3.12 + torch 2.10 (cu128) instead of the md's CARLA 0.9.10.1 / Python 3.7 legacy env.**
  The machine has an RTX 5090 (sm_120); no legacy torch build runs on it and the only CARLA
  wheel present is `carla-0.9.16-cp312`. One env (`rrf`, clone of the user's `interfuser` env)
  serves both simulation and training. Consequence: the released checkpoint was trained on
  0.9.10.1 data; 0.9.16 rendering/LiDAR differ, so B0 clean performance on our data is a
  *diagnostic*, not a reproduction of the paper. Recorded in `docs/UPSTREAM_AUDIT.md`.
- **Upstream repo is used unmodified via a directory junction** (`third_party/InterFuser` →
  the user's clone at commit `f0be8ea2`, working tree differs from HEAD only by CRLF).
  Compatibility shims live in `rrf/upstream.py` (np.int alias, imgaug stub, no ImageNet
  download); the upstream files are not patched.
- **Data collection uses a lightweight collector (`rrf/collector.py`), not the leaderboard
  expert.** The 0.9.10 leaderboard/scenario_runner stack does not run on 0.9.16. The collector
  reproduces the expert's sensor rig and blueprint attributes (from
  `leaderboard/autoagents/agent_wrapper.py`), the `measurements`/`actors_data` schema needed
  by upstream GT generation, and drives the ego with CARLA's `BasicAgent` on a
  `GlobalRoutePlanner` route. Conventions follow the user's own 0.9.16 InterFuser adapters
  (MARSHAL `interfuser_model.py`, SD-2 `interfuser_record.py`).
- **Planner frame is computed from the world transform, not from GNSS.** In 0.9.10 the
  leaderboard's GPS→metric conversion yields `pos ≈ (-y_world, x_world)`; 0.9.16's GNSS flips
  the latitude sign (see SD-2 notes). To keep the checkpoint's input convention we define
  `gps_x, gps_y = (-y, x)` directly, `theta = yaw_rad + π/2` (upstream compass convention,
  north = -y). The IMU compass is also recorded and compared in the collector log.
  [to verify] validated by the GT-vs-LiDAR overlay figure in `artifacts/baseline_smoke/`.
- **Model inputs mirror the inference agent, not the training loader:** `command` and
  `target_point` come from the sparse global plan through `team_code.planner.RoutePlanner(4, 50)`
  as `interfuser_agent.py` does at test time. `future_waypoints` (auxiliary waypoint target)
  come from the dense route, as in the expert.
- **Traffic-only adapter:** `stop_sign` affordance and `is_red_light_present` are recorded
  (`ego.get_traffic_light_state()`), but the pilot trains/evaluates only the traffic-grid head.
  Auxiliary losses are switched off identically for every method.
- **LiDAR sweep phase (measured, 2026-09-24):** the leaderboard LiDAR runs at
  `rotation_frequency=10` while the simulation ticks at 20 Hz, so each measurement is a half
  sweep (all points have raw y ≤ 0 or all y ≥ 0). In the raw sensor frame at yaw −90,
  **+y = ego forward, +x = ego left** (verified by matching actor positions to point clusters
  across 30 frames; see `artifacts/baseline_smoke/overlay_trial`). Upstream's histogram keeps
  raw y ∈ [0, 28] after its y flip, i.e. only a front-half sweep carries information. The
  collector therefore captures only ticks whose measurement is a front-half sweep (every 2nd
  tick, phase-locked), stores the single raw measurement (no accumulation, like the agent), and
  records `rear_half_sweeps_skipped`. A first collection run without this check stored rear
  sweeps only (empty BEV) and was discarded.
- **Coordinate verification:** `rrf/tools/overlay.py` draws the GT grid and the B0 prediction
  on the LiDAR BEV. On trial sample `Town01_ep00_seed3600/0030` the oncoming car seen in the
  camera, its LiDAR cluster, the GT foreground cells and the B0 prediction (max 0.998)
  coincide. The IMU compass equals `yaw + π/2` to 1e-7 rad, confirming the theta convention.
- **Point dropout nesting:** the dropout uniform draw is seeded without severity so higher
  severities remove supersets of points (md §7.6). Other families include severity in the seed.
- **Reference probes → masked upstream reference (2026-09-24).** The md's CameraProbe/LiDARProbe
  (1 decoder layer on frozen tokens, 400 queries, balanced BCE) were implemented and trained on
  the 5 `probe_fit_groups` episodes: they overfit 8 fixed samples to AP 1.0 in 300 steps but
  reached only AP ≈ 0.10 on other frames of the *training* episodes and AP ≈ 0.01 (chance) on
  validation after 2000 steps — the 1,500 highly redundant frames are far too few to learn a
  detector from scratch. Replaced by a deterministic reference that needs no training: the
  frozen upstream model with the other modality hard-masked (test-time sensor removal, using the
  availability mask of `rrf.gated`). Measured on 120 validation samples: camera-only AP
  0.282 clean → 0.126 blur s5 / 0.078 noise s5 / 0.025 dark s5 / 0.011 blackout and exactly
  0.282 under every LiDAR corruption; lidar-only AP 0.472 clean → 0.051 dropout s5 / 0.354 jitter
  s5 / 0.061 empty and exactly 0.472 under every camera corruption. So E_camera/E_lidar respond
  only to their own modality. Because the reference is never trained on our data, the
  probe-holdout split is unnecessary and the reliability estimator uses all train episodes.
  Reference scores are upstream sigmoid outputs (soft-target regressors), not calibrated
  probabilities; q* remains a proxy tied to this reference. The md's naming distinction stands:
  this is "test-time sensor removal of the original model", reported as such, not a trained
  unimodal baseline.
- **Reliability estimator input: feature statistics + raw input cues (2026-09-24).** The
  feature-only estimator (md default: pooled token mean/std/norm statistics) trained to loss
  ~1e-3 but on validation reached MAE ≈ 0.20–0.26, Spearman(q̂, −E) ≈ 0.3–0.45 overall and
  **negative** (−0.1 to −0.4) on foreground-present frames: it learned scene priors (empty road ⇒
  high q*) rather than degradation, and overfit 12 training episodes. The pilot therefore adds
  the md's optional "raw-statistics auxiliary input" (`reliability.add_raw_statistics: true`):
  per-view mean/std/Laplacian variance of the normalized model-input images (mean and worst view)
  and occupied fraction/mean/max of the LiDAR BEV histogram — all computed from the model input
  tensors at inference, never from files or metadata. The feature-only run is kept as the failed
  ablation (numbers above, log in the report).
- **Quality loss only on foreground-present frames (2026-09-24).** With q* = exp(−E/τ) on all
  frames, a dead sensor on an empty road gets E ≈ 0 and q* ≈ 1 (the md's own caveat in §8.5).
  The estimator trained on all frames learned exactly that: validation q̂ = 0.96 for camera
  silent blackout (q* 0.39) and 1.0 for LiDAR silent empty. Such a q would make the gate *trust*
  a dead sensor. Since q* is uninformative without foreground, `L_quality` (reliability stage and
  method P) is applied only to samples whose GT has ≥1 foreground cell; evaluation still reports
  foreground-present and foreground-absent blocks. The all-frames run is kept as
  `artifacts/runs/ablation_allframes_reliability_seed0/`.
- **Fusion fine-tuning budget (2026-09-24):** with the md start values (lr 1e-4, batch 8, 3000
  steps ≈ 6.7 epochs over 3,600 frames) B2's training loss fell monotonically (0.057 → 0.026)
  while validation clean AP went 0.631 → 0.569 → 0.674 → 0.134 → 0.289 → 0.408 (steps 0–1250):
  memorization of 12 redundant episodes. Reduced to lr 2e-5, 2000 steps, eval every 200 steps
  for every fine-tuned method (B1–B4, P). Same budget for all; the aborted run is kept as
  `artifacts/runs/ablation_lr1e-4_B2_seed0/`.
- **Checkpoint rule fallback (2026-09-24):** B2 seed 1 never reached the clean floor (best
  validation clean AP 0.603 vs floor 0.621; trajectory 0.59 → 0.31 → 0.57 → 0.55 → 0.60 → 0.30 …).
  The rule is unchanged, but when no checkpoint is eligible the highest-clean-AP checkpoint is
  saved with `eligible: false` and the run's `final.json` records the fact, so the seed can still be
  evaluated and the trade-off shown instead of the seed silently disappearing. Per-seed training is
  strongly unstable at this data scale (validation AP swings of ±0.3 between evaluations 200 steps
  apart for every method); this is reported as a limitation, not smoothed over.
- **Foreground label:** `y_fg = target[:,0] > 0.01` (strict). MVTL1Loss uses `>= 0.01` for the
  foreground branch and `<= 0.01` for background (both include 0.01); we assign the boundary to
  background and document it.

## 2026-09-24 (optional tracks)

- **Native-weather track:** the four locked test episodes were re-collected with the same
  town/seed (same spawn, destination, Traffic-Manager seed) under `HardRainNoon`, `ClearNight` and a
  custom `fog` condition (ClearNoon + fog_density 70, fog_distance 10, fog_falloff 1, cloudiness 60;
  CARLA has no fog preset). LiDAR blueprint attributes are unchanged, weather is never an input.
  Because BasicAgent reacts to traffic, trajectories are not frame-paired; results are reported as
  an episode-group comparison against the ClearNoon counterpart (`rrf/tools/evaluate_weather.py`).
- **Object-centre AP:** peak extraction uses a fixed score threshold 0.05 and 2 m NMS, greedy
  one-to-one matching at 1 m and 2 m. GT centres are recovered from the soft target's offsets and
  the encode→decode round trip is unit-tested. No yaw/extent decoding (no BEV IoU) is attempted:
  the upstream head's sigmoid output range does not cover signed offsets symmetrically, which is
  documented rather than patched.
- **Closed loop:** the ego is driven by the evaluated model through the unmodified upstream
  tracker and controller (`rrf/closed_loop.py`), on the four test routes, with corruption injected
  once at sensor acquisition. Termination and failure rules are identical for every method:
  route completed (< 6 m from goal), stuck 20 s, off-route > 15 m, persistent collision, or 1600
  ticks (80 s). Route completion is the fraction of the dense route reached. Single seed,
  three conditions; not a leaderboard run, and no driving-safety claim follows from it.
- **Collection interruption:** the second collection batch was killed after 5 of 12 episodes when
  the pipeline processes were stopped; the dataset therefore has 23 episodes (20 valid). Earlier
  status messages that said 30/27 were wrong and have been corrected.

## 2026-09-24 (extension)

- **Dataset extension with a locked test set.** 30 further episodes (5 per town, seed offset 20)
  were collected with the unchanged collector. `configs/extended.yaml` (artifacts in
  `artifacts_ext/`) uses a split built with `make_splits --lock-from artifacts/manifests/split.json`:
  the 4 validation and 4 test episodes are copied unchanged, every other valid episode goes to
  train. Test numbers of the pilot and the extension are therefore on identical frames and seeds.
  Budget: 3000 optimizer steps (batch 8), eval every 250 — the same for every method.
- **PG — gate-only training.** Identical to P, but the fusion (encoder/decoder/query/traffic head)
  stays frozen at the released weights; only the reliability estimator (150k params) is trained by
  task + quality loss. Isolates the gate from fine-tuning drift.
- **PR — relative quality target.** q* = exp(−max(E_corrupt − E_clean, 0)/τ_rel), τ_rel = 0.1,
  where both errors come from the frozen masked reference on the same frame (clean counterpart
  used for the *target* only, never as input). Scene difficulty cancels, so q* is 1 for an intact
  sensor in a hard scene; the estimator stage is trained separately with `--target relative`
  (`reliability_rel_seed*.pt`). Compared against P (absolute target) at equal budget.
- **Closed-loop route search.** The pilot's test routes stall for every method on 0.9.16, so
  routes are searched among 18 new-seed routes (seed offset 100, 3 per town) with B0 clean; routes
  with route completion ≥ 0.8 are used for the extended closed-loop runs. Selecting routes on B0
  (not on any fine-tuned method) keeps the selection method-neutral.
- **Closed loop: traffic-light / stop-sign heads bypassed (`--ignore-signals`).** Route search with
  the stock controller found no route with B0 completion ≥ 0.8 (best 0.58 of 18); a diagnostic run
  showed the stall mechanism: the upstream traffic-light head outputs p(red) = 1.0 after ~18 s and
  the controller holds the car until its 1000-step red-light timeout, i.e. longer than the stuck
  timeout. The light head is trained on 0.9.10 renders and is not part of this research, so the
  extended closed loop drives "perception-only": the light and stop-sign inputs of the unmodified
  controller are fixed to green/no-stop for **every** method; the traffic-grid → tracker →
  controller path is untouched. Routes are then selected on B0 with this policy.
- **Closed-loop v2 routes.** With signals ignored, B0 reached ≥ 0.5 route completion on 7 of 18
  new-seed routes; the six with a max-ticks/completed status (Town01_ep02_seed3702,
  Town02_ep00_seed3800, Town02_ep01_seed3801, Town03_ep00_seed3900, Town05_ep02_seed4102,
  Town10HD_Opt_ep02_seed9402) are the closed-loop v2 route set; the limit is raised to 2400 ticks
  (120 s) because several routes are longer than 80 s of driving. Results in
  `artifacts/closed_loop_v2/` (pilot checkpoints) and `artifacts_ext/closed_loop/` (extended).


## 2026-09-25 (follow-ups)

- **Second gating mechanism: token scaling (PS / PGS).** The narrative listed "gating mechanisms
  other than the additive log-prior" as untested. `GatedInterFuser(mechanism="token_scaling")`
  multiplies every token of modality m (spatial + global, encoder input; hence keys, values and
  queries of the self-attention and the decoder memory) by q_m; q = 1 is the identity (unit test),
  hard masks for unavailable sensors are unchanged. Unlike the log-prior it is not invariant to a
  common factor of q, so the *level* of q matters, not only the ratio. PS = joint fine-tuning
  (same as P), PGS = gate-only (same as PG), same estimator checkpoints, budget, selection rule and
  seeds as the extension. No other mechanism (learned temperature, view-level gates) is evaluated.
- **Reliability report against the relative target.** `rrf.tools.reliability_report --target
  relative` recomputes E_rel = max(E_corrupt − E_clean, 0) on the test suite from the clean
  counterpart of each sample and scores q against q*_rel = exp(−E_rel/τ_rel); output
  `reliability_rel_<method>_seed<s>.json` / `reliability_rows_rel_*.csv`, `target` column in
  `reliability.csv`. Run for PR (its own target); the absolute report for PR stays as is.
- **Weather track for the extended checkpoints** (`artifacts_ext/metrics/weather_*`, all methods and
  seeds); same 12 re-collected episodes and the same episode-group caveat as the pilot.
- **Closed loop with the extended checkpoints** (`artifacts_ext/closed_loop/`): the six v2 routes,
  signals ignored, 2400 ticks, methods B0/B2/P/PG/PR/PS/PGS at seed 0. B0 is re-driven rather than
  copied from v2 so that all methods share one simulator session; the earlier statement that the
  extended checkpoints "were not run in closed loop" is superseded.

## 2026-09-25 (remaining items)

- **Third gating mechanism: key scaling with a learned per-modality exponent (PT / PGT).** Only the
  *keys* of modality m in the encoder self-attention and the decoder cross-attention are multiplied
  by q_m^γ_m (γ ∈ R², init 1, trained with the estimator's learning rate and stored in the
  checkpoint under `gate`); values, queries and the decoder's query self-attention are untouched.
  This is a per-key attention temperature — the "learned per-modality temperature" listed as
  untested. Implemented by patching the attention *instances* (state_dict keys unchanged, unit
  test checks identity at q = 1 for any γ and that the patch is inert when the gate is off).
  PT = joint fine-tuning, PGT = gate-only; same estimator checkpoints, budget, selection rule.
- **View-level gates are not evaluated, on purpose.** The corruption protocol applies camera
  corruptions to the whole rig (`primary_camera_scope: whole_rig`, master prompt line 1422), so the per-view quality
  of front/left/right/center is identical by construction and a view-level gate reduces to the
  modality-level gate on this test suite. Evaluating view-level gates needs a per-view corruption
  protocol, i.e. a different (unlocked) test suite; that is a protocol change, not a follow-up.
- **Weather-focused evaluation set (`dataset_weather2/`).** 4 new routes per town (seed offset 300,
  never in any split) collected in ClearNoon, then replayed under HardRainNoon, fog and ClearNight
  (96 episodes). Evaluated with `evaluate_weather --weather-root dataset_weather2 --clear-root
  dataset_weather2 --out-tag _w2`; same episode-group caveat as the first weather track (traffic
  reacts differently under each weather), but 6× the episodes and routes unseen by every method.
- **Large training set (`configs/large.yaml`, `artifacts_large/`).** 42 new routes per town (seed
  offset 30, 252 episodes) added to `dataset/`; split locked to the pilot's validation/test, every
  other valid episode is train (~260). Budget doubled to 6000 steps (eval every 500) for *every*
  method because the training set is ~6× larger; everything else identical to the extension.
  Methods: B2 / P / PS / PT × 3 seeds, PG / PGS / PGT / PR seed 0.
- **Budget ablation for the large run (`configs/large_3k.yaml`, `artifacts_large_3k/`).** With 278
  training episodes and 6000 steps every fine-tuned method, B2 included, fell far below B0 on the
  locked test (B2 clean 0.32 vs B0 0.43) while validation stayed at the extension's level. To
  separate "more data" from "more steps", B2 and PS are re-trained on the same large split with
  the extension budget (3000 steps, eval every 250), same seeds, reliability checkpoints copied.
- **Weather-focused set evaluated at 2.5 Hz.** On `dataset_weather2` (24 routes × 4 conditions × 300
  frames) the weather tool evaluates every 4th frame (`--frame-stride 4`) and evaluates the ClearNoon
  counterpart once per checkpoint instead of once per weather; consecutive 10 Hz frames are nearly
  duplicates, and the full-rate run of 21 checkpoints would have taken > 20 h. Extended checkpoints:
  all seeds for B0/B2/P/PG/PGS/PS, seed 0 for PR/PT/PGT/B3/B4; large checkpoints: seed 0 (outputs
  `weather_*_w2.json` / `weather_w2.csv`, `_w2b` for the single-seed extended group).
- **Data audit of the 252 new episodes (2026-09-26).** Identical collector settings, no skipped
  sweeps, same collision-exclusion rate, but the seed-offset-30 routes are longer, faster and hold
  half the foreground per frame (B0 audit: 2,209 vs 4,430 fg cells per 720 frames; fg-present AP
  equal). Recorded as the identified cause of the large-run collapse together with checkpoint
  drift; the large-run numbers are reported but marked as not comparable across data scales.

## 2026-10-16 (journal extension: two models)

- **Second model: TransFuser (2022).** Code junction `third_party/TransFuser` → the user's
  0.9.16 port `TransFuser_UI_V2`; checkpoints junction `weights_tf` → `models_2022`
  (transfuser, late_fusion, geometric_fusion, latentTF; same backbone and heads, different fusion
  blocks, used for a controlled fusion-structure comparison).
- **Separate environment `rrf_tf`** = clone of the user's `transfuser_ui` env (+ pytest, scipy).
  TransFuser needs PyPI timm, which would shadow InterFuser's vendored timm inside `rrf`; the two
  models therefore run in separate interpreters and share only stored data and CSVs.
- **Implementation split:** Codex CLI writes code from written task specs; Claude reviews the diffs
  against a pre-change snapshot, runs tests and experiments.
- **Dual-rig collector (`--rig both`), verified live 2026-10-16.** TransFuser's sensors (three
  960×480 FOV-120 cameras at yaw −60/0/60, a 20 Hz full-sweep LiDAR, GNSS, IMU) are spawned next to
  InterFuser's on the same ego and stored for the same simulator frame (asserted). Two fixes after the
  first live run: the TransFuser route planner read the route tuple wrongly; TransFuser's own
  `sensor_tick` values (IMU 0.05 s = fixed_delta) made the IMU skip a tick through float
  accumulation, so `sensor_tick` is not applied (synchronous mode emits every tick anyway). Live check
  on a trial episode: LiDAR BEV and ground-truth boxes align, the released TransFuser detects 18/20 GT
  vehicles within 2 m (score > 0.3, 30 sampled frames). Dual-rig episodes take ~61 s instead of ~30 s.
- **Journal dataset `dataset_v2/` (dual rig):** test v2 = 4 new routes per town (seed offset 500),
  val v2 = 2 per town (offset 600), train = replay of the 42 extension training routes. Both models are
  trained and evaluated on these drives; the old `dataset/` remains the source of the published
  pilot/extension/large results and of the legacy 4-episode test.
- **First-pass diagnostics** (`scripts/run_diagnostics_ext.sh`) run on the existing extension
  checkpoints and the legacy test (reduced suite, every 2nd frame) to locate where the
  reliability → utility → gate → gain chain breaks before any new training. The `meantoken` utility
  variant replaces a modality's tokens after q has been estimated from the original tokens.
- **TransFuser ground truth in the LiDAR frame (2026-10-17).** TransFuser's BEV (and its CenterNet
  boxes) is centred on the LiDAR, mounted `lidar_pos[0]` = 1.3 m ahead of the vehicle centre. The
  first version of `tf_targets` measured from the vehicle centre; detections then sat a constant
  1.41 m (std 0.06 m) behind the ground truth and AP@1m was 0.00 while AP@2m was 1.00. Fixed in
  `rrf/transfuser_io.py::tf_targets` (ego exclusion still on the vehicle-centre distance); clean
  AP@1m on the trial episode is now 0.95.
- **TransFuser cell-level metric uses CenterNet's label convention.** Positives = heatmap peak cells
  (rendered target ≥ 0.99), negatives = cells outside every Gaussian (< 0.01), tail cells ignored —
  for `heatmap_peak_ap` and for the q* reference error (`rrf/tf_probes.py`). With tails counted as
  foreground (≥ 0.01, InterFuser's grid convention) correct, sharp predictions scored a cell AP of
  0.06; with peaks 0.77 (0.98 on the trial episode at stride 10). `heatmap_ap_fg05` (≥ 0.5) is kept as
  a sensitivity variant. The two models' cell-level metrics therefore follow each model's own
  training target and are not compared across models in absolute terms.
- **Codex tasks C and D1 accepted after review.** C: role-based split v2 (`make_splits --by-role`),
  `--keep-all-checkpoints` + `rrf.tools.select_checkpoint`, method `BQ` (reliability head and quality
  loss, gate always off), `corruption.train_profile: weak`, `configs/journal*.yaml`. D1: `rrf/tf_gated.py`
  (three mechanisms per fusion stage, modality mask / mean replacement, attention capture),
  `rrf/tf_probes.py`, `rrf/tf_corruptions.py`, `rrf/tf_evaluate.py`. Diagnostics shuffle fix: the
  within-condition cross-episode permutation is now constructive (rotation by the largest episode),
  the rejection sampler never succeeded with 600 samples.
- **Split v2 built (2026-10-17), `artifacts_journal/manifests/split.json`:** train 41 (replays), validation
  v2 11, test v2 23; three Town10HD episodes excluded for collisions. Mean foreground cells per frame:
  train 4.32, validation 7.42, test 7.26 — validation and test match; the training routes are sparser
  (same direction as the large-run density shift). Kept as is for both models and analysed as a
  moderator; see `docs/PREREGISTRATION_JOURNAL.md` (draft awaiting author approval; no test v2
  evaluation before approval).
- **TransFuser fine-tuning throughput:** ~0.75 s/step at batch 8 with 4 loader workers (60-step trial).
  The velocity loss term starts high (1.04 → 0.07 in 50 steps): our actor speed (velocity projected on the
  actor's heading) differs from TransFuser's original label convention; it does not enter any reported
  metric.
- **Transient GPU out-of-memory (2026-10-17 02:54).** With nine processes on the GPU (3 InterFuser fusion,
  3 TransFuser reliability, 2 diagnostics), TransFuser reliability seed 0 died with CUDA OOM and the B3
  diagnostics with a cuDNN internal error at the same moment. No code change; the failed pieces are rerun
  one at a time (`scripts/run_journal_chain2.sh`), and every queue ends with a check that reruns runs
  without `final.json`. Reruns use the same seeds and data, so results are unaffected.
- **Gate oracles recomputed in one pass (2026-10-17).** Codex task F (temporal, leakage-free gate oracle)
  refactored the shared cross-fit code while a cross-fit job was running, so that job's outputs would mix two
  code versions; a cheap old-vs-new comparison was not possible (the old version fails at coarse frame
  strides). The job was stopped and all three gate-oracle variants (in-sample, cross-fitted, temporal) are
  recomputed with the final code for every extension seed-0 method (`scripts/run_oracles_ext.sh`), after the
  InterFuser fusion queue. The P cross-fit numbers reported on 2026-10-17 (macro 0.402 vs neutral 0.345)
  are preliminary until then.
- **InterFuser fine-tuning falls below the released model on validation v2 (2026-10-17).** All 45 InterFuser
  journal fusion runs (lr 2e-5) fell back: validation v2 clean AP ~0.50 for B2/BQ/B3/P/PS (B0 0.591), from the
  first evaluation (step 250) on; only gate-only PG keeps B0's level (0.585). TransFuser on the same data and
  schedule improves strongly (validation clean 0.572 → ~0.80, all runs eligible), so the data pipeline is not
  the cause. The old 2-episode validation had hidden this for InterFuser (consistent with the large-run drift).
  Before the pre-registration is approved (no test v2 evaluation so far), a validation-only learning-rate
  check for InterFuser B2 (1e-5, 5e-6; seeds 0, 1; `scripts/run_if_lr_sweep.sh`) decides whether the InterFuser
  protocol keeps lr 2e-5 (comparison among degraded fine-tuned models, reported as such) or switches to a
  lower lr for every InterFuser method.
- **Learning-rate check and regression test (2026-10-17): the InterFuser drop is genuine.** B2 at lr 1e-5 and
  5e-6 (seeds 0, 1) also falls back on validation v2 (best clean 0.508–0.536 vs B0 0.591). The step-250
  checkpoint (lr 2e-5) is below B0 even on its own training episodes (0.556 vs 0.610) and on both validation
  groups — this early dip also occurs in the historical extension run, which the current code reproduces
  exactly (0.631 → 0.564 → 0.600 at steps 0/250/500), so there is no code regression. With the 2-episode
  validation the extension run recovered above B0 (0.719 at step 1,250); on the 11-episode validation v2 no
  InterFuser fine-tuning run recovers. Proposed handling (awaiting author decision, pre-registration not yet
  approved): keep the lr 2e-5 InterFuser runs; report that InterFuser fine-tuning on this data volume does
  not reach the released model on representative validation; for InterFuser the primary gate comparison
  becomes gate-only (PG/PGS/PGT, fusion frozen) vs B0, joint methods vs B2 secondary; TransFuser keeps joint
  methods vs B2 as primary (fine-tuning helps there: validation clean 0.57 → ~0.80).

## 2026-10-18 (test-v2 evaluation)

- **TransFuser evaluation parallelised before any comparison was computed.** The first test-v2 run of the
  released TransFuser needed ~11 min per condition (~7–8 h per model, weeks for the ~70 TransFuser models)
  because every frame was prepared in one process. Per-frame preparation (loading, corruption, BEV histogram,
  ground-truth heatmap) now runs in DataLoader workers (`--workers`, default 8) with batch 16: ~48 s per
  condition (≈13× faster). Check on the reduced suite against the stopped run: heatmap peak AP 0.552791 vs
  0.552791, AP@1m 0.701217 vs 0.701237 (floating-point difference from the batch size). The pre-registered
  suite and frame stride are unchanged. `scripts/run_journal_test.sh` skips evaluations whose `metrics.json`
  exists and runs `TEST_LANES` evaluations in parallel (4 used). At the time of the change only the released
  models had test outputs; no method comparison had been computed.
