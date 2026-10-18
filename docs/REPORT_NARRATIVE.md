### Completion level

- **Implementation complete** (code, config, 54 unit tests, docs).
- **Pilot complete**: B0, B1, B2, B3, B4, P trained/evaluated with seed 0; B2 and P with seeds 0, 1, 2.
- **Research validation: not complete.** Three seeds were run, but native-weather and closed-loop
  tracks were not, and the per-seed training instability described below makes the checkpoint
  selection itself a dominant source of variance.

### H1 — task-related reliability

**Partially supported, in the foreground-present regime only, and this part replicates across
seeds.** On the locked test episodes, method P's q (from the model inputs only) correlates with
the masked-reference error on frames that contain foreground: Spearman(q, −E) = 0.50 / 0.58
(camera / LiDAR) for seed 0, 0.53 / 0.64 for seed 1 and 0.28 / 0.55 for seed 2 — the LiDAR
side is stable, the camera side is not — with modality ranking accuracy 0.71–0.74 on ≈ 7,900
pairs, and the reference error decreases
monotonically across q quintiles
(camera E 0.45 → 0.20, LiDAR 0.49 → 0.32); the *final fused* score loss also decreases with
min(q) (Spearman 0.43). q_camera falls monotonically with camera severity for noise, blur,
darkening and occlusion and reaches 0.02 for a black image; q_lidar falls for dropout and reaches
0.04 for an empty cloud. Two failures: (1) on foreground-absent frames q carries no information
(Spearman ≈ 0), which is by construction of q* and is why those frames were excluded from the
quality loss; (2) modality attribution is imperfect — LiDAR point dropout lowers q_camera
(0.21 → 0.11) more than q_lidar, so the gate shifts attention *towards* the degraded LiDAR under
that corruption. q is not a calibrated probability; no ECE is reported.

### H2 — adaptive fusion beats the equally-augmented baseline

**Not supported.** Against B2 (same checkpoint, split, corruption schedule, freeze policy,
optimizer budget and selection rule), P's test macro-AP differences over three seeds are
+0.016 ± 0.032 (clean), +0.009 ± 0.033 (seen), −0.011 ± 0.045 (unseen severity),
−0.007 ± 0.032 (unseen family), −0.003 ± 0.028 (mixed). Seed 0 favours P on every condition
(clean +0.061; episode-paired bootstrap CIs exclude 0 for clean, noise s5, dropout s5, sector s5,
LiDAR empty), seeds 1 and 2 favour B2 by a similar margin (e.g. blur s5 −0.035 and −0.056). The
seed-0 advantage is also present on *clean* input, i.e. it reflects a better fine-tuned checkpoint
rather than corruption-specific adaptation. B4 (two learned constants, no input adaptation)
matches P's seed-0 clean/seen numbers (0.456 / 0.439), and B3 (same gate, no quality supervision)
equals B2 — so neither the input-adaptive gate nor the quality supervision produced a
reproducible gain at this scale.

### H3 — generalization to unseen corruptions

**Not supported.** On unseen families (occlusion, sector dropout) and unseen severities the mean
P − B2 difference is negative (−0.007, −0.011) with seed spread larger than the effect. Under
full sensor loss P ≈ B2 (blackout +0.005 / −0.065 / −0.016; empty +0.014 / 0.000 / +0.017).

### H4 — clean performance preserved

**Supported relative to B2, not relative to B0.** P's clean AP is not below B2's on average
(+0.016 ± 0.032) and the selection rule enforced the 1-pp validation floor. However every
fine-tuned method except B2-seed2, B4 and P-seed0/2 is below the released checkpoint's clean AP
(B0 0.432; B2 mean 0.420; P mean 0.436): fine-tuning on 12 episodes buys robustness at severity
4–5 and under full loss (B0 0.09–0.18 vs fine-tuned 0.24–0.29) at the cost of the mild regime
(B0 seen macro-AP 0.412 vs B2 0.411, P 0.420).

### Failure cases and limitations

- **Training instability dominates.** Validation clean AP of every fine-tuned method swings by
  ±0.3 between evaluations 200 steps apart (e.g. B2 seed 0: 0.63 → 0.57 → 0.67 → 0.13 → 0.29 →
  0.41 at lr 1e-4; still 0.60 → 0.31 → 0.57 → 0.55 → 0.60 → 0.30 at lr 2e-5 for seed 1). Seed 1
  never met the clean floor for either B2 or P (fallback checkpoints, flagged ineligible). The
  cause is the data scale: 12 training episodes × 300 frames at 10 Hz are highly redundant, so
  the model memorizes scenes. Any single-seed comparison at this scale is noise.
- **Distribution shift between splits.** Validation clean AP ≈ 0.63–0.74 versus test ≈ 0.39–0.46:
  the four test episodes are simply harder scenes, so the validation-selected checkpoint is not
  the best test checkpoint (no test-based re-selection was done, by protocol).
- **The learned probes of the md did not work** (validation AP ≈ chance after 2000 steps); the
  masked upstream reference replaced them (DECISIONS). q* therefore measures how much each
  modality helps *the frozen original model*, not an independent unimodal detector.
- **Reliability estimator generalization is weak:** a feature-only estimator learned scene priors
  (negative Spearman on foreground frames); raw input cues had to be added, and cross-modal
  attribution errors remain (see H1).
- **B2's operating point:** with the threshold calibrated for precision 0.90 on the clean
  calibration split, B2/B1/B3 need thresholds ≈ 0.998 (FNR ≈ 0.96–0.97), i.e. their score channel
  became sharply peaked after fine-tuning; P (0.516, FNR 0.56) and B0 (0.511, FNR 0.65) keep a
  usable operating point. This is a calibration side-effect worth noting but it is a single-seed
  observation.
- **Scope**: CARLA 0.9.16 data evaluated with a checkpoint trained on 0.9.10.1; synthetic
  corruptions only; grid-foreground AP only (no object-level AP, no closed loop, no weather
  track); one small dataset; no claim about driving safety, buses, SOTIF or fail-operational
  behaviour follows from any number here.

### Optional tracks (run after the pilot)

- **Native weather (rain / fog / night re-collections of the four test routes).** Within each
  weather group every method sees the same frames. Night: P 0.605 / 0.565 / 0.575 (seeds 0/1/2)
  vs B2 0.533 / 0.473 / 0.597, B0 0.550, B4 0.589; rain: P 0.540 / 0.503 / 0.526 vs B2 0.458 /
  0.519 / 0.562, B0 0.507; fog: P 0.637 / 0.623 / 0.636 vs B2 0.618 / 0.604 / 0.640, B0 0.575.
  P ≥ B2 in 6 of 9 seed-weather pairs (mean +0.05 at night, +0.01 rain and fog), consistent in
  direction with seed 0 of the synthetic track but again inside the seed spread. The weather groups
  have *higher* AP than their ClearNoon counterparts for every method because the re-collected
  drives contain different, closer traffic (foreground counts in `artifacts/REPORT.md` §8b), so no
  "weather effect" can be read from that comparison; the LiDAR is physically unaffected by CARLA
  weather, and q_lidar stays flat while q_camera drops for P (night 0.17 vs clean 0.22).
- **Object-centre AP@1m** (peaks + 1 m matching, `artifacts/csv/object_ap.csv`): the grid-AP
  ordering does not carry over. Clean: B2 0.351 (3-seed mean), P 0.337, B4 0.331, B1 0.331, B3
  0.328, B0 0.236; P − B2 per seed −0.013 / −0.010 / −0.025 averaged over all conditions. Every
  fine-tuned model produces 1.2–2× more peaks than B0 at the 0.05 threshold (more false
  positives at low score), which the cell-level AP rewards less than the object-level one.
- **Closed loop (3 methods × 4 test routes × clean / blur s3 / dropout s3, seed 0, 36 runs,
  `artifacts/csv/closed_loop.csv`).** The outcome is dominated by a route-level failure that is
  common to every method: on Town03, Town04 and Town05 the upstream controller stops the car
  within the first 2–33 % of the route in every condition (`stuck_timeout`, 20 s without motion,
  no collision) — B0 does so too, so this is the released controller/tracker on CARLA 0.9.16
  traffic, not the fusion change. Only the Town02 route is driven: route completion for clean / blur s3 /
  dropout s3 is B0 0.88 / 0.98 / 0.96 (no collisions), B2 0.98 / 0.62 / 0.28 (the clean run
  completed with 5 collision events; blur and dropout runs stalled), P 0.96 / 0.98 / 0.97 (no
  collisions). Mean completion over the 4 routes: B0 0.33 / 0.38 / 0.37, B2 0.34 / 0.27 / 0.13,
  P 0.30 / 0.39 / 0.33. With one informative route and a single seed this is a smoke check that the gated model
  drives through the unmodified controller under corruption; it supports no claim about driving
  safety or fail-operational behaviour.

### What would change the conclusion (answered by the extension, see docs/REPORT_NARRATIVE_EXTENDED.md)

More episodes (the estimator and the fusion fine-tuning both memorize at 12 episodes), a
lower-variance fine-tuning regime (or no fine-tuning of the fusion at all: gate-only training on
top of the frozen upstream fusion, which was not run), and a quality target that separates scene
difficulty from sensor quality (e.g. E_corrupt − E_clean).
