### Extension: more data, gate-only training, relative target (2026-09-25)

Setup: 30 further episodes (train 12 → 42 episodes, 12,600 frames); validation and test episodes
identical to the pilot (locked). Budget 3000 steps, batch 8, lr 2e-5 for every fine-tuned method;
B2, P, PG (gate-only on the frozen released fusion), PR (relative target) with seeds 0/1/2; B3, B4
seed 0. Test suite, corruption seeds and samples identical to the pilot.

### Result on the locked test suite (grid_fg_ap, mean ± std over 3 seeds)

| method | clean | seen | unseen severity | unseen family | mixed |
|---|---|---|---|---|---|
| B0 (released) | 0.432 | 0.412 | 0.328 | 0.333 | 0.275 |
| B2 | 0.446 ± 0.031 | 0.430 ± 0.026 | 0.363 ± 0.010 | 0.359 ± 0.011 | 0.348 ± 0.009 |
| P | 0.377 ± 0.082 | 0.375 ± 0.080 | 0.351 ± 0.061 | 0.329 ± 0.061 | 0.327 ± 0.052 |
| PG (gate-only) | 0.421 ± 0.005 | 0.400 ± 0.005 | 0.305 ± 0.010 | 0.321 ± 0.008 | 0.266 ± 0.004 |
| PR (relative q*) | 0.378 ± 0.016 | 0.372 ± 0.015 | 0.351 ± 0.027 | 0.346 ± 0.024 | 0.325 ± 0.026 |
| B3 (seed 0) | 0.327 | 0.325 | 0.297 | 0.302 | 0.283 |
| B4 (seed 0) | 0.379 | 0.370 | 0.339 | 0.338 | 0.311 |

- **More data helps the plain baseline, not the gate.** B2 gains over the pilot (clean 0.420 →
  0.446, unseen severity 0.365 → 0.363, mixed 0.338 → 0.348) with a *smaller* seed spread, and now
  exceeds B0 on clean input. P − B2 per seed is −0.032 / +0.013 / −0.189 (clean) and −0.024 /
  +0.033 / −0.176 (seen): mean −0.069 ± 0.087 and −0.055 ± 0.088. Seed 2 of P never met the
  clean floor (fallback checkpoint, validation clean 0.596 vs floor 0.621) and seeds 0/1 selected
  at validation clean 0.635 / 0.630 versus 0.697–0.719 for B2 — i.e. the joint task + quality
  objective made fine-tuning *worse* and *less stable* than plain augmentation at this budget.
  **H2 is rejected, not merely unsupported, at 42 training episodes.** Every variant with the
  additive attention prior (P, PR, B3, B4) is below B2.
- **Gate-only (PG) is stable but does not add robustness.** With the fusion frozen, the three
  seeds agree to ±0.005, yet PG is at or below B0 on every kind (unseen severity 0.305 vs B0 0.328,
  mixed 0.266 vs 0.275). Re-weighting the released model's attention by an input-estimated q
  cannot recover what the fixed fusion does not represent; the small consistent loss suggests the
  prior mostly injects noise into attention.
- **Relative target (PR)** removes the scene-difficulty confound in q* but not the training
  problem: PR ≈ P on average (0.378 vs 0.377 clean) with a smaller spread; still below B2. Its q
  correlates only weakly with the *absolute* reference error used by the reliability report
  (Spearman camera 0.05–0.34, LiDAR ≈ 0), which is expected by construction; the report tool
  evaluates absolute targets only, so PR's reliability quality is not directly comparable.
- **H1 replicates for P and PG on foreground-present frames**: Spearman(q, −E) camera 0.50 / 0.54
  / 0.63 and LiDAR 0.57 / 0.48 / 0.59 for P, 0.47–0.49 / 0.51–0.57 for PG; ranking accuracy
  0.66–0.88. The estimator learns a usable modality-quality signal; using it in the attention
  prior is what does not pay off.
- **Operating point (precision 0.90 on the clean calibration split):** with more data B2's score
  channel is well behaved (threshold 0.509, clean FNR 0.488 vs 0.96 in the pilot), PR 0.520 /
  0.540, PG 0.508 / 0.633, B0 0.511 / 0.654; P and B3 again need thresholds ≈ 0.92 (FNR ≈ 0.93),
  i.e. their fine-tuned scores collapse towards 0/1.
- **Object-centre AP@1m does *not* follow the grid ranking** (`artifacts_ext/csv/object_ap.csv`,
  clean, mean over seeds): B4 0.409 (1 seed), P 0.374, PR 0.363, B3 0.353, B2 0.343, PG 0.282,
  B0 0.236. The gated fine-tuned models place their peaks closer to actor centres than B2 even
  though their cell-level ranking (grid AP) is worse — the additive prior seems to sharpen peaks
  at the cost of the score ordering across the grid. Cell-level AP is the pre-registered primary
  metric (md §12.1) and the object-level one is secondary with validation-free decoding
  parameters, so this discrepancy is reported as a caveat, not as support for H2; it is a real
  reason to look at object-level metrics in a follow-up.

### Closed loop v2 (signals ignored, 6 B0-selected routes, 120 s, pilot checkpoints, seed 0)

Mean route completion clean / blur s3 / dropout s3: B0 0.73 / 0.66 / 0.63, P 0.71 / 0.43 / 0.39,
B2 0.49 / 0.33 / 0.62; B2 registered the only two collision runs. Outcomes are route-dominated
(e.g. Town10HD: B0 0.98 in all three conditions, P 0.98 / 0.28 / 0.29, B2 0.61 / 0.07 / 0.98) and
single-seed, so they do not separate the methods; the fine-tuned models stop more often than B0
on phantom obstacles (`stuck_timeout` without collision). With the traffic-light head active, no
route was drivable for any method (route search: best 0.58 of 18), see DECISIONS. The extended
checkpoints were driven on the same routes in the follow-up below (section 8d of this report).

### Follow-ups (2026-09-25): token-scaling gate, relative-target report, weather, closed loop

**Second gating mechanism (PS joint / PGS gate-only, token scaling by q, 3 seeds, same budget):**

| method | clean | seen | unseen severity | unseen family | mixed |
|---|---|---|---|---|---|
| B2 | 0.446 ± 0.031 | 0.430 ± 0.026 | 0.363 ± 0.010 | 0.359 ± 0.011 | 0.348 ± 0.009 |
| PS (token scaling, joint) | 0.405 ± 0.020 | 0.399 ± 0.019 | 0.371 ± 0.003 | 0.355 ± 0.009 | 0.347 ± 0.008 |
| PGS (token scaling, gate-only) | 0.447 ± 0.002 | 0.421 ± 0.002 | 0.322 ± 0.002 | 0.349 ± 0.001 | 0.279 ± 0.001 |

- PS − B2 per seed (mean ± std): clean −0.041 ± 0.035, seen −0.031 ± 0.025, unseen severity
  +0.008 ± 0.013, unseen family −0.003 ± 0.018, mixed −0.001 ± 0.010. Token scaling is *much*
  more stable than the log-prior (P: −0.069 ± 0.087 clean) and matches B2 on every corruption
  kind, but it still loses on clean/seen input and never beats B2. All three PS seeds selected
  eligible checkpoints at validation clean / macro AP 0.73 / 0.72, 0.68 / 0.67, 0.72 / 0.72 — on
  par with B2's 0.72 / 0.72, 0.70 / 0.69, 0.72 / 0.70 — so on validation PS ≈ B2 and the clean
  deficit appears only on the test episodes (validation → test gap). **H2 stays rejected with the
  second mechanism.**
- PGS (frozen fusion) is at B2 level on clean (0.447) and *above* B0 on clean (0.432) and unseen
  family (0.349 vs 0.333), but at B0 level on unseen severity and mixed; two seeds never met the
  clean floor on validation (fallback checkpoints) yet all three agree to ±0.002 on test. Its q is
  ≈ 0.2–0.3 on clean input, which tracks the absolute target itself: on clean foreground-present
  test frames q* is 0.36 (camera) / 0.15 (LiDAR), because the single-modality references are poor
  at τ = 0.2. PGS therefore multiplies *all* tokens by ≈ 0.2–0.3 nearly uniformly; with the
  post-LayerNorm encoder a common token scale is largely cancelled and only the camera : LiDAR
  ratio plus an attention-temperature effect survive, which is why the frozen model is barely
  changed (PGS ≈ B0 on synthetic corruption). PS shows the same q level (0.18–0.27 on clean) while
  its fusion re-adapts to it. Spearman(q, −E) on foreground-present frames is 0.48–0.56 (camera) /
  0.54–0.66 (LiDAR) for both, ranking accuracy 0.72–0.83, MAE ≈ 0.2 / 0.12; some responses go the
  wrong way (PS seed 0: q_LiDAR 0.22 clean → 0.28 under LiDAR dropout s5, q_camera 0.24 → 0.08 when
  the LiDAR is silent), i.e. the estimator partly reads *scene* cues rather than sensor quality.
- Operating point (precision 0.90 on clean calibration): PS threshold 0.530 / clean FNR 0.56,
  PGS 0.502 / 0.61 — PS keeps a usable score channel (unlike P, threshold 0.92 / FNR 0.92).
- Object-centre AP@1m (clean, mean over seeds): PS 0.371, PGS 0.273 — PS joins P/PR/B4 above B2
  (0.343), PGS joins PG below it; the grid-vs-object discrepancy of the extension persists.

**Third gating mechanism: key scaling with a learned per-modality exponent (PT joint / PGT
gate-only; keys of modality m × q_m^γ_m, γ trained, values/queries untouched):**

| method | clean | seen | unseen severity | unseen family | mixed |
|---|---|---|---|---|---|
| B2 | 0.446 ± 0.031 | 0.430 ± 0.026 | 0.363 ± 0.010 | 0.359 ± 0.011 | 0.348 ± 0.009 |
| PT (key scaling, joint) | 0.373 ± 0.016 | 0.374 ± 0.018 | 0.362 ± 0.050 | 0.371 ± 0.031 | 0.333 ± 0.028 |
| PGT (key scaling, gate-only) | 0.202 ± 0.021 | 0.179 ± 0.012 | 0.134 ± 0.005 | 0.161 ± 0.014 | 0.087 ± 0.001 |

- PT − B2 per seed: clean −0.073 ± 0.019, seen −0.056 ± 0.021, unseen severity −0.001 ± 0.040,
  unseen family +0.012 ± 0.036, mixed −0.015 ± 0.022 — the same picture as PS (equal on
  corruption kinds, worse on clean/seen), with a larger spread on the corruption kinds. All three
  seeds selected eligible checkpoints (validation clean 0.72 / 0.70 / 0.67), and the **learned
  exponent moved from 1.0 to γ = 0.71 / 0.79 / 0.78 (camera) and 0.81 / 0.86 / 0.86 (LiDAR)**: given
  the freedom to choose how strongly q acts on the attention, joint training *weakens* the gate
  on every seed. PT's operating point degenerates like P's (threshold 0.89, clean FNR 0.91);
  object-centre AP@1m 0.372 (again above B2's 0.343).
- PGT collapses (clean 0.20, mixed 0.09; validation clean 0.23–0.25 vs the 0.62 floor on every
  seed, fallback checkpoints). With the fusion frozen, multiplying the keys by q^γ ≈ 0.3–0.4
  (q ≈ 0.25–0.35 on this suite, γ → 0.82–0.92) flattens the attention logits of the released
  encoder, which the frozen weights cannot compensate; the gate-only variant is only viable for
  a mechanism whose common scale cancels (token scaling, PGS) or that is additive (log-prior, PG).
- H1 holds for both (foreground-present Spearman camera 0.48–0.61, LiDAR 0.51–0.67; ranking
  0.66–0.84). Native weather (pooled, 3 seeds): PT night 0.55 / 0.55 / 0.62, rain 0.47 / 0.56 /
  0.65, fog 0.63 / 0.63 / 0.67 — on fog and rain the mean is above B2 (0.62 / 0.46) but the seed
  spread (up to 0.18 on rain) is far larger than any gap; PGT 0.52–0.55 everywhere. **H2 remains
  rejected with the third mechanism**; across the three mechanisms the only consistent effect of
  the gate under joint training is a clean-input deficit of 0.04–0.07 relative to B2.

**Relative-target reliability report (PR scored against its own target, test suite,
foreground-present, n = 14,028 per seed):** Spearman(q, −E_rel) camera 0.54 / 0.51 / 0.46, LiDAR
0.14 / 0.14 / 0.27; ranking accuracy 0.81 / 0.80 / 0.72 (4,228 pairs); MAE camera 0.13, LiDAR
0.07–0.15. The relative target itself is nearly saturated for LiDAR (mean q*_rel 0.96 vs camera
0.87): most LiDAR corruptions add little error *beyond* the clean LiDAR-only reference, so there is
little variance to rank — the low LiDAR correlation is a property of the target, not only of the
estimator. Against the *absolute* target PR's camera correlation was 0.05–0.34, so the earlier
statement that the two are not comparable is confirmed and resolved: PR does learn its own target
for the camera.

**Native-weather track, extended checkpoints (pooled grid AP on the 4 re-collected test
episodes, mean over 3 seeds; B0 single):** ClearNight B0 0.550, B2 0.564, P 0.525, PG 0.555, PR
0.550, PS 0.559, PGS 0.544; HardRainNoon B0 0.507, B2 0.455, P 0.442, PG 0.509, PR 0.463, PS
0.482, **PGS 0.557** (0.545 / 0.568 / 0.558 per seed); fog B0 0.575, B2 0.623, P 0.607, PG 0.578,
PR 0.596, PS 0.616, PGS 0.576. Every fine-tuned method with a trainable fusion (B2, P, PR, PS)
*loses* on rain relative to B0, whereas the two frozen-fusion gates (PG, PGS) keep or exceed B0;
PGS is the best method on rain by 0.05 with a consistent seed spread. Rain is a real, unseen
sensor degradation (camera streaks, LiDAR returns), so this is the one setting in the study where a
gate on the released model helps — but it is 4 episodes, the weather/ClearNoon groups differ in
traffic (see the caveat in 8b), and the same PGS is only at B0 level on synthetic corruption (mixed 0.279 vs
0.275, unseen severity 0.322 vs 0.328), far below B2. It is reported as a lead for a
weather-focused follow-up, not as support for H2/H3.

**Weather-focused set: 24 new routes (never in any split) × ClearNoon / HardRainNoon / fog /
ClearNight, every 4th frame, extended checkpoints (pooled AP, mean over 3 seeds; B0 single;
`artifacts_ext/csv/weather_w2.csv`):**

| method | ClearNoon (same 24 routes) | HardRainNoon | fog | ClearNight |
|---|---|---|---|---|
| B0 | 0.648 | 0.544 | 0.588 | 0.467 |
| B2 | 0.647 | 0.537 (0.585 / 0.511 / 0.516) | 0.633 | 0.565 |
| P | 0.643 | 0.556 | 0.667 | 0.559 |
| PS | 0.668 | 0.554 | 0.660 | 0.584 |
| PG (gate-only) | 0.633 | 0.533 | 0.561 | 0.470 |
| PGS (gate-only) | 0.663 | 0.567 (0.567 / 0.569 / 0.566) | 0.596 | 0.486 |

- **The rain lead of the frozen-fusion gates does not survive 6× more routes.** On the 4-episode
  set PGS beat B0 by 0.05 and B2 by 0.10 on rain; on 24 fresh routes PGS is 0.567 vs B0 0.544 and B2
  0.537 — +0.02 / +0.03, inside B2's seed spread (0.51–0.59). The 4-episode rain result was mostly
  scene content.
- On **fog and night the fine-tuned fusions (B2, P, PS) are clearly above B0** (fog +0.05 … +0.08,
  night +0.09 … +0.12) while the gate-only variants stay at B0: fine-tuning with synthetic
  corruptions transfers to real rendering changes, and the gate contributes nothing on top (P/PS ≈
  B2 ± 0.03 on every weather).
- On the 24 ClearNoon routes every extended model is at B0's level (0.633–0.668 vs 0.648), whereas
  on the 4 locked test episodes the same B2 was +0.014 and P/PS −0.04 … −0.07 relative to B0: the
  locked test set is small enough that its absolute differences of a few hundredths are
  sample-specific (see the large-data narrative for the same conclusion from the other side).

**Closed loop with the extended checkpoints (same 6 routes, signals ignored, 120 s, seed 0;
126 runs, `artifacts_ext/closed_loop/`, `artifacts_ext/csv/closed_loop.csv`).** Mean route
completion clean / blur s3 / dropout s3 (completed runs, runs with a collision, of 6):

| method | clean | blur s3 | dropout s3 | completed / collision runs (of 18) |
|---|---|---|---|---|
| B0 | 0.58 | 0.52 | 0.50 | 3 / 0 |
| B2 | 0.33 | 0.47 | 0.45 | 2 / 4 |
| P | 0.72 | 0.65 | 0.71 | 5 / 1 |
| PG | 0.67 | 0.76 | 0.58 | 4 / 0 |
| PR | 0.54 | 0.44 | 0.51 | 1 / 10 |
| PS | 0.77 | 0.74 | 0.71 | 5 / 0 |
| PGS | 0.76 | 0.76 | 0.69 | 4 / 0 |

- **The simulator session, not the method, sets the scale of these numbers.** B0 — identical
  checkpoint, routes, seeds, code and policy as closed-loop v2 — scored 0.58 / 0.52 / 0.50 here
  versus 0.73 / 0.66 / 0.63 in v2 (e.g. Town05 clean 0.03 now vs a completed drive before): the
  session-to-session spread of one method is ≈ 0.15, larger than most gaps between methods.
  Route outcomes are also bimodal and shared across methods (Town10HD clean: 0.98 for B0, P, PG,
  PS, PGS and 0.46–0.47 for B2 and PR; Town01: 0.72–0.78 for every method), i.e. a method mostly
  inherits the route's outcome and differs only where a single stop or collision flips it.
- Within that caveat, the gated fine-tuned models (PS, PGS, PG, P) drove *at least as well as B0*
  under all three conditions and better than the two methods without a gate (B2) or with the
  relative target (PR); B2 and PR are the only methods with repeated collisions (4 and 10 of 18
  runs), consistent with v2 where B2 had the only collisions. This is the opposite of the
  offline grid-AP ranking (B2 best), a single seed, six routes and one session, so it does not
  support H2/H3 either — it shows once more that the offline metric and the driving outcome are
  not one-to-one for this controller.

### Bottom line of the whole study

On CARLA 0.9.16 data with the released InterFuser checkpoint, a task-aware reliability estimate can
be learned from the inputs (H1, foreground-present frames; replicated for P, PG, PS, PGS and for
PR against its own relative target), but steering the fusion with it — as an additive attention
log-prior or as token scaling, jointly fine-tuned or gate-only, absolute or relative target — does
not improve traffic-grid foreground detection over the same-augmentation baseline at 12 or 42
training episodes on the primary cell-level metric (H2, H3 rejected; H4 holds only relative to
the also-fine-tuned baseline). Token scaling removes the instability of the log-prior (PS matches
B2 on every corruption kind with seed spread ≤ 0.01) but not the clean-input deficit; key scaling
with a learned exponent (PT) behaves like PS and its exponent shrinks during training, i.e. the
optimizer itself turns the gate down. The
secondary object-centre AP favours the gated models, and on native rain the frozen-fusion gates
(PG, PGS) beat every fine-tuned method — two leads that contradict the primary metric and that the
pre-registered protocol does not let us promote to a claim. The pilot's single favourable seed was
checkpoint luck. View-level gates are not evaluable under this protocol (whole-rig camera
corruptions make per-view quality identical by construction, see DECISIONS). The large-data run
(~300 episodes) and the weather-focused evaluation set are reported in `artifacts_large/REPORT.md`
and `docs/REPORT_NARRATIVE_LARGE.md`.
