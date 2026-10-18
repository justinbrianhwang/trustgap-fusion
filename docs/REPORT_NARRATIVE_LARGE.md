### Large-data experiment (2026-09-26): 278 training episodes, 6000 steps

Setup: 252 further ClearNoon episodes (42 new routes per town, seed offset 30) were added to
`dataset/` (305 episodes, 133 GB); 19 episodes are excluded for collisions, the 4 validation and 4
test episodes are the pilot's (locked), every other valid episode is train (278 episodes, 83,400
frames — 6.6× the extension). Because the training set is ~6× larger, the optimizer budget was
doubled for **every** method (6000 steps, batch 8, lr 2e-5, eval every 500). Methods: B2 / P / PS / PT
× seeds 0-2, PG / PGS / PGT / PR seed 0; reliability estimators re-trained on the large split
(foreground-present Spearman 0.56 camera / 0.80 LiDAR on validation, slightly better than the
extension's 0.48 / 0.76).

### Result on the locked test suite (grid_fg_ap, mean ± std over 3 seeds)

| method | clean | seen | unseen severity | unseen family | mixed |
|---|---|---|---|---|---|
| B0 (released) | 0.432 | 0.412 | 0.328 | 0.333 | 0.275 |
| B2 | 0.320 ± 0.019 | 0.312 ± 0.018 | 0.278 ± 0.013 | 0.287 ± 0.005 | 0.259 ± 0.009 |
| P | 0.326 ± 0.016 | 0.322 ± 0.019 | 0.299 ± 0.030 | 0.306 ± 0.029 | 0.276 ± 0.028 |
| PS (token scaling) | 0.335 ± 0.005 | 0.326 ± 0.006 | 0.294 ± 0.013 | 0.298 ± 0.015 | 0.284 ± 0.006 |
| PT (key scaling) | 0.307 ± 0.015 | 0.304 ± 0.020 | 0.296 ± 0.032 | 0.308 ± 0.025 | 0.254 ± 0.022 |
| PG (gate-only, seed 0) | 0.435 | 0.414 | 0.329 | 0.337 | 0.273 |
| PGS (gate-only, seed 0) | 0.434 | 0.410 | 0.320 | 0.349 | 0.272 |
| PGT (gate-only, seed 0) | 0.406 | 0.390 | 0.312 | 0.322 | 0.257 |
| PR (seed 0) | 0.309 | 0.302 | 0.276 | 0.289 | 0.254 |

- **Every method that fine-tunes the fusion is now far below the released model** — B2 included
  (clean 0.320 vs B0 0.432; the extension's B2 was 0.446). Six times more training data made the
  same-augmentation baseline *worse* on the locked test, not better. Only the frozen-fusion gates
  (PG, PGS) stay at B0's level (PG +0.003 clean, +0.004 unseen family; PGS +0.016 unseen family —
  within noise).
- Among the fine-tuned methods the gated ones are now slightly *above* B2 (PS − B2: clean +0.015 ±
  0.024, mixed +0.025 ± 0.016; P − B2: +0.006 … +0.022; PT − B2: −0.013 … +0.021), i.e. H2's sign
  flips relative to the extension — but every one of them is 0.10 below B0, so this ordering among
  degraded models carries no support for H2/H3. PS is again the most stable (seed spread ≤ 0.015).
- The learned key-scaling exponent went further down than in the extension: PT γ = 0.58 / 0.59 /
  0.72 (camera) and 0.68 / 0.70 / 0.81 (LiDAR); PGT (frozen fusion) drove γ to 0.09 / 0.15, i.e. it
  switched its own gate off and landed at 0.406 ≈ a slightly damaged B0. The optimizer, given the
  choice, turns the gate down every time.

### Why fine-tuning collapsed: drift the validation set cannot see

Validation numbers were *not* worse than in the extension (B2 selected at validation macro-AP
0.64 / 0.69 / 0.67 vs 0.72 / 0.69 / 0.70), yet test fell by 0.13. Across all 34 fine-tuned fusion
runs of the study (pilot, extension, large; every method with a trainable fusion) the selected
checkpoint's **training step predicts the locked-test clean AP with Spearman −0.74** (p < 1e-6):
mean test clean AP 0.447 for selections at ≤ 750 steps (n = 6), 0.390 at 751–1500 (n = 8), 0.350 at
1501–3000 (n = 15), 0.327 above 3000 (n = 5). Validation clean AP predicts test with only ρ = 0.47.
The model-selection validation split is 2 episodes (Town01, Town10HD; 200 frames) while the test
episodes are Town02–05; it therefore rewards continued fitting of the training distribution that
the test episodes penalise. The doubled budget and the larger set both pushed the selected step
later (B2: 2000 / 2500 / 2000 vs 1250 / 500 / 500 in the extension), which is the proximate cause
of the collapse. The 3000-step ablation on the same split (below) separates budget from data.

What "collapse" means for the traffic head: the operating point (precision 0.90 on the clean
calibration split) of every fine-tuned method degenerates (B2 threshold 0.919 / clean FNR 0.93, P
0.894 / 0.91, PT 0.912 / 0.93; PS 0.450 / 0.32 is the exception), i.e. the score channel becomes
near-binary, while the **object-centre AP@1m rises** (clean: B2 0.409, PS 0.425, P 0.413, PR 0.394,
PT 0.362 vs B0 0.236 and the extension's B2 0.343). Fine-tuning sharpens peaks at actor centres and
destroys the grid-wide score ordering that the pre-registered cell-level metric measures — the
two metrics disagree more, not less, with more data. This is the third time the study sees this
discrepancy and it is now the strongest single finding about the *metric*, not about the gate.

### Reliability (H1) at 278 episodes

Foreground-present Spearman(q, −E) on the test suite: P 0.47–0.50 / 0.40–0.57, PS 0.44–0.60 /
0.41–0.57, PT 0.42–0.56 / 0.51–0.57 (camera / LiDAR); ranking accuracy 0.75–0.89. Unchanged from
42 episodes: H1 holds, and does not improve with data.

### Native weather, 4 locked test episodes (pooled AP, mean over seeds; B0 single)

ClearNight: B0 0.550, B2 0.497, P 0.512, PS 0.515, PT 0.528, PG 0.564, PGS 0.546, PGT 0.542, PR 0.510.
HardRainNoon: B0 0.507, B2 0.401, P 0.402, PS 0.384, PT 0.513, **PG 0.510, PGS 0.551, PGT 0.567**,
PR 0.400. Fog: B0 0.575, B2 0.547, P 0.558, PS 0.564, PT 0.583, PG 0.583, PGS 0.570, PGT 0.566, PR
0.555. The extension's pattern repeats: fine-tuned fusions lose on rain (−0.10 vs B0), the
frozen-fusion gates keep or exceed B0.

### Weather-focused set: 24 new routes × ClearNoon / rain / fog / night (every 4th frame; large checkpoints, seed 0)

| method | ClearNoon (24 routes) | HardRainNoon | fog | ClearNight |
|---|---|---|---|---|
| B0 | 0.648 | 0.544 | 0.588 | 0.467 |
| B2 | 0.556 | 0.455 | 0.567 | 0.458 |
| P | 0.645 | 0.540 | 0.621 | 0.560 |
| PS | 0.572 | 0.500 | 0.595 | 0.518 |
| PT | 0.586 | 0.508 | 0.600 | 0.569 |
| PR | 0.563 | 0.467 | 0.561 | 0.465 |
| PG / PGS / PGT (gate-only) | 0.643 / 0.606 / 0.636 | 0.539 / 0.516 / 0.532 | 0.580 / 0.551 / 0.551 | 0.486 / 0.478 / 0.462 |
| *extension B2 / P / PS (3 seeds, same routes)* | *0.647 / 0.643 / 0.668* | *0.537 / 0.556 / 0.554* | *0.633 / 0.667 / 0.660* | *0.565 / 0.559 / 0.584* |

- The large-split B2 is also 0.09 below B0 on 24 *fresh* ClearNoon routes (0.556 vs 0.648), where the
  extension's B2 equalled B0 (0.647). The collapse is therefore not an artefact of the 4 locked test
  episodes: training on the 252 new routes made B2 generalise worse everywhere. P (seed 0) did not
  collapse on the fresh routes (0.645 clear, 0.621 fog, 0.560 night), PS/PT partly did; single seeds,
  so this is an observation, not a ranking.
- Fog and night again favour fine-tuned models over B0 when the fine-tuning did not collapse (P
  +0.03 / +0.09, PT +0.01 / +0.10); the gate-only variants stay at B0; the 4-episode rain lead of the
  gate-only variants is absent here (PGS 0.516 vs B0 0.544).

### Data audit: what differs in the 252 new episodes

Same collector, towns, weather, traffic counts (40 vehicles / 14 walkers spawned), no skipped
LiDAR sweeps, same compass/planner-frame agreement (max 4e-4 rad), same collision-exclusion rate
(13 %). What differs: the new routes are longer (656 vs 553 route points) and the ego drives faster
(mean 4.6 vs 3.4 m/s), and — decisive for a grid-detection target — they contain **half the
foreground per frame** (B0 audit on 12 + 12 training episodes, every 5th frame: 2,209 vs 4,430
foreground cells in 720 frames; B0 clean AP 0.482 vs 0.608, foreground-present AP identical 0.654 vs
0.660). The seed-offset-30 routes are sparser in near traffic than the pilot/extension routes and
the 4 test episodes (which come from the pilot's seed range). Training 278 episodes at this
positive rate shifts the traffic head's prior: the score channel binarises (operating thresholds
0.89–0.92) and the grid-wide ordering that AP measures degrades, on the dense test episodes and on
fresh routes alike. "More data" in this study therefore also meant "different foreground density",
which the collector does not control; a continuation must stratify routes by foreground density
(or sample routes from the same seed range as the test set) before scaling.


### Budget ablation: 3000 steps on the same 278-episode split (`artifacts_large_3k/`)

| method | budget | clean | seen | unseen severity | unseen family | mixed |
|---|---|---|---|---|---|---|
| B2 (42 episodes, extension) | 3000 | 0.446 ± 0.031 | 0.430 ± 0.026 | 0.363 ± 0.010 | 0.359 ± 0.011 | 0.348 ± 0.009 |
| B2 (278 episodes) | 3000 | 0.332 ± 0.016 | 0.321 ± 0.015 | 0.281 ± 0.012 | 0.288 ± 0.006 | 0.258 ± 0.010 |
| B2 (278 episodes) | 6000 | 0.320 ± 0.019 | 0.312 ± 0.018 | 0.278 ± 0.013 | 0.287 ± 0.005 | 0.259 ± 0.009 |
| PS (42 episodes, extension) | 3000 | 0.405 ± 0.020 | 0.399 ± 0.019 | 0.371 ± 0.003 | 0.355 ± 0.009 | 0.347 ± 0.008 |
| PS (278 episodes) | 3000 | 0.334 ± 0.028 | 0.328 ± 0.024 | 0.296 ± 0.015 | 0.297 ± 0.016 | 0.282 ± 0.018 |
| PS (278 episodes) | 6000 | 0.335 ± 0.005 | 0.326 ± 0.006 | 0.294 ± 0.013 | 0.298 ± 0.015 | 0.284 ± 0.006 |

- Halving the budget back to the extension's recovers only ≈ 0.01 (B2 0.320 → 0.332); seeds 1 and 2
  even select the *same* checkpoints (steps 2500 / 2000) under both budgets. **The collapse is caused
  by the training set, not by the doubled budget.** At matched selected steps the large split is
  0.04–0.08 below the extension (steps ≤ 1000: 0.450 vs 0.373; 1001–2000: 0.383 vs 0.325; > 2000:
  0.354 vs 0.319, clean AP over all fine-tuned methods), *and* within each split later selections
  are worse (Spearman(step, test) −0.84 in the extension, −0.60 in the 3000-step large run; 0.11
  in the 6000-step run whose selections are all late).
- Two effects therefore stack: (i) fine-tuning drifts away from the released weights with every
  step and the 2-episode validation split does not see it; (ii) the 252 new routes (same
  collector, towns, weather and traffic settings, seed offset 30) shift what the fusion learns in a
  way the 4 locked test episodes (seed offsets 0/10, like the pilot's training routes) penalise. The
  data audit below identifies (ii): the new routes carry half the foreground per frame, so the
  large split is *not* the same distribution as the pilot/extension routes and the test episodes.
  Consequence: **no absolute AP of any fine-tuned model in this study should be compared across
  data scales**, only methods within one scale; and the locked test set (4 episodes) plus the
  2-episode model-selection split are too small to detect either effect during training.


### Bottom line of the large-data run

Scaling the training set 6.6× did not help any method and hurt the fine-tuned fusions, for two
identifiable reasons that have nothing to do with the gate: checkpoint drift that the 2-episode
validation split cannot detect (Spearman(selected step, test AP) = −0.74 over 34 runs) and a
foreground-density shift in the newly collected routes. Within the large run the gated
fine-tuned models are marginally above B2 (+0.01 … +0.03) and the frozen-fusion gates equal B0;
none of this supports H2/H3, which stay rejected at 12, 42 and 278 episodes with three gating
mechanisms. H1 holds at every scale. The reliable positive findings of the whole study concern the
evaluation, not the method: cell-level grid AP and object-centre AP move in opposite directions
under fine-tuning, the 4-episode test set is sample-specific at the ±0.03 level, and fine-tuning
with synthetic corruptions transfers to native fog/night on fresh routes regardless of any gate.
