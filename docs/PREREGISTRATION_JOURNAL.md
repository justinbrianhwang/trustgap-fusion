# Pre-registration — TrustGap journal study

Revision 2, 2026-10-18 (revision 1 of 2026-10-17 revised after an external review; changes listed in §10).
No model has been evaluated on test v2. Training on train/validation has finished (see §9).

Approved: 2026-10-18 — first author (in chat); second author (relayed by the first author the same day)

## 1. Question
Is an accurate estimate of input reliability enough for robust camera–LiDAR fusion? If not, which of the
following is missing: the estimated target, the intervention path of the gate, or the training regime? The
chain studied, per model:

| link | question | primary evidence |
|---|---|---|
| S1 estimation | does q rank a modality's stand-alone error? | Spearman(q_m, −E_m) |
| S2 marginal utility | does q predict the direction and size of a *partial* change of that modality's gate weight? | partial-intervention analysis (§6.3) |
| S3 gate effect | does changing q change the intended modality's influence and the prediction? | intervention trace, oracles (§6.4) |
| S4 gain | is the full method better than the matched baselines? | confirmatory comparisons (§5) |

## 2. Models and their roles
| model | role | why |
|---|---|---|
| InterFuser | core, deep diagnosis | one joint softmax over all 151 camera and LiDAR tokens |
| TransFuser (2022, `transfuser`) | core, stage-wise intervention | 4 fusion stages; detection head reads the LiDAR branch that receives fused features at every stage |
| TransFuser `geometric_fusion` | additional structure comparison (released checkpoint, evaluation-only) | same sensors and heads, projection-based fusion without attention: log-prior/key gates not applicable |
| TransFuser `late_fusion` | waypoint-path diagnosis only | its detection head reads LiDAR-only features (`late_fusion.py:97-98`); a fusion gate cannot reach detection AP |
| TransFuser `latentTF` | optional camera-only baseline | LiDAR replaced by a positional input; excluded from S1/S2 |
Comparisons between released checkpoints are comparisons of model configurations including their pre-training
history, not strict structure controls. Within a model, every method shares data, initialisation, budget,
selection rule and trainable set (§3).

## 3. Data (fixed)
- `dataset_v2/`, dual sensor rig: both models are trained and evaluated on the same drives (replays of the
  42 extension routes as train; new routes for validation v2 and test v2). Every stored frame asserts that all
  sensors carry the same simulator frame id; ground truth comes from the same tick. Previous InterFuser
  results on `dataset/` are legacy.
- Split `artifacts_journal/manifests/split.json` (built by role, 2026-10-17): train 41, validation 11
  (model-select 5, calibration 6), test 23; three Town10HD episodes excluded for collisions.
- Ground truth does not depend on the input corruption: InterFuser's grid target and TransFuser's boxes come
  from `actors_data` and fixed region rules, never from (corrupted) LiDAR point counts.
- Foreground density (fg cells per frame): train 4.32, validation 7.42, test 7.26; per-episode density is a
  moderator (§6.7).

## 4. Methods
Full definitions with code references: `docs/METHODS_TABLE.md` (q input, estimator init, gate mechanism and
position, output range, use of L_quality, task→q gradient, L_quality→backbone gradient, trainable groups,
normalisation of frozen parts). Common: released checkpoint init; 3,000 steps, batch 8, fusion lr 2e-5,
estimator lr 3e-4; selection = max validation corruption macro-AP subject to clean AP ≥ step-0 clean − 0.01,
otherwise max clean AP (flagged "fallback"); backbones frozen with frozen normalisation statistics.

**Representative q-gate: P (additive log-prior).** Chosen a priori because it is the mechanism the study was
designed around and the only one whose effect is defined identically in both models (keys of a joint softmax
over both modalities); token scaling (PS) and key scaling (PT) are diagnostic variants. Claims about S4 are
limited to the tested configurations.

### 4.1 Confirmatory set (5 seeds each)
| id | description |
|---|---|
| B0 | released model |
| B2 | same-augmentation fine-tuning |
| BQ | B2 + reliability head and quality loss, gate off |
| B3A | router aligned with P: same estimator init, inputs, gate, gradients — no quality loss |
| B4 | input-independent learned static gate |
| P | reliability gate, log-prior |
| PG | P gate-only (fusion frozen) — 3 seeds trained so far; 2 more seeds to be trained |

### 4.2 Diagnostic set (exploratory, reported in full)
PS, PT, PGS, PGT, PR (relative target), B3 (random-init router), weak augmentation (B2, P, PS, PT),
TransFuser P at stage {0} and {3}; released `geometric_fusion` (evaluation-only).

## 5. Confirmatory comparisons and decision rules
Primary metric per model: InterFuser grid foreground AP; TransFuser heatmap peak AP (CenterNet positives).
Secondary: object-centre AP@1m (each model's own region) and, if feasible (`rrf/common_ap.py`), a common
vehicle-centre AP in the region covered by both models.

Per model, family F1 (primary metric, kinds: seen, unseen severity, unseen family, mixed, full loss):
1. P vs B2 (total gain of the method)
2. P vs BQ (does the gain come from the gate rather than the auxiliary loss)
3. P vs B3A (does reliability supervision add to a task-trained router)
4. P vs B4 (input-dependent vs static weighting)
5. PG vs B0 (gate on the unchanged released fusion)
For InterFuser, comparisons 1–4 are among fine-tuned models that all fall below B0 on validation v2
(DECISIONS 2026-10-17); comparison 5 is therefore reported first for InterFuser, without changing the family.

δ (smallest effect of practical interest), fixed a priori: **0.02 absolute AP on each model's primary metric**
(≈ 4–5 % relative at InterFuser's ~0.4–0.5 and ≈ 2.5 % at TransFuser's ~0.75 AP; the same absolute δ is used
for the secondary metrics). Variability is not used to set δ; if precision is insufficient the outcome is
"inconclusive".

Verdicts per comparison and kind (crossed bootstrap, §7): **improves** (95% CI lower bound > 0, Holm-adjusted
within the family); **worse** (95% CI upper bound < 0, Holm-adjusted); **equivalent** (90% CI within (−δ, +δ),
TOST at α = 0.05 per comparison, not multiplicity-adjusted, stated as such); **rules out δ** (90% upper < +δ,
not equivalent); **inconclusive** otherwise.

## 6. Diagnostic analyses (all models with a gate on the intervention path)
1. S1: Spearman(q_m, −E_m) on foreground-present frames, per kind. E_m = per-sample error of the **frozen
   released model** with the other modality removed (camera-only / LiDAR-only reference).
2. Removal utility U_m (sensitivity only): loss increase when the evaluated model loses modality m. For B0, U
   and E come from the same model and mask, so U_C − U_L = E_L − E_C holds by construction; cross-modality
   rankings of U vs E are therefore not evidence for S2 and are not reported as such. TransFuser:
   **sensor-level** removal (input replaced, whole network rerun) and **stage-level** removal (tokens/features
   of one branch suppressed at all stages) are reported separately; stage-level removal can leave the other
   modality's earlier fused information in place and is not called a sensor contribution. Two removal variants
   (hard mask, mean token from the training split, same positional/modality embeddings) are reported as a
   sensitivity pair; disagreement is reported as intervention dependence.
3. S2 (marginal utility): per sample, q_m multiplied by {0.5, 0.8, 1.25, 2.0} at the learned q;
   fraction of samples where down-weighting helps; Spearman(q_m, −ΔL_m(0.5)), Spearman(q_m, ΔL_m(2.0)); same
   against E_m.
4. S3: intervention trace (attention mass per modality and |Δ prediction| under q sweeps); reliability oracle
   (q*); gate-selection oracle over 16 fixed candidates + the sample's learned q (implementation check:
   oracle loss ≤ learned-q loss on every sample), in-sample, cross-fitted (cell halves) and temporal (choice on
   the previous frame of the same episode). Readings are possibilities, not verdicts: reliability oracle fails
   but gate oracle succeeds → the reliability target or the q→gate mapping may not identify the useful
   intervention; both fail → no improvement found for this checkpoint and candidate set. Per-frame loss gains
   are not upper bounds of AP. Gate-model-at-neutral-q is a different model from B2 and both are reported.
5. Shuffled q: global and within condition across episodes (q pairs moved together); evaluation-time
   replacement, not causal proof.
6. Conditions: single- vs both-modality corruption; seen vs held-out family; weak vs standard augmentation
   (Δ_weak − Δ_standard tested directly); gate-only vs joint; TransFuser stage {0}, {3}, all.
7. Moderator: per-episode foreground density in a regression of per-episode Δ.

## 7. Statistics
Crossed design: each training seed's model is evaluated on the same test episodes. Bootstrap replicates
resample training seeds and test episodes with replacement, independently, with the same draws for both
methods of a comparison (pairing kept; all conditions generated from one source episode move together; B0 has
one seed and is broadcast). Aggregation inside a replicate, fixed: per seed, pool predictions of the resampled
episodes → AP per condition → mean over the conditions of a kind (equal weights) → mean over seeds. No seed
ensembling. 2,000 replicates; Holm adjustment from bootstrap p-values (two-sided fraction of replicates beyond
0) within each family. With 5 seeds the seed resampling is coarse; this limitation is reported with every
interval. Test v2 is evaluated at frame stride 2 on the full suite.

## 8. Outcome branches
- Both models improve (P vs B2 "improves" on ≥ 3 of 5 kinds): positive result; diagnostics explain the size.
- One model improves: the paper centres on which chain link differs between the models.
- Neither improves and intervals are narrow (equivalent / rules out δ): diagnostic paper on where the chain breaks.
- Intervals too wide for a verdict: reported as inconclusive; no claim of absence.
- TransFuser data/ground-truth alignment fails: TransFuser performance comparisons are dropped (the structure
  comparison shares the pipeline and is dropped too). Alignment verified but gate integration fails (q = 1
  identity or response checks fail): the released-structure comparison is kept.
- Every executed experiment is reported, whatever its outcome. Deviations go to `docs/DECISIONS.md`.

## 9. Already executed before this revision (reported as such)
All runs of revision 1's table (both models, standard and weak augmentation, 2026-10-17/18), the validation-only
InterFuser learning-rate check, and diagnostics on the legacy extension checkpoints (legacy test). New runs
required by this revision: B3A (5 seeds, both models), PG seeds 3–4 (both models).

## 10. Changes from revision 1
Model roles (§2); S2 redefined as marginal utility, removal utility demoted with the B0 identity stated (§1,
§6.2–6.3); TransFuser sensor- vs stage-level removal (§6.2); oracle wording, learned q among candidates,
neutral ≠ B2 (§6.4); aligned router B3A and a methods table (§4); confirmatory set reduced to a fixed family
with a pre-chosen representative gate (§4–5); δ justified as a minimum effect of practical interest, not from
variability, and an inconclusive outcome (§5); crossed seed × episode bootstrap with a fixed aggregation order
(§7); outcome branches split (§8).
