# Evaluation protocol

Task: **InterFuser traffic-grid foreground detection robustness** (20×20 cells, forward −2…18 m,
lateral ±10 m). Not 3D object detection; `grid_fg_ap` is never reported as mAP.

## Metrics (`rrf/metrics.py`)
- `grid_fg_ap`: AP over all cells of a condition pooled, score = raw traffic channel 0 (pre-tracker),
  label = `target[:,0] > 0.01`. `None` when no positive cell. Per-episode AP stored as auxiliary.
- Fixed operating point: threshold = highest-recall threshold with precision ≥ 0.90 on the *clean
  calibration split*; applied unchanged to every test condition (`operating_point_unavailable` if
  none). FN/FP/FNR/FPR with `null` on zero denominators.
- Native losses: upstream MVTL1 score / geometry / speed L1 separately.
- Reliability: MAE(q̂, q*), Spearman(q̂, −E), modality ranking accuracy on pairs with |Δq*| ≥ 0.05,
  split by foreground present/absent. The estimator checkpoint is selected on validation
  foreground-present MAE (q* is uninformative without foreground, see DECISIONS).
- Efficiency: median/p95 latency per sample (GPU synchronized), peak GPU memory, trainable params.

## Conditions
- `smoke` (≤ 32 samples): clean, blur s3, dropout s3, both, camera blackout, lidar empty.
- `validation` (model selection): clean + 5 seen families at severity 2.
- `full`: clean; seen families s1–5 (noise, blur, darkening, dropout, jitter); unseen families s1–5
  (occlusion, sector dropout); 4 mixed; 2 silent full-loss. Each condition is tagged
  clean/seen/unseen_severity/unseen_family/mixed/full_loss and macro-averaged within its tag only.
- Every method is evaluated on the same samples with the same corruption seeds
  (`corruption.base_seed`); severity 4–5 and unseen families are never used in training.

## Fairness
B2/B3/B4/P share: initial checkpoint, split, corruption schedule (seeded by seed+sample+step,
not by method), full-loss augmentation rate, tokenizer freeze, trainable fusion set, optimizer
updates and effective batch, checkpoint rule (max validation corruption macro-AP subject to
clean AP ≥ step-0 clean AP − 0.01), evaluation samples and seeds. Probe/estimator training is an
extra cost of P and is reported separately.

## Statistics
Paired bootstrap over episodes (2000 resamples) of per-episode AP differences vs B2; seeds 0/1/2
reported individually and as mean ± std. Bootstrap CIs do not cover training-seed variance.

## Optional tracks (run after the pilot)
- Native weather (`rrf.tools.evaluate_weather`): rain / fog / night re-collections of the test
  routes, clean suite only, episode-group comparison with the ClearNoon counterpart.
- Object-centre AP (`rrf.tools.object_ap`): class-agnostic AP at 1 m / 2 m from stored predictions.
- Closed loop (`rrf.closed_loop`): route completion, collisions and termination status per run.
- CSV exports (`rrf.tools.export_csv` and the tools above) under `artifacts/csv/`.

