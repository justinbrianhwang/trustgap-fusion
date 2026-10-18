# RRF-InterFuser — reliability-aware robust fusion (research scaffold)

Question: when camera or LiDAR quality degrades, does steering InterFuser's fusion with a
task-aware reliability estimate (from the inputs only) make traffic-grid foreground detection more
robust than the same fusion fine-tuned with the same corruption augmentation?

Environment: CARLA 0.9.16, Python 3.12, torch 2.10 cu128, RTX 5090 (see `docs/DECISIONS.md` for
why the md's legacy stack was not used). Conda env: `rrf`.

## Layout
```
third_party/InterFuser   junction -> upstream clone (unmodified, commit f0be8ea2)
carla, weights           junctions -> CARLA_0.9.16, interfuser.pth
rrf/                     upstream.py (adapter), corruptions.py, collector.py, data.py, gated.py,
                         probes.py (masked reference + q targets), train.py, evaluate.py, metrics.py, tools/
configs/pilot.yaml       validated config
docs/                    DECISIONS, UPSTREAM_AUDIT, DATA_PROTOCOL, EVALUATION_PROTOCOL, BLOCKERS
artifacts/               environment.json, tensor_contract.json, manifests/, checkpoints/, runs/, metrics/
```

## Pipeline (env `rrf`, from the project root)
```powershell
python -m rrf.tools.doctor                       # artifacts/environment.json
python -m rrf.tools.smoke_forward                # strict load + synthetic forward + tensor contract
python -m pytest -q tests
# CARLA server: .\carla\CarlaUE4.exe -RenderOffScreen -quality-level=Epic
python -m rrf.collector                          # dataset/ (23 episodes x 300 frames)
python -m rrf.tools.make_splits                  # manifest + episode split (read-only afterwards)
python -m rrf.tools.evaluate --method B0 --suite smoke --split validation
python -m rrf.train --stage reference --seed 0
python -m rrf.train --stage reliability --seed 0
python -m rrf.train --stage fusion --method B2 --seed 0
python -m rrf.train --stage fusion --method P  --seed 0
python -m rrf.tools.evaluate --method B2 --suite full --split test
python -m rrf.tools.evaluate --method P  --suite full --split test
python -m rrf.tools.summarize --suite full --split test
python -m rrf.tools.operating_point --methods B0 B2 P
python -m rrf.tools.reliability_report --method P
python -m rrf.tools.figures --methods B0 B2 P
python -m rrf.tools.report
```
Or: `bash scripts/run_pilot.sh` (reference → reliability → B2 → P → evals → reports),
`scripts/run_parallel_evals.sh` (ablations B1/B3/B4 and seeds 1–2 on the test suite).
Results and their scope are written to `artifacts/REPORT.md`; every deviation from the md
prompt is in `docs/DECISIONS.md`.

## Extension (more data, gate-only, relative target)
`configs/extended.yaml` → `artifacts_ext/` (42 training episodes, same locked validation/test).
```powershell
bash scripts/run_extended.sh splits   # split locked to the pilot's validation/test
bash scripts/run_extended.sh train    # reliability (abs/rel) x3 seeds, B2/P/PG/PR x3, B3/B4
bash scripts/run_extended.sh eval
bash scripts/run_extended.sh report   # artifacts_ext/REPORT.md + docs/REPORT_NARRATIVE_EXTENDED.md
python -m rrf.closed_loop --config configs/pilot.yaml --ignore-signals --search 3   # drivable-route search
```
Follow-ups (token-scaling gate PS/PGS, relative-target reliability report, weather track and
closed loop for the extended checkpoints): `bash scripts/run_followups.sh` (stages
`scale_train`, `relrep`, `weather`, `scale_eval`, `report2` of `run_extended.sh` plus
`scripts/run_closed_loop_ext.sh`).

Third mechanism (key scaling with a learned exponent, PT/PGT): `run_extended.sh temp_train|temp_eval|report3`.

## Large data (278 training episodes) and weather-focused set
`configs/large.yaml` → `artifacts_large/` (`bash scripts/run_large.sh {splits|train|eval|report}`),
budget ablation `configs/large_3k.yaml` → `artifacts_large_3k/` (`scripts/run_large_3k.sh`),
collection `scripts/run_collect_large.sh` + `run_collect_retry.sh` (`dataset/` 305 episodes,
`dataset_weather2/` 24 routes × 4 weathers), weather-focused evaluation `scripts/run_weather2.sh`
(`evaluate_weather --frame-stride 4`, outputs `*_w2*`).

Result: the reliability estimate is learnable (H1) but no gating variant — additive log-prior,
token scaling or key scaling, joint or gate-only, absolute or relative target — beats the plain
augmentation baseline B2 at 12 or 42 episodes on the primary metric (H2/H3 rejected). At 278
episodes every fine-tuned fusion (B2 included) falls ~0.1 below the released model on the 4 locked
test episodes while frozen-fusion gates stay at its level; a budget ablation shows this is a
training-set/test-set effect, not the doubled budget, and checkpoint step predicts test AP
(Spearman −0.74) — the 2-episode validation split cannot see the drift. Details in
`docs/REPORT_NARRATIVE_EXTENDED.md` and `docs/REPORT_NARRATIVE_LARGE.md`; CSVs in
`artifacts_ext/csv/`, `artifacts_large/csv/`, `artifacts_large_3k/csv/`, `artifacts/csv/closed_loop_v2.csv`.

