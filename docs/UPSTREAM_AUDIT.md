# Upstream audit (InterFuser, commit f0be8ea294c49a59febaa032dd6da40b475e3657)

Source: `third_party/InterFuser` (junction to the user's clone; working tree = HEAD except CRLF).
Verified by running the code in env `rrf` on 2026-09-24 (`artifacts/tensor_contract.json`).

| item | location | verified fact |
|---|---|---|
| model registration | `interfuser/timm/models/interfuser.py:1067` | `interfuser_baseline`: embed 256, enc 6 / dec 6, 8 heads, ff 2048, r50 RGB backbone shared by 4 views, r18 LiDAR backbone, GRU waypoints |
| inputs | `interfuser.py:980` | `rgb 3x224x224`, `rgb_left/right/center 3x128x128`, `lidar 3x224x224`, `measurements 7` (6 one-hot command + speed), `target_point 2` |
| outputs | `interfuser.py:1064` | `traffic [B,400,7]` (sigmoid inside the head), `waypoints [B,10,2]`, `is_junction/traffic_light_state/stop_sign [B,2]` logits, `traffic_feature [B,400,256]` |
| token sequence | `interfuser.py:843-978` | order front(49+1), left(16+1), right(16+1), center(16+1), lidar(49+1) = 151 tokens, `[seq, batch, 256]`; derived dynamically in `rrf.gated.assemble`, never hard-coded |
| attention | `interfuser.py:389,465` | `nn.MultiheadAttention`, encoder `attn_mask` and decoder `memory_mask` accepted and unused (None) in the baseline; float masks of shape `[B*heads, L, S]` are added to the scores |
| rgb_center | `leaderboard/team_code/interfuser_agent.py:171,412` | derived from the front image with `CenterCrop(128)` and no resize; front uses `Resize((341,256))+CenterCrop(224)`, sides `Resize((195,146))+CenterCrop(128)`; ImageNet normalization |
| LiDAR histogram | `leaderboard/team_code/utils.py:6` | `crop` argument unused; bins: 224 over x∈[-28,29] (0.254 m), 224 over y∈[-28,0] (0.125 m); 3 channels (below/above z=-2, sum), clipped at 5 pts/bin and divided by 5 → values in [0,2] for the sum channel |
| LiDAR transform | `interfuser_agent.py:349-360` | `y *= -1` then `transform_2d_points(x, r, t, r, t)` which is the identity (tested) |
| traffic target | `interfuser/timm/data/det_utils.py:20` | 20x20 grid, cell (i,j) centre lateral `j-9.5`, forward `17.5-i`; channel 0 = `sqrt(0.5/max(0.5, dist))` soft score, others offsets/3.5, yaw/2, box/3.5 & /2, speed/8; cells with heatmap < 5% skipped |
| heatmap | `interfuser/timm/data/heatmap_utils.py:113` | vehicles+walkers within a 36 m box, traffic lights skipped (`FIXME`), boxes <1.5 m enlarged x1.5, ego removed, uses `np.int` (shimmed) |
| loss | `interfuser/train.py:801,1374` | `MVTL1Loss`: fg mask `>=0.01`, bg `<=0.01`; returns `(0.5*score+0.5*geometry, speed)`; total = `0.5*traffic + 0.2*waypoints + 0.05*velocity + 0.05*junction + 0.1*light + 0.01*stop` |
| expert sensors | `leaderboard/autoagents/agent_wrapper.py:104-160` | RGB: lens_circle_multiplier 3, falloff 3, chromatic 0.5; LiDAR: range 85, rot 10 Hz, 64 ch, fov 10/-30, 600k pps, atmosphere 0.004, dropoff 0.45/0.8/0.4 |
| expert save schema | `leaderboard/team_code/base_agent.py:405` | `measurements/%04d.json` keys gps_x, gps_y, x, y, theta, speed, command, x_command, y_command, future_waypoints, is_junction, is_*_present, affected_light_id; `actors_data/%04d.json` loc/ori/box/vel/tpe |
| checkpoint | `weights/interfuser.pth` | sha256 `39548dcfa56cdc06b8ccc52d7f6187a7d70ef36b05b7027db4d64a57ac3e30b2`, `state_dict` loads strictly (0 missing / 0 unexpected) |
| single-GPU DDP guard | `train.py` | upstream training is not used; `rrf/train.py` never initializes a process group |

Deviations and shims are listed in `docs/DECISIONS.md`.
