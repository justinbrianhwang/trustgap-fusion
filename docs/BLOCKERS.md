# Blockers and open items

- **Checkpoint provenance:** `interfuser.pth` (sha256 39548dcf…) was downloaded by the user earlier;
  the README states it was trained on a subset of towns/weathers but not which. `pretraining_overlap:
  unknown`. Only "new episode / synthetic corruption" generalization is claimed.
- **CARLA version mismatch:** checkpoint trained on 0.9.10.1 data, evaluated on 0.9.16 data.
  B0 numbers are a diagnostic of transfer, not a reproduction of the paper.
- **Native-weather track and closed-loop evaluation:** not run (optional per md §13.3 / §14).
- **View-local gates, temporal corruptions, object-level AP:** not implemented (optional).
