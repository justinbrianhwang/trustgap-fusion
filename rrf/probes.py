"""Modality-specific reference and task-aware quality targets.

Reference = the *frozen* upstream model with one modality hard-masked (camera-only: LiDAR
tokens removed as keys/values; lidar-only: all camera tokens removed), i.e. test-time sensor
removal of the original model. Its channel-0 traffic output is the per-cell foreground score.
It is deterministic, never trained on our data, and reacts only to its own modality
(verified in docs/DECISIONS.md). A trained 1-layer probe was tried first and did not
generalize on 12 training episodes, see DECISIONS.
"""
import torch

CAMERA_ONLY = [1.0, 1.0, 1.0, 0.0]
LIDAR_ONLY = [0.0, 0.0, 0.0, 1.0]


@torch.no_grad()
def masked_reference_probs(reference, batch):
    """reference: GatedInterFuser wrapping the frozen upstream model. Returns [B, 2, 400] with
    camera-only and lidar-only foreground scores (upstream channel 0, sigmoid output)."""
    B, dev = batch["rgb"].shape[0], batch["rgb"].device
    outs = []
    for avail in (CAMERA_ONLY, LIDAR_ONLY):
        a = torch.tensor([avail], dtype=torch.float32, device=dev).expand(B, 4)
        outs.append(reference(batch, availability=a, gate_enabled=False)["traffic"][..., 0])
    return torch.stack(outs, 1)


def reference_error(prob, y):
    """Per-sample balanced squared error E_m of reference scores against y_fg. [B,400] -> [B]."""
    y = y.float()
    pos, neg = y > 0.5, y <= 0.5
    e_pos = ((prob - 1) ** 2 * pos).sum(1) / pos.sum(1).clamp_min(1)
    e_neg = (prob ** 2 * neg).sum(1) / neg.sum(1).clamp_min(1)
    has_pos, has_neg = pos.any(1), neg.any(1)
    both = has_pos & has_neg
    return torch.where(both, 0.5 * e_pos + 0.5 * e_neg, torch.where(has_pos, e_pos, e_neg))


def quality_target(prob, y, tau=0.2):
    return torch.exp(-reference_error(prob, y) / tau)


@torch.no_grad()
def q_targets(reference, batch, fg, tau):
    """q* [B,2] and reference errors E [B,2] for the batch's (possibly corrupted) inputs."""
    probs = masked_reference_probs(reference, batch)
    e = torch.stack([reference_error(probs[:, i], fg) for i in range(2)], 1)
    return torch.exp(-e / tau), e


@torch.no_grad()
def q_targets_relative(reference, batch_corrupt, batch_clean, fg, tau_relative):
    """Relative target: only the error *added* by the corruption counts.
    E_rel = max(E(corrupt) - E(clean), 0); q* = exp(-E_rel / tau_relative). Scene difficulty
    (present in both) cancels, so a hard scene with intact sensors gets q* = 1. Returns
    (q* [B,2], E_rel [B,2], E_corrupt [B,2], E_clean [B,2])."""
    probs_c = masked_reference_probs(reference, batch_corrupt)
    probs_0 = masked_reference_probs(reference, batch_clean)
    e_c = torch.stack([reference_error(probs_c[:, i], fg) for i in range(2)], 1)
    e_0 = torch.stack([reference_error(probs_0[:, i], fg) for i in range(2)], 1)
    e_rel = (e_c - e_0).clamp_min(0.0)
    return torch.exp(-e_rel / tau_relative), e_rel, e_c, e_0

