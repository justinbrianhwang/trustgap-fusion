"""Masked TransFuser references and task-aware reliability targets."""
import math

import torch

def gaussian_radius(height, width, min_overlap=0.1):
    """MMDetection CenterNet Gaussian radius used by TransFuser ``model.py:444``."""
    a1, b1 = 1.0, height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4.0 * a1 * c1))) / 2.0
    a2, b2, c2 = 4.0, 2.0 * (height + width), (1.0 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 4.0 * a2 * c2))) / 2.0
    a3, b3, c3 = 4.0 * min_overlap, -2.0 * min_overlap * (height + width), (min_overlap - 1.0) * width * height
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4.0 * a3 * c3))) / (2.0 * a3)
    return min(r1, r2, r3)


def _draw_gaussian(heatmap, x, y, radius):
    diameter = 2 * radius + 1
    coords = torch.arange(diameter, dtype=heatmap.dtype, device=heatmap.device) - radius
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    gaussian = torch.exp(-(xx * xx + yy * yy) / (2.0 * (diameter / 6.0) ** 2))
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if left + right > 0 and top + bottom > 0:
        patch = heatmap[y - top:y + bottom, x - left:x + right]
        kernel = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        torch.maximum(patch, kernel, out=patch)


def render_tf_heatmap(boxes, config, shape=(64, 64), device=None):
    """Render ``tf_targets`` boxes with TransFuser's CenterNet target Gaussian."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    heatmap = torch.zeros(shape, dtype=torch.float32, device=boxes.device)
    x_ratio = shape[1] / float(config.lidar_resolution_width)
    y_ratio = shape[0] / float(config.lidar_resolution_height)
    for box in boxes:
        x, y = int(box[0] * x_ratio), int(box[1] * y_ratio)
        if not (0 <= x < shape[1] and 0 <= y < shape[0]):
            continue
        radius = max(2, int(gaussian_radius(float(box[3] * y_ratio), float(box[2] * x_ratio), 0.1)))
        _draw_gaussian(heatmap, x, y, radius)
    return heatmap


PEAK, TAIL = 0.99, 0.01  # CenterNet convention: peak cells positive, Gaussian tails ignored, rest negative


def heatmap_labels(target):
    """(positive, negative) masks for a rendered TransFuser heatmap. TransFuser's CenterNet head is
    trained to peak only at object centres; counting Gaussian tail cells (>= 0.01) as foreground gave a
    cell AP of 0.06 for correct, sharp predictions (0.77 with peak cells), see DECISIONS 2026-10-17."""
    return target >= PEAK, target < TAIL


def heatmap_reference_error(probability, target_heatmap):
    """Balanced per-sample squared error of the heatmap probability vs. the binary peak labels:
    positives = peak cells (1), negatives = cells outside every Gaussian (0), tails ignored."""
    probability = torch.as_tensor(probability).flatten(1)
    target = torch.as_tensor(target_heatmap, device=probability.device).flatten(1)
    foreground, background = heatmap_labels(target)
    fg_error = ((probability - 1).square() * foreground).sum(1) / foreground.sum(1).clamp_min(1)
    bg_error = (probability.square() * background).sum(1) / background.sum(1).clamp_min(1)
    has_fg, has_bg = foreground.any(1), background.any(1)
    return torch.where(has_fg & has_bg, 0.5 * (fg_error + bg_error),
                       torch.where(has_fg, fg_error, bg_error))


@torch.no_grad()
def masked_reference_probs(reference, batch):
    """Return camera-only and LiDAR-only centre heatmaps as ``[B,2,H,W]``."""
    outputs = [reference(batch, modality_mask=modality, gate_enabled=False)["center_heatmap"][:, 0]
               for modality in ("lidar", "camera")]
    return torch.stack(outputs, 1)


@torch.no_grad()
def q_targets(reference, batch, target_heatmap, tau=0.2):
    probs = masked_reference_probs(reference, batch)
    errors = torch.stack([heatmap_reference_error(probs[:, i], target_heatmap) for i in range(2)], 1)
    return torch.exp(-errors / tau), errors


@torch.no_grad()
def q_targets_relative(reference, corrupt_batch, clean_batch, target_heatmap, tau=0.1):
    _, corrupt = q_targets(reference, corrupt_batch, target_heatmap, tau)
    _, clean = q_targets(reference, clean_batch, target_heatmap, tau)
    relative = (corrupt - clean).clamp_min(0.0)
    return torch.exp(-relative / tau), relative, corrupt, clean
