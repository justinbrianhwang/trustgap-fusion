"""Reliability-gated wrapper for the released TransFuser model.

The upstream fusion sequence is defined in
``third_party/TransFuser/transfuser/team_code_transfuser/transfuser.py``:
``GPT.forward`` flattens image anchors first and LiDAR anchors second and
concatenates them (lines 341-350). Velocity, when enabled, is a broadcast
embedding added to every token, not an extra token (lines 351-357). Each of
the four stages sends that whole sequence through ``SelfAttention``; its
single softmax is over the final key dimension (lines 513-522), so camera and
LiDAR keys compete in one normalization.

The returned sequence is split back into image and LiDAR maps (lines 361-365)
and each map is added residually to its own convolutional branch at all four
stages (lines 153-194). Thus stage-level removal means masking all of its
keys, zeroing its GPT input tokens so query residuals cannot leak the removed
features, and zeroing that branch immediately before the fusion residual is
added. A stage-level intervention does not rewind the retained branch, so at a
later fusion boundary that branch can still contain information fused at an
earlier boundary. Sensor-level removal instead replaces the camera input by a
black frame (which then undergoes normal image normalization) or the LiDAR
input by an empty BEV before either encoder runs. No upstream source is changed.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from rrf.gated import ReliabilityEstimator
from rrf.transfuser_io import _batched, _decode_centernet

NEG = -1e4
MODALITIES = ("camera", "lidar")
RAW_STAT_DIM = 6


def _feature_statistics(feature):
    flat = feature.flatten(2)
    norm = flat.norm(dim=1)
    return torch.cat((flat.mean(2), flat.std(2), norm.mean(1, keepdim=True),
                      norm.std(1, keepdim=True)), 1)


def raw_statistics(batch):
    """InterFuser-style camera sharpness/intensity and LiDAR occupancy cues."""
    image = batch["rgb"] / 255.0
    lap = image.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    views = torch.tensor_split(image, 3, dim=3)

    def camera(view):
        gray = view.mean(1, keepdim=True)
        lv = F.conv2d(gray, lap).flatten(1).var(1)
        return torch.stack((view.flatten(1).mean(1), view.flatten(1).std(1), lv), 1)

    cameras = torch.stack([camera(view) for view in views], 1)
    camera_cues = torch.cat((cameras.mean(1), cameras.min(1).values), 1)
    lidar = batch["lidar"].flatten(1)
    lidar_cues = torch.stack(((lidar > 0).float().mean(1), lidar.mean(1), lidar.max(1).values), 1)
    lidar_cues = torch.cat((lidar_cues, torch.zeros_like(lidar_cues)), 1)
    return torch.stack((camera_cues, lidar_cues), 1)


def modality_statistics(image_feature, lidar_feature, batch):
    """First-stage pre-fusion channel statistics plus raw-input cues."""
    return torch.cat((torch.stack((_feature_statistics(image_feature),
                                   _feature_statistics(lidar_feature)), 1),
                      raw_statistics(batch)), 2)


def _normalize_mask(mask, argument="modality_mask"):
    if mask is None:
        return set()
    values = {mask} if isinstance(mask, str) else set(mask)
    if not values <= set(MODALITIES):
        raise ValueError(f"{argument} must be 'camera', 'lidar', or an iterable of those")
    if len(values) == 2:
        raise ValueError("cannot mask both camera and lidar modalities")
    return values


class GatedTransFuser(nn.Module):
    """Run a released ``LidarCenterNet`` with runtime-only fusion interventions."""

    def __init__(self, model, estimator=None, mechanism="log_prior", stages=(0, 1, 2, 3),
                 alpha=1.0, eps=1e-4, gate_enabled_by_default=True):
        super().__init__()
        if mechanism not in ("log_prior", "token_scaling", "key_scaling"):
            raise ValueError(f"unknown gating mechanism: {mechanism}")
        stages = tuple(stages)
        if not set(stages) <= {0, 1, 2, 3}:
            raise ValueError("stages must be a subset of (0, 1, 2, 3)")
        if getattr(model, "backbone", None) != "transFuser":
            raise ValueError("GatedTransFuser requires the transfuser variant")
        self.model = model
        self.estimator = estimator
        self.mechanism = mechanism
        self.stages = stages
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.gate_enabled_by_default = bool(gate_enabled_by_default)
        self.key_gamma = nn.Parameter(torch.ones(2)) if mechanism == "key_scaling" else None
        self.mean_features = {}

    def set_mean_features(self, means):
        """Store ``{stage: {'camera': tensor, 'lidar': tensor}}`` replacement means."""
        self.mean_features = means
        return self

    @staticmethod
    def _replace(feature, mean):
        mean = torch.as_tensor(mean, dtype=feature.dtype, device=feature.device)
        if mean.ndim == 1:
            mean = mean[:, None, None]
        if mean.ndim == 3:
            mean = mean.unsqueeze(0)
        try:
            return mean.expand_as(feature)
        except RuntimeError as exc:
            raise ValueError(f"mean feature shape {tuple(mean.shape)} cannot replace {tuple(feature.shape)}") from exc

    def _attention(self, module, x, q_score, modality, apply_gate, hard_mask, capture):
        batch, tokens, channels = x.shape
        keys = module.key(x).view(batch, tokens, module.n_head, channels // module.n_head).transpose(1, 2)
        query = module.query(x).view(batch, tokens, module.n_head, channels // module.n_head).transpose(1, 2)
        value = module.value(x).view(batch, tokens, module.n_head, channels // module.n_head).transpose(1, 2)
        if apply_gate and self.mechanism == "key_scaling":
            scale = q_score.clamp_min(self.eps).pow(self.key_gamma)[..., modality]
            keys = keys * scale[:, None, :, None]
        logits = (query @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(keys.size(-1)))
        if apply_gate and self.mechanism == "log_prior":
            logits = logits + self.alpha * torch.log(q_score.clamp_min(self.eps))[:, modality][:, None, None, :]
        if hard_mask:
            unavailable = torch.zeros(tokens, dtype=torch.bool, device=x.device)
            for name in hard_mask:
                unavailable |= modality == MODALITIES.index(name)
            logits = logits.masked_fill(unavailable[None, None, None, :], NEG)
        attention = F.softmax(logits, dim=-1)
        if capture is not None:
            capture.append(torch.stack((attention[..., modality == 0].sum(-1),
                                        attention[..., modality == 1].sum(-1)), -1).mean((1, 2)).detach())
        result = module.attn_drop(attention) @ value
        result = result.transpose(1, 2).contiguous().view(batch, tokens, channels)
        return module.resid_drop(module.proj(result))

    def _gpt(self, gpt, image, lidar, velocity, q_score, stage, hard_mask, return_attention):
        batch = lidar.shape[0]
        image_h, image_w = image.shape[2:]
        lidar_h, lidar_w = lidar.shape[2:]
        image_tokens = image.permute(0, 2, 3, 1).reshape(batch, -1, gpt.n_embd)
        lidar_tokens = lidar.permute(0, 2, 3, 1).reshape(batch, -1, gpt.n_embd)
        if "camera" in hard_mask:
            image_tokens = torch.zeros_like(image_tokens)
        if "lidar" in hard_mask:
            lidar_tokens = torch.zeros_like(lidar_tokens)
        modality = torch.cat((torch.zeros(image_tokens.shape[1], dtype=torch.long, device=image.device),
                              torch.ones(lidar_tokens.shape[1], dtype=torch.long, device=image.device)))
        apply_gate = stage in self.stages
        tokens = torch.cat((image_tokens, lidar_tokens), 1)
        if apply_gate and self.mechanism == "token_scaling":
            tokens = tokens * q_score[:, modality, None]
        x = gpt.pos_emb + tokens
        if gpt.use_velocity:
            x = x + gpt.vel_emb(velocity).unsqueeze(1)
        x = gpt.drop(x)
        masses = []
        for block in gpt.blocks:
            y = block.ln1(x)
            x = x + self._attention(block.attn, y, q_score, modality, apply_gate,
                                    hard_mask, masses if return_attention else None)
            x = x + block.mlp(block.ln2(x))
        x = gpt.ln_f(x)
        split = image_tokens.shape[1]
        # Preserve upstream's unusual direct token-major view (transfuser.py:363-364).
        camera = x[:, :split].contiguous().view(batch, -1, image_h, image_w)
        lidar = x[:, split:].contiguous().view(batch, -1, lidar_h, lidar_w)
        return camera, lidar, masses

    def _forward_backbone(self, batch, q_override, gate_enabled, hard_mask,
                          modality_replace, mean_features, return_attention):
        backbone = self.model._model
        image = batch["rgb"]
        if backbone.image_encoder.normalize:
            from transfuser import normalize_imagenet
            image = normalize_imagenet(image)
        lidar = batch["lidar"]
        image = backbone.image_encoder.features.maxpool(backbone.image_encoder.features.act1(
            backbone.image_encoder.features.bn1(backbone.image_encoder.features.conv1(image))))
        lidar = backbone.lidar_encoder._model.maxpool(backbone.lidar_encoder._model.act1(
            backbone.lidar_encoder._model.bn1(backbone.lidar_encoder._model.conv1(lidar))))

        image_layers = [backbone.image_encoder.features.layer1, backbone.image_encoder.features.layer2,
                        backbone.image_encoder.features.layer3, backbone.image_encoder.features.layer4]
        lidar_layers = [backbone.lidar_encoder._model.layer1, backbone.lidar_encoder._model.layer2,
                        backbone.lidar_encoder._model.layer3, backbone.lidar_encoder._model.layer4]
        transformers = [backbone.transformer1, backbone.transformer2,
                        backbone.transformer3, backbone.transformer4]
        attentions = {}
        stats = q_score = effective_q = None
        for stage, (image_layer, lidar_layer, transformer) in enumerate(zip(image_layers, lidar_layers, transformers)):
            image, lidar = image_layer(image), lidar_layer(lidar)
            if stage == 0:
                stats = modality_statistics(image, lidar, batch)
                if q_override is not None:
                    q_score = torch.as_tensor(q_override, dtype=image.dtype, device=image.device)
                    if q_score.ndim == 1:
                        q_score = q_score.unsqueeze(0).expand(image.shape[0], -1)
                elif self.estimator is not None:
                    availability = image.new_ones((image.shape[0], 4))
                    if "camera" in hard_mask:
                        availability[:, :3] = 0
                    if "lidar" in hard_mask:
                        availability[:, 3] = 0
                    q_score = self.estimator(stats, availability)
                else:
                    q_score = image.new_ones((image.shape[0], 2))
                effective_q = q_score if gate_enabled else torch.ones_like(q_score)
            if modality_replace:
                stage_means = mean_features.get(stage, {})
                if modality_replace not in stage_means:
                    raise ValueError(f"missing stored stage-{stage} mean for {modality_replace}")
                if modality_replace == "camera":
                    image = self._replace(image, stage_means[modality_replace])
                else:
                    lidar = self._replace(lidar, stage_means[modality_replace])
            pooled_image, pooled_lidar = backbone.avgpool_img(image), backbone.avgpool_lidar(lidar)
            fused_image, fused_lidar, masses = self._gpt(transformer, pooled_image, pooled_lidar,
                                                         batch["velocity"], effective_q, stage,
                                                         hard_mask, return_attention)
            if return_attention:
                attentions[f"stage_{stage}"] = {f"layer_{i}": mass for i, mass in enumerate(masses)}
            fused_image = F.interpolate(fused_image, size=image.shape[2:], mode="bilinear", align_corners=False)
            fused_lidar = F.interpolate(fused_lidar, size=lidar.shape[2:], mode="bilinear", align_corners=False)
            if "camera" in hard_mask:
                image = torch.zeros_like(image)
            if "lidar" in hard_mask:
                lidar = torch.zeros_like(lidar)
            image, lidar = image + fused_image, lidar + fused_lidar

        image = backbone.change_channel_conv_image(image)
        lidar = backbone.change_channel_conv_lidar(lidar)
        features = backbone.top_down(lidar)
        image_grid = image
        image_global = torch.flatten(backbone.image_encoder.features.global_pool(image), 1)
        lidar_global = torch.flatten(backbone.lidar_encoder._model.global_pool(lidar), 1)
        return features, image_grid, image_global + lidar_global, q_score, stats, attentions

    def forward(self, batch, q_override=None, gate_enabled=None, modality_mask=None,
                modality_replace=None, mean_features=None, return_attention=False,
                sensor_remove=None):
        hard_mask = _normalize_mask(modality_mask)
        sensor_remove = _normalize_mask(sensor_remove, "sensor_remove")
        if modality_replace not in (None, *MODALITIES):
            raise ValueError("modality_replace must be 'camera' or 'lidar'")
        device = next(self.model.parameters()).device
        if gate_enabled is None:
            gate_enabled = self.gate_enabled_by_default
        inputs = _batched(batch, device)
        if "camera" in sensor_remove:
            inputs["rgb"] = torch.zeros_like(inputs["rgb"])
        if "lidar" in sensor_remove:
            inputs["lidar"] = torch.zeros_like(inputs["lidar"])
        lidar = inputs["lidar"]
        if self.model.use_target_point_image:
            lidar = torch.cat((lidar, inputs["target_point_image"]), 1)
        model_batch = {**inputs, "lidar": lidar}
        features, _, fused, q_score, stats, attention = self._forward_backbone(
            model_batch, q_override, gate_enabled, hard_mask, modality_replace,
            self.mean_features if mean_features is None else mean_features, return_attention)
        waypoints = self.model.forward_gru(fused, inputs["target_point"])[0]
        raw = tuple(items[0] for items in self.model.head([features[0]]))
        boxes, labels = _decode_centernet(raw, self.model.config)
        bev_logits = F.interpolate(self.model.pred_bev(features[0]),
                                   (self.model.config.bev_resolution_height,
                                    self.model.config.bev_resolution_width),
                                   mode="bilinear", align_corners=True)
        names = ("center_heatmap", "box_wh", "box_offset", "yaw_class",
                 "yaw_residual", "box_velocity", "box_brake")
        result = {name: value for name, value in zip(names, raw)}
        result.update(waypoints=waypoints, boxes=boxes, box_labels=labels,
                      bev_logits=bev_logits, q=q_score, stats=stats)
        if return_attention:
            result["attention"] = attention
        return result


__all__ = ["GatedTransFuser", "RAW_STAT_DIM", "ReliabilityEstimator",
           "modality_statistics", "raw_statistics"]
