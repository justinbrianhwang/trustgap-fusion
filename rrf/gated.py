"""Reliability-aware gated InterFuser.

Wraps the unmodified upstream model. The token sequence, Q/K/V projections and weights are
untouched; we (1) recompute the per-view tokens so pre-fusion modality statistics are
available, (2) estimate q = (q_camera, q_lidar) from those statistics, and (3) add an additive
attention prior b_j = alpha * log(max(q[modality(j)], eps)) on every key token j in the
shared encoder self-attention and the decoder cross-attention, plus a hard mask for tokens of
unavailable physical sensors. With q = 1 and all sensors available the prior is exactly 0 and
the forward equals the upstream forward.

Second mechanism (extension, methods PS/PGS): ``mechanism="token_scaling"`` multiplies every
token of modality m (spatial + global, i.e. the encoder input, hence keys, values and queries)
by q_m instead of adding the log-prior; q = 1 is again the identity. Hard masks for
unavailable sensors are applied in both mechanisms.

Third mechanism (methods PT/PGT): ``mechanism="key_scaling"`` multiplies only the *keys* of
modality m by q_m ** gamma_m in the encoder self-attention and the decoder cross-attention
(values and queries untouched), i.e. a per-key attention temperature; gamma (one learnable
scalar per modality, init 1) sets how strongly q acts. q = 1 is the identity for any gamma.
"""
import torch
from torch import nn

from rrf import upstream

VIEWS = ("front", "left", "right", "center", "lidar")
PHYSICAL = ("front", "left", "right", "lidar")           # availability is per physical sensor
VIEW_SENSOR = {"front": "front", "left": "left", "right": "right", "center": "front", "lidar": "lidar"}
VIEW_MODALITY = {"front": 0, "left": 0, "right": 0, "center": 0, "lidar": 1}  # 0 camera, 1 lidar
NEG = -1e4  # finite "minus infinity" for hard masks (keeps softmax NaN-free)


# ------------------------------------------------------------------ tokenization (upstream order)
def tokenize(model, x):
    """Per-view tokens exactly as upstream forward_features builds them.
    Returns dict view -> {"spatial": [n,B,C], "global": [1,B,C], "raw": [B,C,h,w] (pre-embedding)}."""
    m = model
    out = {}
    specs = [("front", x["rgb"], m.rgb_patch_embed, 0), ("left", x["rgb_left"], m.rgb_patch_embed, 1),
             ("right", x["rgb_right"], m.rgb_patch_embed, 2), ("center", x["rgb_center"], m.rgb_patch_embed, 3),
             ("lidar", x["lidar"], m.lidar_patch_embed, 4)]
    for name, img, embed, v in specs:
        tok, glob = embed(img)
        raw = tok
        tok = tok + m.view_embed[:, :, v:v + 1, :] + m.position_encoding(tok)
        tok = tok.flatten(2).permute(2, 0, 1)
        glob = (glob + m.view_embed[:, :, v, :] + m.global_embed[:, :, v:v + 1]).permute(2, 0, 1)
        out[name] = {"spatial": tok, "global": glob, "raw": raw}
    return out


def assemble(tokens):
    """Concatenate in upstream order: front, left, right, center, lidar (spatial then global)."""
    seq, layout = [], []
    for v in VIEWS:
        seq += [tokens[v]["spatial"], tokens[v]["global"]]
        n = tokens[v]["spatial"].shape[0]
        layout += [v] * n + [v]
    return torch.cat(seq, 0), layout


# ------------------------------------------------------------------ statistics -> q
RAW_STAT_DIM = 6


def raw_statistics(x, availability):
    """Hand-crafted degradation cues from the *model input tensors* (never from raw files or
    metadata): per camera view mean, std and Laplacian variance (blur proxy) of the normalized
    image, averaged over available cameras; for the LiDAR BEV histogram the occupied-bin
    fraction, mean and max. [B, 2, RAW_STAT_DIM] (camera row, lidar row; zero padded)."""
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x["rgb"].dtype, device=x["rgb"].device).view(1, 1, 3, 3)

    def cam(img):  # [B,3,H,W] normalized
        gray = img.mean(1, keepdim=True)
        lv = torch.nn.functional.conv2d(gray, lap).flatten(1).var(1)
        return torch.stack([img.flatten(1).mean(1), img.flatten(1).std(1), lv], 1)  # [B,3]
    views = torch.stack([0.5 * (cam(x["rgb"]) + cam(x["rgb_center"])), cam(x["rgb_left"]), cam(x["rgb_right"])], 1)  # [B,3,3]
    w = availability[:, :3].unsqueeze(-1)
    c = (views * w).sum(1) / w.sum(1).clamp_min(1.0)
    c = torch.cat([c, (views * w + (1 - w) * 1e9).min(1).values.clamp_max(1e8)], 1)  # + per-view minimum (worst camera)
    h = x["lidar"].flatten(1)
    l = torch.stack([(h > 0).float().mean(1), h.mean(1), h.max(1).values], 1) * availability[:, 3:4]
    l = torch.cat([l, torch.zeros_like(l)], 1)
    return torch.stack([c, l], 1)


def modality_statistics(tokens, availability, x=None, add_raw=False):
    """Pre-fusion feature statistics per modality, [B, 2] x [B, 2C+2+RAW_STAT_DIM].
    Camera rig: front & center (same physical camera) are averaged first, then averaged with
    left/right over *available* cameras; unavailable sensors are excluded (not zero-filled
    into the mean). availability: float [B, 4] over PHYSICAL. The raw-statistics block is
    zero unless add_raw (config reliability.add_raw_statistics)."""
    def stats(raw):  # raw [B,C,h,w]
        f = raw.flatten(2)  # [B,C,n]
        mean, std = f.mean(2), f.std(2)
        norm = f.norm(dim=1)  # [B,n]
        return torch.cat([mean, std, norm.mean(1, keepdim=True), norm.std(1, keepdim=True)], 1)
    front = 0.5 * (stats(tokens["front"]["raw"]) + stats(tokens["center"]["raw"]))
    cams = torch.stack([front, stats(tokens["left"]["raw"]), stats(tokens["right"]["raw"])], 1)  # [B,3,D]
    w = availability[:, :3].unsqueeze(-1)
    cam = (cams * w).sum(1) / w.sum(1).clamp_min(1.0)
    lid = stats(tokens["lidar"]["raw"]) * availability[:, 3:4]
    feat = torch.stack([cam, lid], 1)  # [B, 2, D]
    raw = raw_statistics(x, availability) if (add_raw and x is not None) else feat.new_zeros(feat.shape[0], 2, RAW_STAT_DIM)
    return torch.cat([feat, raw], 2)


class ReliabilityEstimator(nn.Module):
    """Small MLP: normalized modality statistics + availability -> (q_camera, q_lidar) in (0,1)."""

    def __init__(self, stat_dim, hidden=128, layers=2):
        super().__init__()
        self.register_buffer("mu", torch.zeros(2, stat_dim))
        self.register_buffer("sigma", torch.ones(2, stat_dim))
        dims = [2 * stat_dim + 4] + [hidden] * layers
        mlp = []
        for a, b in zip(dims, dims[1:]):
            mlp += [nn.Linear(a, b), nn.GELU()]
        mlp.append(nn.Linear(dims[-1], 2))
        self.mlp = nn.Sequential(*mlp)

    @torch.no_grad()
    def fit_normalizer(self, stats):  # stats [N, 2, D] from the *training* split only
        self.mu.copy_(stats.mean(0))
        self.sigma.copy_(stats.std(0).clamp_min(1e-6))

    def forward(self, stats, availability):
        z = (stats - self.mu) / self.sigma
        return torch.sigmoid(self.mlp(torch.cat([z.flatten(1), availability], 1)))


class StaticGate(nn.Module):
    """B4: two learnable sample-independent scores."""

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(2))

    def forward(self, stats, availability):
        return torch.sigmoid(self.logit).unsqueeze(0).expand(stats.shape[0], 2)


# ------------------------------------------------------------------ gated model
class GatedInterFuser(nn.Module):
    def __init__(self, model, estimator=None, alpha=1.0, eps=1e-4, apply_encoder=True, apply_decoder=True, add_raw_statistics=False,
                 mechanism="log_prior", gate_enabled_by_default=True):
        super().__init__()
        assert mechanism in ("log_prior", "token_scaling", "key_scaling"), mechanism
        self.model = model
        self.estimator = estimator
        self.mechanism = mechanism
        self.key_gamma = nn.Parameter(torch.ones(2)) if mechanism == "key_scaling" else None
        if mechanism == "key_scaling":
            _install_key_scaling(model)
        self.alpha, self.eps = float(alpha), float(eps)
        self.apply_encoder, self.apply_decoder = apply_encoder, apply_decoder
        self.add_raw_statistics = add_raw_statistics
        self.gate_enabled_by_default = gate_enabled_by_default
        self.num_heads = model.encoder.layers[0].self_attn.num_heads

    # -- masks
    def key_bias(self, q, layout, availability):
        """[B, S] additive bias per key token: alpha*log(q_modality) and NEG for unavailable sensors."""
        B, S = q.shape[0], len(layout)
        mod = torch.tensor([VIEW_MODALITY[v] for v in layout], device=q.device)
        sensor = torch.tensor([PHYSICAL.index(VIEW_SENSOR[v]) for v in layout], device=q.device)
        bias = self.alpha * torch.log(q.clamp_min(self.eps))[:, mod]  # [B,S], differentiable in q
        avail = availability[:, sensor]  # [B,S]
        return torch.where(avail > 0.5, bias, torch.full_like(bias, NEG))

    def _expand(self, key_bias, L):
        """[B,S] -> [B*heads, L, S] as nn.MultiheadAttention expects (batch-major head blocks)."""
        B, S = key_bias.shape
        return key_bias[:, None, None, :].expand(B, self.num_heads, L, S).reshape(B * self.num_heads, L, S)

    def forward(self, x, availability=None, q_override=None, gate_enabled=None,
                token_means=None, replace_modality=None):
        m = self.model
        B = x["rgb"].shape[0]
        dev = x["rgb"].device
        if availability is None:
            availability = torch.ones(B, 4, device=dev)
        availability = availability.float()
        tokens = tokenize(m, x)
        features, layout = assemble(tokens)
        stats = modality_statistics(tokens, availability, x, self.add_raw_statistics)
        if q_override is not None:
            q = q_override
        elif self.estimator is not None:
            q = self.estimator(stats, availability)
        else:
            q = torch.ones(B, 2, device=dev)
        valid = availability.sum(1) > 0
        out = {"q": q, "stats": stats, "prediction_valid": valid, "layout": layout}
        if not bool(valid.any()):
            out["status"] = "ALL_SENSORS_UNAVAILABLE"
            return out
        S = features.shape[0]
        if replace_modality is not None:
            if replace_modality not in ("camera", "lidar") or token_means is None:
                raise ValueError("token_means and replace_modality='camera'|'lidar' are required together")
            means = token_means.to(device=features.device, dtype=features.dtype)
            if means.shape != (S, features.shape[2]):
                raise ValueError(f"token_means must have shape {(S, features.shape[2])}, got {tuple(means.shape)}")
            modality = 0 if replace_modality == "camera" else 1
            select = torch.tensor([VIEW_MODALITY[v] == modality for v in layout], device=dev)
            features = torch.where(select[:, None, None], means[:, None, :], features)
        if gate_enabled is None:
            gate_enabled = self.gate_enabled_by_default
        use_gate = gate_enabled and (self.estimator is not None or q_override is not None)
        if use_gate and self.mechanism == "token_scaling":
            mod = torch.tensor([VIEW_MODALITY[v] for v in layout], device=dev)
            features = features * q[:, mod].t().unsqueeze(-1)  # [S,B,1] scale per modality, differentiable in q
        m._rrf_key_scale = None
        if use_gate and self.mechanism == "key_scaling":
            mod = torch.tensor([VIEW_MODALITY[v] for v in layout], device=dev)
            scale = q.clamp_min(self.eps) ** self.key_gamma  # [B,2], differentiable in q and gamma
            m._rrf_key_scale = scale[:, mod].t().unsqueeze(-1)  # [S,B,1], read by the patched attention forwards
        kb = self.key_bias(q if (use_gate and self.mechanism == "log_prior") else torch.ones_like(q), layout, availability)
        needs_mask = use_gate or bool((availability < 0.5).any())
        enc_mask = self._expand(kb, S) if (needs_mask and self.apply_encoder) else None
        if needs_mask and not self.apply_encoder and bool((availability < 0.5).any()):
            enc_mask = self._expand(self.key_bias(torch.ones_like(q), layout, availability), S)
        memory = m.encoder(features, mask=enc_mask)

        tgt = m.position_encoding(torch.ones((B, 1, 20, 20), device=dev)).flatten(2)
        tgt = torch.cat([tgt, m.query_pos_embed.repeat(B, 1, 1)], 2).permute(2, 0, 1)
        L = m.query_embed.shape[0]
        dec_mask = self._expand(kb, L) if (needs_mask and self.apply_decoder) else None
        if needs_mask and not self.apply_decoder and bool((availability < 0.5).any()):
            dec_mask = self._expand(self.key_bias(torch.ones_like(q), layout, availability), L)
        hs = m.decoder(m.query_embed.repeat(1, B, 1), memory, memory_mask=dec_mask, query_pos=tgt)[0].permute(1, 0, 2)
        m._rrf_key_scale = None

        measurements, target_point = x["measurements"], x["target_point"]
        traffic_feature = hs[:, :400]
        wp_feature = hs[:, 401:411]
        waypoints = m.waypoints_generator(wp_feature, target_point)
        velocity = measurements[:, 6:7].unsqueeze(-1).repeat(1, 400, 32)
        out.update({
            "traffic": m.traffic_pred_head(torch.cat([traffic_feature, velocity], 2)),
            "waypoints": waypoints,
            "is_junction": m.junction_pred_head(hs[:, 400]),
            "traffic_light_state": m.traffic_light_pred_head(hs[:, 400]),
            "stop_sign": m.stop_sign_head(hs[:, 400]),
            "traffic_feature": traffic_feature,
            "features": features, "tokens": tokens,
        })
        return out


def _install_key_scaling(model):
    """Patch the encoder self-attention and decoder cross-attention *instances* so that the key
    tensor is multiplied by model._rrf_key_scale ([S,B,1]) when set. Parameters and state_dict
    keys are unchanged; the decoder's query self-attention is not touched."""
    attns = [l.self_attn for l in model.encoder.layers] + [l.multihead_attn for l in model.decoder.layers]
    for attn in attns:
        if getattr(attn, "_rrf_key_scaled", False):
            continue
        orig = attn.forward

        def fwd(query, key, value, *a, _orig=orig, _m=model, **kw):
            s = getattr(_m, "_rrf_key_scale", None)
            if s is not None:
                key = key * s
            return _orig(query, key, value, *a, **kw)
        attn.forward = fwd
        attn._rrf_key_scaled = True
    model._rrf_key_scale = None


# ------------------------------------------------------------------ freeze policy
TOKENIZER_PREFIXES = ("rgb_backbone", "lidar_backbone", "rgb_patch_embed", "lidar_patch_embed", "view_embed", "global_embed")
FUSION_PREFIXES = ("encoder", "decoder", "query_embed", "query_pos_embed", "traffic_pred_head")


def set_trainable(model, train_prefixes=FUSION_PREFIXES):
    """Freeze everything except the fusion parameters. Returns (n_trainable, n_total)."""
    n_tr = n_tot = 0
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(train_prefixes)
        n_tot += p.numel()
        n_tr += p.numel() if p.requires_grad else 0
    return n_tr, n_tot


def train_mode(model):
    """train() for fusion layers, eval() for the frozen tokenizer (BatchNorm stats must not move)."""
    model.train()
    for name, mod in model.named_modules():
        if name.startswith(TOKENIZER_PREFIXES):
            mod.eval()
    return model
