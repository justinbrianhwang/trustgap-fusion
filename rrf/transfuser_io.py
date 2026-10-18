"""TransFuser 2022 checkpoint and dataset I/O (use from the ``rrf_tf`` environment).

``predict_tf`` returns tensors with a batch dimension: ``waypoints`` (B,4,2),
``boxes`` (B,K,8) as CenterNet [center_x, center_y, width, height, yaw, speed,
brake, score] in the 256x256 TransFuser BEV image, ``box_labels`` (B,K), and
``bev_logits`` (B,3,160,160). The 64x64 raw maps are ``center_heatmap`` (B,1),
``box_wh``/``box_offset`` (B,2), ``yaw_class`` (B,12), ``yaw_residual`` and
``box_velocity`` (B,1), and ``box_brake`` (B,2), with H,W appended.

``tf_targets`` uses the same box convention without score (N,7). World points
are rotated into the ego CARLA frame (forward, right), converted to the
TransFuser right-handed frame (forward, left), then mapped to BEV pixels by
``x_px=(right+16)*8, y_px=(32-(forward-1.3))*8`` (BEV origin = LiDAR, 1.3 m ahead of the vehicle centre).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from rrf import ROOT

TF_ROOT = ROOT / "third_party" / "TransFuser"
TF_SRC = TF_ROOT / "transfuser" / "team_code_transfuser"
VARIANTS = ("transfuser", "late_fusion", "geometric_fusion", "latentTF")


def bootstrap_tf():
    """Put the UI and upstream TransFuser/leaderboard sources on ``sys.path``."""
    root = str(TF_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from app.inference import (_drop_conflicting_transfuser_modules, _insert_sys_paths)
    from app.config import LEADERBOARD_DIR, SCENARIO_RUNNER_DIR, TRANSFUSER_SRC_DIR

    _insert_sys_paths([TRANSFUSER_SRC_DIR, LEADERBOARD_DIR, SCENARIO_RUNNER_DIR])
    _drop_conflicting_transfuser_modules(TRANSFUSER_SRC_DIR)
    return Path(TRANSFUSER_SRC_DIR)


def _checkpoint_path(variant, seed_file):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    directory = ROOT / "weights_tf" / variant
    if seed_file is None:
        files = sorted(directory.glob("*.pth"))
        if not files:
            raise FileNotFoundError(f"no checkpoint in {directory}")
        return files[0]
    path = Path(seed_file)
    if not path.is_absolute():
        path = directory / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_tf_model(variant, seed_file=None, device="cpu"):
    """Build one released variant and strictly load one seed checkpoint."""
    bootstrap_tf()
    import timm
    import torch
    from config import GlobalConfig
    from model import LidarCenterNet

    weights = _checkpoint_path(variant, seed_file)
    args = json.loads((weights.parent / "args.txt").read_text())
    config = GlobalConfig(setting="eval")
    for key in ("sync_batch_norm", "use_point_pillars", "n_layer", "use_target_point_image",
                "img_vert_anchors", "img_horz_anchors", "lidar_vert_anchors", "lidar_horz_anchors"):
        if key in args:
            setattr(config, key, bool(args[key]) if key in ("sync_batch_norm", "use_target_point_image") else args[key])
    config.backbone = args.get("backbone", variant)
    config.checkpoint_path = str(weights)
    config.checkpoint_args = args
    use_velocity = bool(args.get("use_velocity", True))

    # Every parameter is replaced by the strict checkpoint; avoid timm trying an online pretrained download.
    create_model = timm.create_model
    def create_uninitialized(*a, **kw):
        kw["pretrained"] = False
        return create_model(*a, **kw)
    timm.create_model = create_uninitialized
    try:
        model = LidarCenterNet(config, "cpu", config.backbone,
                               args.get("image_architecture", "resnet34"),
                               args.get("lidar_architecture", "resnet18"), use_velocity)
    finally:
        timm.create_model = create_model
    if config.sync_batch_norm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    # Released DDP files include a few training-only/aliased modules that the
    # evaluation constructor deletes. Restore them as unused modules so strict
    # loading still validates every checkpoint tensor.
    lidar_model = model._model.lidar_encoder._model
    stem_prefix = "_model.lidar_encoder._model.stem."
    if any(k.startswith(stem_prefix) for k in state) and not hasattr(lidar_model, "stem"):
        lidar_model.stem = torch.nn.Module()
    conv_key = stem_prefix + "conv.weight"
    if conv_key in state and not hasattr(lidar_model.stem, "conv"):
        weight = state[conv_key]
        lidar_model.stem.conv = torch.nn.Conv2d(weight.shape[1], weight.shape[0], weight.shape[2:],
                                                stride=lidar_model.conv1.stride,
                                                padding=lidar_model.conv1.padding,
                                                bias=stem_prefix + "conv.bias" in state)
    if stem_prefix + "bn.weight" in state and not hasattr(lidar_model.stem, "bn"):
        lidar_model.stem.bn = lidar_model.bn1
    if any(k.startswith("point_pillar_net.") for k in state) and not hasattr(model, "point_pillar_net"):
        from point_pillar import PointPillarNet
        model.point_pillar_net = PointPillarNet(config.num_input, config.num_features,
                                                 min_x=config.min_x, max_x=config.max_x,
                                                 min_y=config.min_y, max_y=config.max_y,
                                                 pixels_per_meter=int(config.pixels_per_meter))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.device = torch.device(device)
    return model, config


def load_raw_tf(dataset_root, sample_id):
    """Load the three lossless raw camera views, full-sweep LiDAR, and TF scalars."""
    episode, frame = sample_id.split("/", 1)
    directory = Path(dataset_root) / episode
    strip = np.asarray(Image.open(directory / "tf_rgb" / f"{frame}.png").convert("RGB"))
    if strip.shape[1] % 3:
        raise ValueError(f"TransFuser RGB strip width is not divisible by three: {strip.shape}")
    left, front, right = (view[:, :, ::-1].copy() for view in np.split(strip, 3, axis=1))
    tf_meas = json.loads((directory / "tf_measurements" / f"{frame}.json").read_text())
    return {"sample_id": sample_id, "rgb_left": left, "rgb_front": front, "rgb_right": right,
            "lidar": np.load(directory / "tf_lidar" / f"{frame}.npy").astype(np.float32, copy=False),
            "speed": float(tf_meas["speed"]), "target_point": np.asarray(tf_meas["target_point"], np.float32),
            "compass": float(tf_meas["compass"]), "gps": np.asarray(tf_meas["gps"], np.float64)}


def _scale_crop_view(array, config):
    image = Image.fromarray(array)
    width, height = image.width // config.scale, image.height // config.scale
    if config.scale != 1:
        image = image.resize((width, height))
    image = np.asarray(image)
    y, x = config.img_resolution[0], config.img_width
    return image[y:y + y, x:x + x]


def preprocess_tf(raw, config):
    """Return unbatched rgb (3,160,704), lidar (2,256,256), goal, and velocity tensors."""
    bootstrap_tf()
    import torch
    from data import draw_target_point, lidar_bev_cam_correspondences, lidar_to_histogram_features

    # HybridAgent receives CARLA BGR arrays and applies cv2.COLOR_BGR2RGB first.
    views = [_scale_crop_view(raw[f"rgb_{name}"][:, :, ::-1], config) for name in ("left", "front", "right")]
    stitched_image = Image.fromarray(np.concatenate(views, axis=1))
    stitched_image = stitched_image.resize((stitched_image.width // config.scale,
                                             stitched_image.height // config.scale))
    stitched = np.asarray(stitched_image)
    crop_h, crop_w = config.img_resolution
    y = stitched.shape[0] // 2 - crop_h // 2
    x = stitched.shape[1] // 2 - crop_w // 2
    rgb = np.transpose(stitched[y:y + crop_h, x:x + crop_w], (2, 0, 1)).copy()

    lidar_raw = np.asarray(raw["lidar"], dtype=np.float32)
    lidar_xyz = lidar_raw[:, :3].copy()
    lidar_xyz[:, 1] *= -1
    if config.backbone == "latentTF":
        lidar_bev = np.zeros((2, config.lidar_resolution_width, config.lidar_resolution_height), np.float32)
    else:
        lidar_bev = lidar_to_histogram_features(lidar_xyz)
    target_point = np.asarray(raw["target_point"], np.float32)
    batch = {"rgb": torch.from_numpy(rgb).float(), "lidar": torch.from_numpy(lidar_bev).float(),
             "target_point": torch.from_numpy(target_point).float(),
             "target_point_image": torch.from_numpy(draw_target_point(target_point)).float(),
             "velocity": torch.tensor([float(raw["speed"])], dtype=torch.float32)}
    if config.backbone == "geometric_fusion":
        bev_points, cam_points = lidar_bev_cam_correspondences(lidar_raw[:, :3].copy(), debug=False)
        batch["bev_points"] = torch.from_numpy(bev_points).long()
        batch["cam_points"] = torch.from_numpy(cam_points).long()
    return batch


def _batched(batch, device):
    import torch
    dims = {"rgb": 3, "lidar": 3, "target_point": 1, "target_point_image": 3,
            "velocity": 1, "bev_points": 4, "cam_points": 4}
    result = {}
    for key, value in batch.items():
        if key not in dims:
            continue
        value = value if torch.is_tensor(value) else torch.as_tensor(value)
        if value.ndim == dims[key]:
            value = value.unsqueeze(0)
        result[key] = value.to(device)
    return result


def _gather(feature, index):
    channels = feature.shape[1]
    flat = feature.permute(0, 2, 3, 1).reshape(feature.shape[0], -1, channels)
    return flat.gather(1, index.unsqueeze(-1).expand(-1, -1, channels))


def _decode_centernet(raw, config):
    import torch
    import torch.nn.functional as F

    heatmap, wh, offset, yaw_class, yaw_res, velocity, brake = raw
    pooled = F.max_pool2d(heatmap, config.center_net_max_pooling_kernel, stride=1,
                         padding=(config.center_net_max_pooling_kernel - 1) // 2)
    heatmap = heatmap * (pooled == heatmap)
    b, classes, h, w = heatmap.shape
    k = min(config.top_k_center_keypoints, classes * h * w)
    scores, flat_index = heatmap.reshape(b, -1).topk(k)
    labels = torch.div(flat_index, h * w, rounding_mode="floor")
    index = flat_index % (h * w)
    ys = torch.div(index, w, rounding_mode="floor").float()
    xs = (index % w).float()
    wh_g, off_g = _gather(wh, index), _gather(offset, index)
    yaw_cls = _gather(yaw_class, index).argmax(-1)
    yaw = yaw_cls.float() * (2 * math.pi / config.num_dir_bins) + _gather(yaw_res, index)[..., 0]
    yaw = torch.where(yaw > math.pi, yaw - 2 * math.pi, yaw)
    velocity_g = _gather(velocity, index)[..., 0]
    brake_g = _gather(brake, index).argmax(-1).to(velocity_g.dtype)
    x_ratio, y_ratio = config.lidar_resolution_width / w, config.lidar_resolution_height / h
    boxes = torch.stack([(xs + off_g[..., 0]) * x_ratio, (ys + off_g[..., 1]) * y_ratio,
                         wh_g[..., 0] * x_ratio, wh_g[..., 1] * y_ratio,
                         yaw, velocity_g, brake_g, scores], dim=-1)
    return boxes, labels


def predict_tf(model, batch):
    """Run the released network and return waypoints plus raw and decoded BEV heads."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    inputs = _batched(batch, device)
    with torch.inference_mode():
        lidar = inputs["lidar"]
        if model.use_target_point_image:
            lidar = torch.cat((lidar, inputs["target_point_image"]), dim=1)
        if model.backbone == "geometric_fusion":
            features, image_grid, fused = model._model(inputs["rgb"], lidar, inputs["velocity"],
                                                        inputs["bev_points"], inputs["cam_points"])
        else:
            features, image_grid, fused = model._model(inputs["rgb"], lidar, inputs["velocity"])
        waypoints = model.forward_gru(fused, inputs["target_point"])[0]
        head_lists = model.head([features[0]])
        raw = tuple(items[0] for items in head_lists)
        boxes, labels = _decode_centernet(raw, model.config)
        bev_logits = F.interpolate(model.pred_bev(features[0]),
                                   (model.config.bev_resolution_height, model.config.bev_resolution_width),
                                   mode="bilinear", align_corners=True)
    names = ("center_heatmap", "box_wh", "box_offset", "yaw_class", "yaw_residual", "box_velocity", "box_brake")
    result = {name: value for name, value in zip(names, raw)}
    result.update(waypoints=waypoints, boxes=boxes, box_labels=labels, bev_logits=bev_logits)
    return result


def tf_targets(dataset_root, sample_id, config):
    """Create vehicle boxes in the released CenterNet head's 256x256 BEV convention."""
    episode, frame = sample_id.split("/", 1)
    directory = Path(dataset_root) / episode
    meas = json.loads((directory / "measurements" / f"{frame}.json").read_text())
    actors = json.loads((directory / "actors_data" / f"{frame}.json").read_text())
    yaw = math.radians(float(meas.get("yaw_deg", math.degrees(float(meas["theta"]) - math.pi / 2))))
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rows = []
    for actor in actors.values():
        if int(actor.get("tpe", -1)) != 0:
            continue
        dx, dy = float(actor["loc"][0]) - float(meas["x"]), float(actor["loc"][1]) - float(meas["y"])
        forward = cos_yaw * dx + sin_yaw * dy
        right = -sin_yaw * dx + cos_yaw * dy
        if math.hypot(forward, right) < 0.1:  # the ego vehicle itself (vehicle-centre distance)
            continue
        # TransFuser's BEV is centred on the LiDAR, mounted lidar_pos[0] = 1.3 m ahead of the vehicle centre
        # (live check 2026-10-17: without this shift detections sat a constant 1.41 m behind the GT).
        forward -= float(config.lidar_pos[0])
        center_x = (right + 16.0) * config.pixels_per_meter
        center_y = (32.0 - forward) * config.pixels_per_meter
        if not (0.0 < center_x < config.lidar_resolution_width - 1 and
                0.0 < center_y < config.lidar_resolution_height - 1):
            continue
        ori = actor["ori"]
        actor_yaw = math.atan2(float(ori[1]), float(ori[0]))
        relative_yaw = (actor_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        extent = actor["box"]
        vel = actor["vel"]
        speed = sum(float(v) * float(o) for v, o in zip(vel, ori))
        rows.append([center_x, center_y, 2 * float(extent[1]) * config.pixels_per_meter,
                     2 * float(extent[0]) * config.pixels_per_meter, relative_yaw, speed, 0.0])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 7)
