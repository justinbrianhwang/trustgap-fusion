"""TransFuser raw-input corruptions.

TransFuser stores three 960x480 camera views under the same ``rgb_left``,
``rgb_front`` and ``rgb_right`` keys used by :mod:`rrf.corruptions`, and its
LiDAR array is likewise an ``(N, >=3)`` full-sweep point cloud.  The shared
engine therefore already supplies the required whole-rig scope, stable seed
derivation and nested point-dropout severities; this module only gives that
contract a TransFuser-specific name.
"""
from rrf.corruptions import (CAMERA_FAMILIES, CAMERA_LEVELS, FULL_LOSS,
                             LIDAR_FAMILIES, LIDAR_LEVELS, condition_name,
                             corrupt_image, corrupt_lidar, corruption_seed,
                             corrupt_sample)


def corrupt_tf_sample(raw, spec, experiment_seed, repeat=0):
    """Apply the shared corruption engine to a raw TransFuser sample."""
    return corrupt_sample(raw, spec, experiment_seed, repeat)


__all__ = ["CAMERA_FAMILIES", "CAMERA_LEVELS", "FULL_LOSS", "LIDAR_FAMILIES",
           "LIDAR_LEVELS", "condition_name", "corrupt_image", "corrupt_lidar",
           "corruption_seed", "corrupt_tf_sample"]
