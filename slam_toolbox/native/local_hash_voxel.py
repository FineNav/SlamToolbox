from __future__ import annotations

from typing import Any, Mapping

from .backend import native_module


def create_engine(values: Mapping[str, Any]):
    module = native_module().local_hash_voxel
    config = module.Config()
    config.voxel_size = float(values["voxel_size"])
    config.max_range = float(values["max_range"])
    config.local_z_min = float(values["local_z_min"])
    config.local_z_max = float(values["local_z_max"])
    config.body_radius = float(values["body_radius"])
    ground_max = values.get("ground_protect_local_z_max")
    config.ground_protection_enabled = ground_max is not None
    if ground_max is not None:
        config.ground_protect_local_z_max = float(ground_max)
    config.lidar_hz = float(values["lidar_hz"])
    config.ray_stride = int(values["ray_stride"])
    config.max_ray_endpoints = int(values["max_ray_endpoints"])
    config.min_visible_frames = int(values["min_visible_frames"])
    config.min_visible_time = float(values["min_visible_time"])
    config.static_min_hit_ratio = float(values["static_min_hit_ratio"])
    config.dynamic_max_hit_ratio = float(values["dynamic_max_hit_ratio"])
    config.dynamic_max_hit_time = float(values["dynamic_max_hit_time"])
    config.keep_unknown = values["unknown_policy"] == "keep"
    config.validate()
    return module.Engine(config)
