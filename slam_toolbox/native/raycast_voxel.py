from __future__ import annotations

from typing import Any, Mapping

from .backend import native_module


def create_engine(values: Mapping[str, Any]):
    module = native_module().raycast_voxel
    config = module.Config()
    config.voxel_size = float(values["voxel_size"])
    config.max_range = float(values["max_range"])
    config.body_radius = float(values["body_radius"])
    config.local_z_min = float(values["local_z_min"])
    config.local_z_max = float(values["local_z_max"])
    ground_max = values.get("ground_protect_local_z_max")
    config.ground_protection_enabled = ground_max is not None
    if ground_max is not None:
        config.ground_protect_local_z_max = float(ground_max)
    config.ray_point_stride = int(values["ray_point_stride"])
    config.ray_step_factor = float(values["ray_step_factor"])
    config.endpoint_margin = float(values["endpoint_margin"])
    config.hit_log_odds = float(values["hit_log_odds"])
    config.miss_log_odds = float(values["miss_log_odds"])
    config.occupied_threshold = float(values["occupied_threshold"])
    config.free_threshold = float(values["free_threshold"])
    config.keep_unknown = values["unknown_policy"] == "keep"
    config.track_full_free_space = bool(values.get("export_octomap", False))
    config.validate()
    return module.Engine(config)
