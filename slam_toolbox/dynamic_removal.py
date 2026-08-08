"""Dynamic object removal public API.

The implementation is split by responsibility across:
- dynamic_removal_common.py: constants, resource monitoring, measured commands
- dynamic_removal_kitti.py: frame/bag to KITTI conversion
- dynamic_removal_erasor2.py: ERASOR2 workflow
- dynamic_removal_removert.py: Removert workflow
- dynamic_removal_local_voxel.py: Local Hash and Raycast voxel workflows
- dynamic_removal_results.py: result publishing, PCD differences, GIF helpers
"""

from .dynamic_removal_common import (
    KITTI_CAM2LIDAR,
    KITTI_CAM2LIDAR_INV,
    TF_ORIGIN,
    TF_ORIGIN_INV,
    _ProcessResourceMonitor,
    _add_docker_cidfile,
    _docker_cidfile,
    _mat3x4_line,
    _parse_resource_pair,
    _parse_resource_size,
    _run_measured_command,
    _timestamped_output_dir,
)
from .dynamic_removal_kitti import (
    convert_bag_to_kitti,
    convert_frames_to_kitti,
    _apply_interactive_slam_correction,
    _collect_bag_metadata,
    _find_bag_storage,
    _load_interactive_slam_correction,
    _lookup_or_identity,
    _message_time_sec,
    _reconstruct_legacy_frame_timestamps,
)
from .dynamic_removal_erasor2 import (
    generate_erasor2_config,
    run_erasor2_docker,
    start_erasor2,
    _kitti_point_count,
    _prepare_erasor2_limited_dataset,
    _voxel_limit_xyzi,
)
from .dynamic_removal_dataset import (
    _ensure_kitti_dataset,
    _ensure_local_kitti_dataset,
    _has_current_bag_local_transform,
    _load_map_config,
    _prepare_z_limited_kitti_dataset,
    _require_sensor_trajectory,
)
from .dynamic_removal_local_voxel import (
    start_local_hash_voxel,
    start_raycast_voxel,
    _build_algorithm_cli_args,
    _configure_algorithm_parameters,
    _format_parameter_value,
    _parameter_relationship_errors,
    _run_python_script,
    _show_algorithm_parameters,
)
from .dynamic_removal_removert import start_removert, _run_removert_with_progress, _strip_ansi
from .dynamic_removal_docker import _ensure_or_pull_image
from .dynamic_removal_results import (
    _keep_only_standard_run_pcds,
    _keys_in_sorted_unique,
    _prepare_dynamic_frames_from_kitti,
    _prompt_generate_dynamic_gif,
    _publish_run_result,
    _restore_intensity_from_reference,
    _restore_or_keep_intensity,
    _voxel_keys,
    _voxelize_pcd_file,
    _write_accumulated_kitti_map,
    _write_removed_difference,
    _write_voxel_difference,
)

__all__ = [name for name in globals() if not name.startswith("__")]
