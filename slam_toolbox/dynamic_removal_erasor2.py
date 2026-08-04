"""ERASOR2 dynamic object removal workflow."""

import glob
import os
import shutil
from pathlib import Path

import numpy as np
import questionary
import yaml

from .dynamic_removal_common import (
    _ERASOR2_BIN_DIR,
    _ERASOR2_HARD_MAX_POINTS,
    _ERASOR2_IMAGE,
    _ERASOR2_MEMORY_LIMIT,
    _ERASOR2_MEMORY_SWAP_LIMIT,
    _ERASOR2_MEMORY_SWAPPINESS,
    _ERASOR2_SCRIPTS_DIR,
    _ERASOR2_TARGET_MAX_POINTS,
    _add_docker_cidfile,
    _docker_cidfile,
    _run_measured_command,
    _timestamped_output_dir,
    console,
)
from .dynamic_removal_docker import _ensure_or_pull_image
from .dynamic_removal_kitti import convert_bag_to_kitti
from .dynamic_removal_results import (
    _keep_only_standard_run_pcds,
    _prompt_generate_dynamic_gif,
    _publish_run_result,
    _restore_intensity_from_reference,
    _restore_or_keep_intensity,
    _write_accumulated_kitti_map,
    _write_removed_difference,
)
from .extractor import _read_pcd, _write_pcd

def _kitti_point_count(path):
    size = os.path.getsize(path)
    bytes_per_point = 4 * np.dtype(np.float32).itemsize
    if size % bytes_per_point != 0:
        raise ValueError(f"KITTI 点云文件大小不是 16 字节的整数倍: {path}")
    return size // bytes_per_point


def _voxel_limit_xyzi(points, target_points):
    """Deterministically voxel-downsample xyzi until it fits the target."""
    if len(points) <= target_points:
        return points, 0.0

    voxel_size = 0.02
    best = points
    while voxel_size <= 5.0:
        coords = np.floor(
            points[:, :3].astype(np.float64) / voxel_size
        ).astype(np.int64)
        _, indices = np.unique(coords, axis=0, return_index=True)
        indices.sort()
        candidate = points[indices]
        best = candidate
        if len(candidate) <= target_points:
            return candidate.astype(np.float32, copy=False), voxel_size
        voxel_size *= 1.5

    # Extremely dense or pathological scans still need to stay readable.  The
    # deterministic evenly-spaced fallback preserves coverage better than
    # truncating the file head.
    indices = np.linspace(0, len(best) - 1, target_points, dtype=np.int64)
    return best[indices].astype(np.float32, copy=False), voxel_size


def _prepare_erasor2_limited_dataset(kitti_root, map_path):
    """Return an ERASOR2-safe dataset without changing the shared KITTI data.

    Normal per-scan bags are reused directly.  If any scan reaches the bundled
    loader's 500k-point ceiling, a method-specific dataset is generated: safe
    scans are hard-linked, oversized scans are voxel-downsampled, and labels
    are regenerated with exactly matching point counts.
    """
    source_root = Path(kitti_root)
    source_seq = source_root / "dataset" / "sequences" / "00"
    scan_paths = sorted((source_seq / "velodyne").glob("*.bin"))
    if not scan_paths:
        raise FileNotFoundError(f"ERASOR2 输入目录中没有 KITTI 点云: {source_seq}")

    counts = [(path, _kitti_point_count(path)) for path in scan_paths]
    oversized = [item for item in counts if item[1] >= _ERASOR2_HARD_MAX_POINTS]
    max_count = max(count for _, count in counts)
    if not oversized:
        print(
            f"ERASOR2 点数预检通过: 最大单帧 {max_count:,} 点 "
            f"(< {_ERASOR2_HARD_MAX_POINTS:,})"
        )
        return str(source_root)

    limited_root = Path(map_path) / "erasor2_dataset_limited"
    limited_seq = limited_root / "dataset" / "sequences" / "00"
    limited_velodyne = limited_seq / "velodyne"
    limited_labels = limited_seq / "labels"
    if limited_root.exists():
        shutil.rmtree(limited_root)
    limited_velodyne.mkdir(parents=True)
    limited_labels.mkdir(parents=True)

    reports = []
    for source_path, before_count in counts:
        target_path = limited_velodyne / source_path.name
        voxel_size = 0.0
        if before_count >= _ERASOR2_HARD_MAX_POINTS:
            scan = np.fromfile(source_path, dtype=np.float32)
            if scan.size % 4 != 0:
                raise ValueError(f"KITTI 点云不是 xyzi 格式: {source_path}")
            scan = scan.reshape(-1, 4)
            limited, voxel_size = _voxel_limit_xyzi(
                scan, _ERASOR2_TARGET_MAX_POINTS
            )
            limited.tofile(target_path)
            after_count = len(limited)
        else:
            try:
                os.link(source_path, target_path)
            except OSError:
                shutil.copy2(source_path, target_path)
            after_count = before_count

        np.zeros(after_count, dtype=np.uint32).tofile(
            limited_labels / f"{source_path.stem}.label"
        )
        reports.append(
            {
                "frame": source_path.stem,
                "before_points": before_count,
                "after_points": after_count,
                "voxel_size_m": voxel_size,
            }
        )

    for source_path in source_seq.iterdir():
        if source_path.is_file():
            shutil.copy2(source_path, limited_seq / source_path.name)

    (limited_seq / "limit_report.yaml").write_text(
        yaml.safe_dump(
            {
                "source_dataset": str(source_root.resolve()),
                "hard_max_points": _ERASOR2_HARD_MAX_POINTS,
                "target_max_points": _ERASOR2_TARGET_MAX_POINTS,
                "input_frames": len(counts),
                "oversized_frames": len(oversized),
                "max_input_points": max_count,
                "frames": reports,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    console.print(
        f"[yellow]ERASOR2 检测到 {len(oversized)} 帧达到或超过 "
        f"{_ERASOR2_HARD_MAX_POINTS:,} 点，已生成专用限点数据集: "
        f"{limited_root}[/yellow]"
    )
    return str(limited_root)


# ---------------------------------------------------------------------------
# ERASOR2 YAML 配置生成
# ---------------------------------------------------------------------------

def generate_erasor2_config(kitti_root, output_dir, frame_count, min_z, max_z):
    """生成 ERASOR2 的 YAML 配置文件。"""
    seq_dir = os.path.join(kitti_root, "dataset", "sequences")

    yaml_content = f"""\
start_frame: 0
end_frame: {frame_count - 1}
viz_interval: 100
is_large_scale: true
num_omp_cores: 4

dataloader:
    run_traj_clustering: false
    dataset_name: "SemanticKITTI"
    abs_data_dir: "{seq_dir}"
    cloud_dir: ""
    cloud_format: ""
    pose_path: ""
    sequence: "00"
    abs_save_dir: "{output_dir}"
    instance_seg_method: "hdbscan"

    accum_interval: 1
    voxel_size: 0.2
    map_voxel_size: 0.2

    expansion_range: 0

erasor2:
    grid_resolution: 1.0
    egocentric_grid_resolution: 0.6
    range_of_interest: 80.0
    min_z_voi: {min_z}
    max_z_voi: {max_z}
    min_z_diff_thr: 0.4
    scan_ratio_threshold: 0.2
    log_odds:
        increment_gain: 2.0
        increment: 0.15
    region_proposal_thr: 0.8
    kernel_size: 1

    ratio_num_pts: 0.95
    minimum_num_pts: 5

    moving_object_detection:
        negative_log_odds: -2.0
        obj_score_soft_thr: 4.6
        obj_score_hard_thr: 14.0
        hard_thr_radius: 10.0

    over_segmentation:
        minimum_area_thr: 56
        ratio_of_unknown_prior: 0.25

    volumetric_outlier_removal:
        window_size: 1
        use_adaptive_voxel_size: true
        vor_cand_score_thr: 4.6
        dist_thr_gain: 1.732

    viz_flag:
        set_scan_and_pose: false
        set_submap: false
        update: false
        detect: false
        over_seg: false

    save_map: true

stop_for_each_frame: false

extrinsic:
    robot_body_size: 2.7
    sensor_height: 1.73
    rotation: [ 1, 0, 0,
                0, 1, 0,
                0, 0, 1 ]
    translation: [ 0.0, 0.0, 0.0 ]

rerun:
    enabled: false
    spawn: false
    save_path: ""
"""

    config_path = os.path.join(output_dir, "erasor2_config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    Path(config_path).write_text(yaml_content)
    return config_path


# ---------------------------------------------------------------------------
# Docker 运行
# ---------------------------------------------------------------------------

def run_erasor2_docker(kitti_root, output_dir, config_path, frame_count):
    """通过 Docker 运行 ERASOR2（二进制和脚本均在镜像内）。"""

    image = _ensure_or_pull_image(_ERASOR2_IMAGE)

    docker_cmd = [
        "docker", "run", "--rm",
        f"--memory={_ERASOR2_MEMORY_LIMIT}",
        f"--memory-swap={_ERASOR2_MEMORY_SWAP_LIMIT}",
        f"--memory-swappiness={_ERASOR2_MEMORY_SWAPPINESS}",
        "--cpus=4",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{kitti_root}:{kitti_root}",
        "-v", f"{output_dir}:{output_dir}",
        "-w", _ERASOR2_SCRIPTS_DIR,
        image,
        "bash", "-lc",
        "set -euo pipefail; "
        f"python3 {_ERASOR2_SCRIPTS_DIR}/kitti_clustering.py "
        f"  --kitti_dir {kitti_root} "
        f"  --seq 00 "
        f"  --init_stamp 0 "
        f"  --end_stamp {frame_count - 1} "
        f"  --save-instance-labels "
        f"  --save-ground-labels; "
        f"{_ERASOR2_BIN_DIR}/mapgen {config_path}; "
        f"{_ERASOR2_BIN_DIR}/run_erasor2 {config_path}",
    ]

    print("正在 Docker 容器中运行 ERASOR2（可能需要数分钟）...")
    print(f"输出目录: {output_dir}")

    cidfile = _docker_cidfile(output_dir, "erasor2")
    docker_cmd = _add_docker_cidfile(docker_cmd, cidfile)
    returncode = _run_measured_command(
        docker_cmd,
        output_dir,
        "erasor2",
        docker_cidfile=cidfile,
    )
    if returncode != 0:
        console.print(f"[yellow]Docker 返回非零退出码: {returncode}，请检查上方日志[/yellow]")

    return returncode


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def start_erasor2(map_path):
    """ERASOR2 动态障碍物去除主流程。"""
    config_path = os.path.join(map_path, "config.yaml")
    try:
        from .config import DEFAULT_CONFIG, load_config
        config = load_config(config_path) if os.path.exists(config_path) else DEFAULT_CONFIG
    except Exception as e:
        console.print(f"[red]读取 config.yaml 失败: {e}[/red]")
        return

    bag_dir = os.path.join(map_path, "bag")
    if not os.path.isdir(bag_dir):
        print(f"bag 目录 {bag_dir} 不存在。ERASOR2 需要从原始点云话题逐帧转换。")
        return

    print(
        "ERASOR2 将使用 bag 的逐扫描点云，并使用 Interactive SLAM 的 corrected "
        "轨迹生成 KITTI 数据集（不直接使用 frame/ 聚合 PCD）。\n"
    )

    # 用户配置 Z 范围
    min_z_str = questionary.text(
        "请输入 ERASOR2 高度范围下限 Z_min (米):",
        default="-4.5"
    ).ask()
    max_z_str = questionary.text(
        "请输入 ERASOR2 高度范围上限 Z_max (米):",
        default="1.5"
    ).ask()
    try:
        min_z = float(min_z_str)
    except ValueError:
        min_z = -4.5
    try:
        max_z = float(max_z_str)
    except ValueError:
        max_z = 1.5

    # Step 1: bag 逐扫描点云 + corrected 插值位姿 → KITTI
    print()
    try:
        kitti_root, frame_count = convert_bag_to_kitti(map_path, config)
    except Exception as e:
        console.print(f"[red]KITTI 转换失败: {e}[/red]")
        return

    try:
        kitti_root = _prepare_erasor2_limited_dataset(kitti_root, map_path)
    except Exception as e:
        console.print(f"[red]ERASOR2 点数预检失败: {e}[/red]")
        return

    # Step 2: 生成配置
    output_dir = _timestamped_output_dir(map_path, "erasor2")
    config_path = generate_erasor2_config(kitti_root, output_dir, frame_count, min_z, max_z)
    print(f"配置文件已生成: {config_path}")

    # Step 3: 运行 ERASOR2
    print()
    try:
        returncode = run_erasor2_docker(
            kitti_root,
            output_dir,
            config_path,
            frame_count,
        )
    except RuntimeError as e:
        console.print(f"[red]错误: {e}[/red]")
        _keep_only_standard_run_pcds(output_dir)
        return

    # Step 4: 整理运行结果
    import glob

    before_candidates = sorted(glob.glob(os.path.join(output_dir, "*_original.pcd")))
    after_candidates = sorted(glob.glob(os.path.join(output_dir, "*_estimated.pcd")))

    run_before = os.path.join(output_dir, "before.pcd")
    run_static = os.path.join(output_dir, "static.pcd")
    intensity_reference_path = None
    temp_paths = []

    if returncode != 0:
        if before_candidates:
            _publish_run_result(
                before_candidates[0],
                output_dir,
                "before.pcd",
                label="ERASOR2 失败前生成的原始地图",
            )
        _keep_only_standard_run_pcds(output_dir)
        console.print(
            "[yellow]ERASOR2 运行失败，已清理中间 PCD；"
            "不会保留非标准文件名。[/yellow]"
        )
        print(f"  输出目录: {output_dir}/")
        return

    if not before_candidates:
        console.print(
            "\n[yellow]ERASOR2 未找到 *_original.pcd，无法可靠生成 before/dynamic。"
            "请重新运行 ERASOR2。[/yellow]"
        )
        _keep_only_standard_run_pcds(output_dir)
        print(f"  输出目录: {output_dir}/")
        return

    if not after_candidates:
        console.print(f"\n[yellow]ERASOR2 仅生成了原始地图，未找到 estimated 结果。[/yellow]")
        run_before = _publish_run_result(
            before_candidates[0],
            output_dir,
            "before.pcd",
            label="原始地图",
        )
        _keep_only_standard_run_pcds(output_dir)
        print(f"  输出目录: {output_dir}/")
        return

    before_source = before_candidates[0]
    static_source = after_candidates[0]
    before_xyz, before_i_source = _read_pcd(before_source)

    try:
        intensity_reference_path = os.path.join(output_dir, "_intensity_reference_before.pcd")
        intensity_reference_path = _write_accumulated_kitti_map(kitti_root, intensity_reference_path)
        if intensity_reference_path:
            temp_paths.append(intensity_reference_path)
            ref_xyz, ref_i = _read_pcd(intensity_reference_path)
        else:
            ref_xyz, ref_i = None, None
    except Exception as exc:
        console.print(f"[yellow]强度参考生成失败，将使用原始强度占位: {exc}[/yellow]")
        ref_xyz, ref_i = None, None

    before_i = _restore_or_keep_intensity(before_xyz, before_i_source, ref_xyz, ref_i)
    _write_pcd(run_before, before_xyz, before_i)

    static_xyz, _ = _read_pcd(static_source)
    static_i = _restore_intensity_from_reference(static_xyz, before_xyz, before_i)
    _write_pcd(run_static, static_xyz, static_i)

    console.print(f"\n[bold green]ERASOR2 处理完成！[/bold green]")
    print(f"  原始地图（去除前）: {run_before}")
    print(f"  静态地图（去除后）: {run_static}")
    run_dynamic = os.path.join(output_dir, "dynamic.pcd")
    if _write_removed_difference(run_before, run_static, run_dynamic):
        print(f"  动态点云（被移除）: {run_dynamic}")
    _keep_only_standard_run_pcds(output_dir)
    for path in temp_paths:
        if path and os.path.exists(path):
            os.remove(path)
    print(f"  完整输出目录: {output_dir}/")
    _prompt_generate_dynamic_gif(
        map_path,
        output_dir,
        "erasor2",
        "static.pcd",
        kitti_root=kitti_root,
        before_reference=run_before,
        static_reference=run_static,
        dynamic_reference=run_dynamic if os.path.exists(run_dynamic) else None,
        z_min=min_z,
        z_max=max_z,
        trail=-1,
    )


