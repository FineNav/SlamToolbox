"""Result publishing, PCD post-processing, and dynamic GIF helpers."""

import os
import shutil
from pathlib import Path

import numpy as np
import questionary
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from .dynamic_removal_common import console
from .extractor import _read_pcd, _write_pcd

def _publish_run_result(src, output_dir, run_name, map_path=None, map_name=None, label=None):
    """Copy one result into the timestamped run dir."""
    if not src or not os.path.exists(src):
        return None
    run_path = os.path.join(output_dir, run_name)
    if os.path.abspath(src) != os.path.abspath(run_path):
        shutil.copy2(src, run_path)
    if label:
        print(f"  {label}: {run_path}")
    return run_path


def _keep_only_standard_run_pcds(output_dir):
    """Keep only before/dynamic/static point-cloud results in the run root."""
    keep = {"before.pcd", "dynamic.pcd", "static.pcd"}
    for path in Path(output_dir).glob("*.pcd"):
        if path.name not in keep:
            path.unlink()


def _write_accumulated_kitti_map(kitti_root, output_path):
    """Accumulate KITTI local scans into a global binary PCD."""
    seq_dir = Path(kitti_root) / "dataset" / "sequences" / "00"
    scan_paths = sorted((seq_dir / "velodyne").glob("*.bin"))
    pose_path = seq_dir / "poses_odom_base.txt"
    if not scan_paths or not pose_path.exists():
        return None

    poses = []
    for line in pose_path.read_text().splitlines():
        values = [float(value) for value in line.split()]
        if len(values) == 12:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
        elif len(values) == 16:
            pose = np.asarray(values, dtype=np.float64).reshape(4, 4)
        else:
            raise ValueError(f"不支持的 pose 格式: {pose_path}")
        poses.append(pose)
    if len(poses) < len(scan_paths):
        raise ValueError("传感器位姿数量少于逐帧点云数量")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path = output_path.with_suffix(output_path.suffix + ".payload")
    point_count = 0

    with payload_path.open("wb") as payload:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("生成 before.pcd...", total=len(scan_paths))
            for frame_id, scan_path in enumerate(scan_paths):
                scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
                scan = scan[np.isfinite(scan).all(axis=1)]
                if scan.size:
                    pose = poses[frame_id]
                    world_xyz = scan[:, :3].astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
                    out = np.empty_like(scan, dtype=np.float32)
                    out[:, :3] = world_xyz.astype(np.float32)
                    out[:, 3] = scan[:, 3]
                    out.tofile(payload)
                    point_count += int(out.shape[0])
                progress.update(task, advance=1)

    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {point_count}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {point_count}
DATA binary
"""
    with output_path.open("wb") as out, payload_path.open("rb") as payload:
        out.write(header.encode("ascii"))
        shutil.copyfileobj(payload, out, length=8 * 1024 * 1024)
    payload_path.unlink(missing_ok=True)
    return str(output_path)


def _voxelize_pcd_file(src_path, output_path, voxel_size=0.2):
    """Voxelize an existing PCD file deterministically, preserving one intensity per voxel."""
    xyz, intensity = _read_pcd(src_path)
    if len(xyz) == 0:
        _write_pcd(output_path, np.empty((0, 3), dtype=np.float32), None)
        return str(output_path)

    coords = np.floor(xyz.astype(np.float64) / voxel_size).astype(np.int64)
    coords = np.ascontiguousarray(coords)
    key_dtype = np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))
    keys = coords.view(key_dtype).reshape(-1)
    _, indices = np.unique(keys, return_index=True)
    indices.sort()
    _write_pcd(
        output_path,
        xyz[indices].astype(np.float32, copy=False),
        intensity[indices].astype(np.float32, copy=False) if intensity is not None else None,
    )
    return str(output_path)


def _restore_intensity_from_reference(target_xyz, reference_xyz, reference_intensity):
    """Assign each target point the intensity of its nearest reference point."""
    if len(target_xyz) == 0:
        return np.empty((0,), dtype=np.float32)
    if reference_intensity is None or len(reference_intensity) == 0 or len(reference_xyz) == 0:
        return np.ones(len(target_xyz), dtype=np.float32)

    from scipy.spatial import cKDTree

    tree = cKDTree(reference_xyz.astype(np.float64, copy=False))
    _, indices = tree.query(target_xyz.astype(np.float64, copy=False), k=1, workers=-1)
    return np.asarray(reference_intensity, dtype=np.float32)[indices]


def _restore_or_keep_intensity(target_xyz, source_intensity=None, reference_xyz=None, reference_intensity=None):
    """Prefer nearest-neighbor intensity restoration; fall back to source intensity or ones."""
    if reference_xyz is not None and reference_intensity is not None and len(reference_xyz) > 0:
        return _restore_intensity_from_reference(target_xyz, reference_xyz, reference_intensity)
    if source_intensity is not None and len(source_intensity) == len(target_xyz):
        return np.asarray(source_intensity, dtype=np.float32)
    return np.ones(len(target_xyz), dtype=np.float32)


def _voxel_keys(xyz, resolution):
    coords = np.floor(xyz.astype(np.float64) / resolution).astype(np.int64)
    coords = np.ascontiguousarray(coords)
    return coords.view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))).reshape(-1)


def _keys_in_sorted_unique(keys, sorted_unique_keys, chunk_size=1_000_000):
    """Return mask for keys present in sorted_unique_keys without np.isin's peak memory."""
    mask = np.zeros(len(keys), dtype=bool)
    if len(keys) == 0 or len(sorted_unique_keys) == 0:
        return mask

    for start in range(0, len(keys), chunk_size):
        end = min(start + chunk_size, len(keys))
        chunk = keys[start:end]
        indices = np.searchsorted(sorted_unique_keys, chunk)
        valid = indices < len(sorted_unique_keys)
        chunk_mask = np.zeros(len(chunk), dtype=bool)
        if np.any(valid):
            chunk_mask[valid] = sorted_unique_keys[indices[valid]] == chunk[valid]
        mask[start:end] = chunk_mask
    return mask


def _write_voxel_difference(source_path, subtract_path, output_path, voxel_size=0.2, description="差集"):
    """Write points present in source_path but absent from subtract_path."""
    if not source_path or not subtract_path:
        return False
    if not os.path.exists(source_path) or not os.path.exists(subtract_path):
        return False

    console.print(f"[dim]正在生成{description} ...[/dim]")
    source_xyz, source_i = _read_pcd(source_path)
    subtract_xyz, _ = _read_pcd(subtract_path)
    console.print(
        f"[dim]  差集输入: source={len(source_xyz):,} 点, subtract={len(subtract_xyz):,} 点[/dim]"
    )
    if len(source_xyz) == 0:
        _write_pcd(
            output_path,
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
        return True
    if len(subtract_xyz) == 0:
        _write_pcd(output_path, source_xyz, source_i)
        return True
    source_keys = _voxel_keys(source_xyz, voxel_size)
    subtract_keys = np.unique(_voxel_keys(subtract_xyz, voxel_size))
    removed_mask = ~_keys_in_sorted_unique(source_keys, subtract_keys)
    removed_i = source_i[removed_mask] if source_i is not None else None
    _write_pcd(output_path, source_xyz[removed_mask], removed_i)
    console.print(f"[dim]  已生成{description}: {output_path} ({int(removed_mask.sum()):,} 点)[/dim]")
    return True


def _write_removed_difference(before_path, static_path, removed_path, voxel_size=0.2):
    return _write_voxel_difference(before_path, static_path, removed_path, voxel_size, "动态点云")


def _prepare_dynamic_frames_from_kitti(
    kitti_root,
    dynamic_frames_dir,
    dynamic_reference=None,
    before_reference=None,
    static_reference=None,
    voxel_size=0.2,
    z_min=None,
    z_max=None,
):
    """Reconstruct per-frame dynamic points for algorithms with aggregated outputs."""
    if dynamic_reference is not None and os.path.exists(dynamic_reference):
        reference_xyz, _ = _read_pcd(dynamic_reference)
        dynamic_keys = np.unique(_voxel_keys(reference_xyz, voxel_size))
    elif before_reference and static_reference and os.path.exists(before_reference) and os.path.exists(static_reference):
        before_xyz, _ = _read_pcd(before_reference)
        static_xyz, _ = _read_pcd(static_reference)
        before_keys = np.unique(_voxel_keys(before_xyz, voxel_size))
        static_keys = np.unique(_voxel_keys(static_xyz, voxel_size))
        dynamic_keys = before_keys[~_keys_in_sorted_unique(before_keys, static_keys)]
    else:
        raise FileNotFoundError("缺少可用于重建逐帧动态点云的算法输出")

    seq_dir = Path(kitti_root) / "dataset" / "sequences" / "00"
    scan_paths = sorted((seq_dir / "velodyne").glob("*.bin"))
    pose_path = seq_dir / "poses_odom_base.txt"
    if not scan_paths or not pose_path.exists():
        raise FileNotFoundError("KITTI 逐帧点云或 poses_odom_base.txt 不存在")

    poses = []
    for line in pose_path.read_text().splitlines():
        values = [float(value) for value in line.split()]
        if len(values) == 12:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
        elif len(values) == 16:
            pose = np.asarray(values, dtype=np.float64).reshape(4, 4)
        else:
            raise ValueError(f"不支持的 pose 格式: {pose_path}")
        poses.append(pose)
    if len(poses) < len(scan_paths):
        raise ValueError("传感器位姿数量少于逐帧点云数量")

    dynamic_frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, scan_path in enumerate(scan_paths):
        scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        if scan.size == 0:
            _write_pcd(dynamic_frames_dir / f"{frame_id:06d}.pcd", np.empty((0, 3), dtype=np.float32))
            continue

        pose = poses[frame_id]
        world_xyz = scan[:, :3].astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
        dynamic_mask = _keys_in_sorted_unique(_voxel_keys(world_xyz, voxel_size), dynamic_keys)
        if z_min is not None:
            dynamic_mask &= world_xyz[:, 2] >= z_min
        if z_max is not None:
            dynamic_mask &= world_xyz[:, 2] <= z_max
        _write_pcd(
            dynamic_frames_dir / f"{frame_id:06d}.pcd",
            world_xyz[dynamic_mask].astype(np.float32),
            scan[dynamic_mask, 3],
        )


def _prompt_generate_dynamic_gif(
    map_path,
    output_dir,
    method_name,
    static_name,
    kitti_root=None,
    dynamic_reference=None,
    before_reference=None,
    static_reference=None,
    z_min=None,
    z_max=None,
    trail=-1,
    target_duration=10.0,
):
    should_generate = questionary.confirm(
        "是否生成动态障碍物轨迹 GIF？",
        default=True,
    ).ask()
    if not should_generate:
        console.print("[dim]已跳过 GIF 生成，动态障碍物清除流程结束。[/dim]")
        return

    dynamic_frames_dir = Path(output_dir) / "dynamic_frames"
    if kitti_root is not None and not dynamic_frames_dir.is_dir():
        try:
            console.print("[dim]正在根据算法输出重建逐帧动态点云...[/dim]")
            _prepare_dynamic_frames_from_kitti(
                kitti_root=kitti_root,
                dynamic_frames_dir=dynamic_frames_dir,
                dynamic_reference=dynamic_reference,
                before_reference=before_reference,
                static_reference=static_reference,
                z_min=z_min,
                z_max=z_max,
            )
        except Exception as exc:
            console.print(f"[yellow]无法重建逐帧动态点云，跳过 GIF: {exc}[/yellow]")
            return
    if not dynamic_frames_dir.is_dir():
        console.print(f"[dim]未找到逐帧动态点云目录，跳过 GIF: {dynamic_frames_dir}[/dim]")
        return

    if static_reference and os.path.exists(static_reference):
        static_path = static_reference
    else:
        static_path = os.path.join(map_path, "map", static_name)
    if not os.path.exists(static_path):
        console.print(f"[dim]未找到静态地图，跳过 GIF: {static_path}[/dim]")
        return

    try:
        from .visualization import create_dynamic_overlay_gif
    except Exception as exc:
        console.print(f"[yellow]无法导入 GIF 生成模块，跳过: {exc}[/yellow]")
        return

    run_viz_dir = Path(output_dir) / "visualize"
    run_viz_dir.mkdir(parents=True, exist_ok=True)
    gif_path = run_viz_dir / f"{method_name}_dynamic_overlay.gif"

    try:
        create_dynamic_overlay_gif(
            static_path=static_path,
            dynamic_dir=dynamic_frames_dir,
            output_path=gif_path,
            duration=None,
            trail=trail,
            apply_odom=False,
            view_mode="3d",
            elev=35.264,
            azim=-45.0,
            rotate_degrees=0.0,
            z_min=z_min,
            z_max=z_max,
            target_duration=target_duration,
        )
    except Exception as exc:
        console.print(f"[yellow]GIF 生成失败，已跳过: {exc}[/yellow]")
        return

    map_viz_dir = Path(map_path) / "visualize"
    map_viz_dir.mkdir(parents=True, exist_ok=True)
    map_gif = map_viz_dir / gif_path.name
    shutil.copy2(gif_path, map_gif)
    print(f"  已复制 GIF: {map_gif}")
