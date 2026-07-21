import os
import numpy as np
import questionary
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from .extractor import _read_pcd, _write_pcd


BATCH_SIZE = 15  # 每批帧数，控制内存峰值


def _voxel_downsample(xyz, voxel_size, intensity=None):
    """体素下采样，对 xyz 和 intensity 做均值聚合。"""

    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)

    # 用 structured array 做去重，无坐标范围限制
    dtype = np.dtype([('i', np.int64), ('j', np.int64), ('k', np.int64)])
    structured = np.empty(len(xyz), dtype=dtype)
    structured['i'] = voxel_indices[:, 0]
    structured['j'] = voxel_indices[:, 1]
    structured['k'] = voxel_indices[:, 2]

    _, inverse, counts = np.unique(structured, return_inverse=True, return_counts=True)
    unique_count = counts.size

    # 平均 xyz
    sum_xyz = np.zeros((unique_count, 3), dtype=np.float64)
    np.add.at(sum_xyz[:, 0], inverse, xyz[:, 0].astype(np.float64))
    np.add.at(sum_xyz[:, 1], inverse, xyz[:, 1].astype(np.float64))
    np.add.at(sum_xyz[:, 2], inverse, xyz[:, 2].astype(np.float64))
    avg_xyz = (sum_xyz / counts[:, None]).astype(np.float32)

    if intensity is not None:
        sum_intensity = np.zeros(unique_count, dtype=np.float64)
        np.add.at(sum_intensity, inverse, intensity.astype(np.float64))
        avg_intensity = (sum_intensity / counts).astype(np.float32)
    else:
        avg_intensity = None

    return avg_xyz, avg_intensity


def start_building(map_path):
    frame_dir = os.path.join(map_path, "frame")
    map_dir = os.path.join(map_path, "map")
    os.makedirs(map_dir, exist_ok=True)

    if not os.path.exists(frame_dir):
        print(f"帧目录 {frame_dir} 不存在。请先运行 Frame Extractor 功能。")
        return

    files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".pcd")])
    if not files:
        print("未在帧目录中找到 .pcd 文件。")
        return

    voxel_str = questionary.text("请输入体素下采样大小 (米):", default="0.05").ask()
    try:
        voxel_size = float(voxel_str)
    except ValueError:
        voxel_size = 0.05

    # 检查是否有 intensity 数据
    sample_xyz, sample_i = _read_pcd(os.path.join(frame_dir, files[0]))
    has_intensity = sample_i is not None
    print(f"正在分批建图（每 {BATCH_SIZE} 帧体素下采样, "
          f"voxel={voxel_size}m, intensity={'✓' if has_intensity else '✗'}）")

    # 累积器（处理过下采样的中间结果）
    acc_xyz = None           # (M, 3)
    acc_intensity = None     # (M,) 或 None
    total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("处理点云帧...", total=len(files))

        batch_xyz_list = []
        batch_intensity_list = []

        for i, file in enumerate(files):
            pcd_path = os.path.join(frame_dir, file)
            odom_path = pcd_path.replace(".pcd", ".odom")

            xyz, intensity = _read_pcd(pcd_path)

            if xyz is None or len(xyz) == 0:
                progress.update(task, advance=1)
                continue

            # 应用 odom 位姿
            if os.path.exists(odom_path):
                try:
                    pose = np.loadtxt(odom_path)
                    if pose.shape == (4, 4):
                        n = len(xyz)
                        pts_h = np.ones((n, 4), dtype=np.float64)
                        pts_h[:, :3] = xyz
                        xyz = (pose @ pts_h.T).T[:, :3].astype(np.float32)
                except Exception:
                    pass

            batch_xyz_list.append(xyz)
            if intensity is not None:
                batch_intensity_list.append(intensity)

            # 批次满：下采样后合并到累积器
            if (i + 1) % BATCH_SIZE == 0:
                batch_xyz = np.vstack(batch_xyz_list)
                batch_i = (np.hstack(batch_intensity_list)
                           if batch_intensity_list else None)

                ds_xyz, ds_i = _voxel_downsample(batch_xyz, voxel_size, batch_i)

                if acc_xyz is None:
                    acc_xyz, acc_intensity = ds_xyz, ds_i
                else:
                    # 合并到累积器再下采样，消除批次间重叠
                    acc_xyz = np.vstack([acc_xyz, ds_xyz])
                    acc_intensity = (np.hstack([acc_intensity, ds_i])
                                     if acc_intensity is not None and ds_i is not None
                                     else None)
                    acc_xyz, acc_intensity = _voxel_downsample(
                        acc_xyz, voxel_size, acc_intensity)

                batch_xyz_list = []
                batch_intensity_list = []

                batch_num = (i + 1) // BATCH_SIZE
                progress.update(task, advance=BATCH_SIZE,
                                description=f"处理点云帧... (批次 {batch_num}/{total_batches})")

        # 处理剩余不足一个批次的帧
        remaining = len(files) % BATCH_SIZE
        if batch_xyz_list:
            batch_xyz = np.vstack(batch_xyz_list)
            batch_i = (np.hstack(batch_intensity_list)
                       if batch_intensity_list else None)
            ds_xyz, ds_i = _voxel_downsample(batch_xyz, voxel_size, batch_i)

            if acc_xyz is None:
                acc_xyz, acc_intensity = ds_xyz, ds_i
            else:
                acc_xyz = np.vstack([acc_xyz, ds_xyz])
                if acc_intensity is not None and ds_i is not None:
                    acc_intensity = np.hstack([acc_intensity, ds_i])
                acc_xyz, acc_intensity = _voxel_downsample(
                    acc_xyz, voxel_size, acc_intensity)

            progress.update(task, advance=remaining)

    # 最终全局下采样
    print("正在最终全局去重...")
    final_xyz, final_intensity = _voxel_downsample(acc_xyz, voxel_size, acc_intensity)

    output_pcd_path = os.path.join(map_dir, "map.pcd")
    _write_pcd(output_pcd_path, final_xyz, final_intensity)
    print(f"全局地图拼接完成 → {output_pcd_path}（共 {len(final_xyz):,} 点）")
