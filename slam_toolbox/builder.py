import os
import numpy as np
import open3d as o3d
import questionary
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn


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

    print("正在增量式哈希建图（逐帧体素去重）...")

    voxel_grid = {}  # (ix, iy, iz) → (x, y, z)，全局唯一哈希表

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("处理点云帧...", total=len(files))

        for file in files:
            pcd_path = os.path.join(frame_dir, file)
            odom_path = pcd_path.replace(".pcd", ".odom")

            # 载入 + 位姿
            pcd = o3d.io.read_point_cloud(pcd_path)

            if os.path.exists(odom_path):
                try:
                    pose = np.loadtxt(odom_path)
                    if pose.shape == (4, 4):
                        pcd.transform(pose)
                except Exception:
                    pass

            points = np.asarray(pcd.points)
            if len(points) == 0:
                progress.update(task, advance=1)
                continue

            # 帧内向量化去重（大幅减少后续哈希表插入量）
            voxel_indices = np.floor(points / voxel_size).astype(np.int64)
            _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
            unique_voxels = voxel_indices[unique_idx]
            unique_points = points[unique_idx]

            # 哈希表全局去重（仅遍历帧内去重后的点）
            for idx, pt in zip(unique_voxels, unique_points):
                key = (int(idx[0]), int(idx[1]), int(idx[2]))
                if key not in voxel_grid:
                    voxel_grid[key] = (float(pt[0]), float(pt[1]), float(pt[2]))

            progress.update(task, advance=1)

    if not voxel_grid:
        print("未加载到有效点云。")
        return

    print(f"哈希建图完成，共 {len(voxel_grid):,} 个体素，正在写入文件...")

    final_points = np.array(list(voxel_grid.values()), dtype=np.float64)
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(final_points)

    output_pcd_path = os.path.join(map_dir, "map.pcd")
    o3d.io.write_point_cloud(output_pcd_path, final_pcd)
    print(f"全局地图拼接完成 → {output_pcd_path}（共 {len(final_points):,} 点）")
