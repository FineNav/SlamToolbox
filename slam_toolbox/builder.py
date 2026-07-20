import os
import numpy as np
import open3d as o3d
import questionary
from rich.progress import track

def start_building(map_path):
    frame_dir = os.path.join(map_path, "frame")
    map_dir = os.path.join(map_path, "map")
    os.makedirs(map_dir, exist_ok=True)

    if not os.path.exists(frame_dir):
        print(f"帧目录 {frame_dir} 不存在。请先运行 Frame Extractor 功能。")
        return

    # 获取所有 pcd 文件
    files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".pcd")])
    if not files:
        print("未在帧目录中找到 .pcd 文件。")
        return

    # 配置体素下采样参数
    voxel_str = questionary.text("请输入体素下采样大小 (米):", default="0.05").ask()
    try:
        voxel_size = float(voxel_str)
    except ValueError:
        voxel_size = 0.05

    print("正在加载并拼合点云中...")
    combined_cloud = o3d.geometry.PointCloud()

    for file in track(files, description="正在拼接点云..."):
        pcd_path = os.path.join(frame_dir, file)
        odom_path = pcd_path.replace(".pcd", ".odom")
        
        # 1. 载入点云
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        # 2. 载入位姿，并转换点云
        if os.path.exists(odom_path):
            try:
                pose = np.loadtxt(odom_path)
                if pose.shape == (4, 4):
                    pcd.transform(pose)
            except Exception as e:
                print(f"无法解析里程计位姿: {odom_path}, 错误: {e}")
        
        combined_cloud += pcd

    # 3. 体素下采样
    print(f"正在进行体素下采样，分辨率: {voxel_size}m...")
    final_pcd = combined_cloud.voxel_down_sample(voxel_size)

    # 4. 输出到目标位置
    output_pcd_path = os.path.join(map_dir, "map.pcd")
    o3d.io.write_point_cloud(output_pcd_path, final_pcd)
    print(f"全局地图拼接完成，文件已保存至 {output_pcd_path}")
