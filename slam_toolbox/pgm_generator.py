import os
import yaml
import numpy as np
import open3d as o3d
import questionary

def start_generation(map_path):
    map_dir = os.path.join(map_path, "map")
    pcd_path = os.path.join(map_dir, "map.pcd")

    if not os.path.exists(pcd_path):
        print(f"未找到点云地图文件: {pcd_path}")
        return

    # 要求配置相关生成参数
    grid_res = float(questionary.text("GRID_RESOLUTION (米/像素):", default="0.05").ask())
    min_z = float(questionary.text("MIN_Z (避障高度下限，米):", default="0.03").ask())
    max_z = float(questionary.text("MAX_Z (避障高度上限，米):", default="0.5").ask())
    padding = int(questionary.text("PADDING_PIXELS (地图边缘留白像素):", default="10").ask())
    occupied_thresh = int(questionary.text("OCCUPIED_THRESH (占有栅格的点数阈值):", default="2").ask())

    print("开始生成 2D 栅格地图...")
    
    # 1. 加载点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # 2. 过滤高度区间内的点
    z_mask = (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    filtered_points = points[z_mask]

    if len(filtered_points) == 0:
        print("设定高度内没有检测到任何点云数据。")
        return

    # 3. 确定栅格范围
    x_min, y_min = np.min(filtered_points[:, :2], axis=0)
    x_max, y_max = np.max(filtered_points[:, :2], axis=0)

    # 4. 地图大小计算
    width = int(np.ceil((x_max - x_min) / grid_res)) + 2 * padding
    height = int(np.ceil((y_max - y_min) / grid_res)) + 2 * padding

    # 5. 原点计算 (在 ROS 坐标系下，原点通常是左下角的物理坐标)
    origin_x = x_min - padding * grid_res
    origin_y = y_min - padding * grid_res

    # 6. 初始化计数矩阵，254 代表 Free (白色)，205 代表 Unknown (灰色)，0 代表 Occupied (黑色)
    grid_counts = np.zeros((height, width), dtype=np.int32)

    # 映射点到像素格子中
    px = ((filtered_points[:, 0] - origin_x) / grid_res).astype(np.int32)
    py = ((filtered_points[:, 1] - origin_y) / grid_res).astype(np.int32)

    # 合法越界筛查
    valid_mask = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[valid_mask]
    py = py[valid_mask]

    # 点计数
    for x_idx, y_idx in zip(px, py):
        grid_counts[y_idx, x_idx] += 1

    # 7. 根据阈值生成 PGM 图像数组
    pgm_img = np.ones((height, width), dtype=np.uint8) * 205  # 默认 Unknown 205
    
    # 填充 Free 与 Occupied 区域 (简单逻辑：投影有数据点记为 Occupied，无数据记为 Free 254)
    # 本算法使用基础投影：若格子内点数超过阈值则判定为障碍，其余物理范围内区域置为白色
    pgm_img[:, :] = 254  # 基础设为白色
    pgm_img[grid_counts >= occupied_thresh] = 0  # 障碍物设为黑色

    # ROS 栅格地图原点方向通常朝上。PGM 是从上往下写，因此需要垂直翻转以匹配。
    pgm_img = np.flipud(pgm_img)

    # 8. 写入 PGM 文件 (P5 格式)
    pgm_path = os.path.join(map_dir, "map.pgm")
    with open(pgm_path, 'wb') as f:
        # PGM 头部
        f.write(f"P5\n{width} {height}\n255\n".encode())
        f.write(pgm_img.tobytes())

    # 9. 写入 YAML 配置文件
    yaml_path = os.path.join(map_dir, "map.yaml")
    yaml_content = {
        "image": "map.pgm",
        "resolution": grid_res,
        "origin": [float(origin_x), float(origin_y), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"2D 地图已生成:\n - PGM: {pgm_path}\n - YAML: {yaml_path}")
