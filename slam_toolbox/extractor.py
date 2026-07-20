import os
import struct
import questionary
import numpy as np
import open3d as o3d
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# 动态尝试引入 rosbag2 接口，如系统无 rosbag2_py 会友好提示
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    rosbag2_py = None


# ---------------------------------------------------------------------------
# TF 工具函数
# ---------------------------------------------------------------------------

def _quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数 → 3×3 旋转矩阵"""
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    R[0, 1] = 2.0 * (qx * qy - qz * qw)
    R[0, 2] = 2.0 * (qx * qz + qy * qw)
    R[1, 0] = 2.0 * (qx * qy + qz * qw)
    R[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    R[1, 2] = 2.0 * (qy * qz - qx * qw)
    R[2, 0] = 2.0 * (qx * qz - qy * qw)
    R[2, 1] = 2.0 * (qy * qz + qx * qw)
    R[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    return R


def _transform_to_matrix(transform_msg):
    """将 ROS TransformStamped / Transform 转为 4×4 齐次矩阵"""
    t = transform_msg.transform.translation
    r = transform_msg.transform.rotation
    R = _quaternion_to_rotation_matrix(r.x, r.y, r.z, r.w)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = (t.x, t.y, t.z)
    return T


def _invert_transform(T):
    """高效求 4×4 齐次矩阵的逆"""
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def _find_transform_at(entries, timestamp):
    """在已排序的 (t, matrix) 列表中二分查找 timestamp 时刻的最近变换"""
    if not entries:
        return None
    lo, hi = 0, len(entries) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if entries[mid][0] <= timestamp:
            best = entries[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def build_tf_buffer(storage_options, converter_options):
    """从 bag 中读取 /tf 与 /tf_static，构建离线 TF 缓存。

    Returns:
        {
            'dynamic': {(parent, child): [(timestamp_sec, 4×4_matrix), ...]},
            'static':  {(parent, child): (timestamp_sec, 4×4_matrix)},
        }
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    # 获取 TF 消息类型
    tf_type = None
    for name, typ in type_map.items():
        if name in ('/tf', '/tf_static'):
            tf_type = typ
            break

    if tf_type is None:
        return {'dynamic': {}, 'static': {}}

    tf_msg_cls = get_message(tf_type)

    dynamic = {}
    static = {}

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in ('/tf', '/tf_static'):
            tf_msg = deserialize_message(data, tf_msg_cls)
            for transform in tf_msg.transforms:
                parent = transform.header.frame_id
                child = transform.child_frame_id
                sec = (transform.header.stamp.sec +
                       transform.header.stamp.nanosec * 1e-9)
                matrix = _transform_to_matrix(transform)
                key = (parent, child)

                if topic == '/tf_static':
                    static[key] = (sec, matrix)
                else:
                    dynamic.setdefault(key, []).append((sec, matrix))

    # 每条动态链按时间排序
    for key in dynamic:
        dynamic[key].sort(key=lambda x: x[0])

    return {'dynamic': dynamic, 'static': static}


def lookup_transform(tf_buffer, parent, child, timestamp):
    """在 TF 缓存中查找从 child → parent 的 4×4 变换矩阵。找不到返回 None。"""
    key = (parent, child)

    # 先查动态 TF
    if key in tf_buffer['dynamic']:
        result = _find_transform_at(tf_buffer['dynamic'][key], timestamp)
        if result is not None:
            return result

    # 再查静态 TF
    if key in tf_buffer['static']:
        return tf_buffer['static'][key][1]

    return None


# ---------------------------------------------------------------------------
# 主提取逻辑
# ---------------------------------------------------------------------------

def start_extraction(map_path):
    if rosbag2_py is None:
        print("错误: 无法导入 rosbag2_py。请确保是在激活的 ROS2 终端中运行。")
        return

    bag_dir = os.path.join(map_path, "bag")
    frame_dir = os.path.join(map_path, "frame")
    os.makedirs(frame_dir, exist_ok=True)

    # 寻找 db3 或 mcap 文件
    db_file = ""
    for root, dirs, files in os.walk(bag_dir):
        for f in files:
            if f.endswith('.db3') or f.endswith('.mcap'):
                db_file = os.path.join(root, f)
                break

    if not db_file:
        print(f"未在 {bag_dir} 下找到有效的 .db3 或 .mcap 录制文件。")
        return

    # 获取用户配置的累计时长
    interval_str = questionary.text("请输入点云累计保存时长间隔 (秒):", default="1.0").ask()
    try:
        interval = float(interval_str)
    except ValueError:
        interval = 1.0

    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    # -------- 第一遍：统计 /cloud_registered 数量 + 收集 TF --------
    print(f"正在扫描 {db_file} …")

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    total_cloud_msgs = 0

    # 收集 TF
    tf_type = None
    for name, typ in type_map.items():
        if name in ('/tf', '/tf_static'):
            tf_type = typ
            break

    tf_msg_cls = None
    dynamic_tf = {}
    static_tf = {}
    if tf_type:
        tf_msg_cls = get_message(tf_type)

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/cloud_registered":
            total_cloud_msgs += 1
        elif tf_msg_cls and topic in ('/tf', '/tf_static'):
            tf_msg = deserialize_message(data, tf_msg_cls)
            for transform in tf_msg.transforms:
                parent = transform.header.frame_id
                child = transform.child_frame_id
                sec = (transform.header.stamp.sec +
                       transform.header.stamp.nanosec * 1e-9)
                matrix = _transform_to_matrix(transform)
                key = (parent, child)

                if topic == '/tf_static':
                    static_tf[key] = (sec, matrix)
                else:
                    dynamic_tf.setdefault(key, []).append((sec, matrix))

    # 排序动态 TF 时间线
    for key in dynamic_tf:
        dynamic_tf[key].sort(key=lambda x: x[0])

    tf_buffer = {'dynamic': dynamic_tf, 'static': static_tf}

    if total_cloud_msgs == 0:
        print("未在 bag 中找到 /cloud_registered 话题消息，退出提取。")
        return

    print(f"检测到 {total_cloud_msgs} 条 /cloud_registered 消息，开始累积提取...")

    # -------- 第二遍：按时间窗口累积点云 + TF 对齐 --------
    reader.open(storage_options, converter_options)

    frame_idx = 0
    window_start = -1.0
    accumulated_points = []       # list of (N, 3) numpy arrays
    reference_pose = None         # T_odom_base(t0)，写入 .odom
    cloud_msg_type = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("累积提取点云帧...", total=total_cloud_msgs)

        msg_idx = 0
        while reader.has_next():
            topic, data, t = reader.read_next()

            if topic == "/cloud_registered":
                if cloud_msg_type is None:
                    cloud_msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, cloud_msg_type)
                sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                cloud_frame = msg.header.frame_id if msg.header.frame_id else "base_link"

                # ---- 查询 TF ----
                T_odom_base = lookup_transform(tf_buffer, "odom", "base_link", sec)
                if T_odom_base is None:
                    T_odom_base = np.eye(4)

                # ---- 新窗口开启 ----
                if window_start < 0:
                    window_start = sec
                    reference_pose = T_odom_base.copy()

                T_odom_base_ref = lookup_transform(tf_buffer, "odom", "base_link", window_start)
                if T_odom_base_ref is None:
                    T_odom_base_ref = np.eye(4)

                # ---- cloud → base_link 的静态/动态变换 ----
                if cloud_frame != "base_link":
                    T_base_cloud = lookup_transform(tf_buffer, "base_link", cloud_frame, sec)
                    if T_base_cloud is None:
                        T_base_cloud = np.eye(4)
                else:
                    T_base_cloud = np.eye(4)

                # ---- 相对变换: cloud(t) → base_link(t0) ----
                # cloud(t) → base_link(t) → odom → base_link(t0)
                T_rel = _invert_transform(T_odom_base_ref) @ T_odom_base @ T_base_cloud

                # ---- 解析点云 ----
                points = parse_pc2_msg(msg)
                if len(points) > 0:
                    # 齐次变换
                    n = len(points)
                    points_h = np.ones((n, 4), dtype=np.float64)
                    points_h[:, :3] = points
                    transformed = (T_rel @ points_h.T).T[:, :3].astype(np.float32)
                    accumulated_points.append(transformed)

                # ---- 时间窗口到？----
                if sec - window_start >= interval:
                    _save_accumulated_frame(frame_dir, frame_idx,
                                            accumulated_points, reference_pose)
                    frame_idx += 1
                    accumulated_points = []
                    reference_pose = None
                    window_start = -1.0

                msg_idx += 1
                progress.update(task, completed=msg_idx)

    # 保存最后一个窗口
    if accumulated_points:
        _save_accumulated_frame(frame_dir, frame_idx,
                                accumulated_points, reference_pose)
        frame_idx += 1

    print(f"提取完成！共保存 {frame_idx} 帧累积点云到 {frame_dir}")


def _save_accumulated_frame(frame_dir, frame_idx, point_arrays, reference_pose):
    """将累积的点云数组拼接并保存为 PCD + odom"""
    if not point_arrays:
        return

    merged = np.vstack(point_arrays)

    pcd_filename = f"{frame_idx:06d}.pcd"
    pcd_path = os.path.join(frame_dir, pcd_filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)
    o3d.io.write_point_cloud(pcd_path, pcd)

    odom_filename = f"{frame_idx:06d}.odom"
    odom_path = os.path.join(frame_dir, odom_filename)
    if reference_pose is None:
        reference_pose = np.eye(4)
    np.savetxt(odom_path, reference_pose, fmt="%.6f")


def parse_pc2_msg(msg):
    """解析 PointCloud2 → (N, 3) float32 数组"""
    try:
        from sensor_msgs_py import point_cloud2
        pts = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        return np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
    except Exception:
        # 手动解析 fallback
        data_len = len(msg.data)
        points_count = data_len // msg.point_step
        points = []
        for i in range(points_count):
            offset = i * msg.point_step
            x, y, z = struct.unpack_from('fff', msg.data, offset)
            points.append([x, y, z])
        return np.array(points, dtype=np.float32)
