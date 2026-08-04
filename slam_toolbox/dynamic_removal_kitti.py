"""KITTI dataset conversion for dynamic-removal methods."""

import os
import shutil
from pathlib import Path

import numpy as np
import questionary
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from .dynamic_removal_common import (
    KITTI_CAM2LIDAR_INV,
    TF_ORIGIN_INV,
    console,
    _mat3x4_line,
)
from .extractor import (
    _read_pcd,
    _transform_to_matrix,
    lookup_transform,
    parse_pc2_msg,
    rosbag2_py,
    deserialize_message,
    get_message,
)

# ---------------------------------------------------------------------------
# 帧 → KITTI 格式转换
# ---------------------------------------------------------------------------

def convert_frames_to_kitti(map_path):
    """将 frame/ 中的 PCD + .odom 转为 KITTI 格式，输出到 map_path/erasor2_dataset/。

    Returns:
        (kitti_root, frame_count) — kitti_root 是 dataset 根目录路径
    """
    frame_dir = os.path.join(map_path, "frame")
    kitti_root = os.path.join(map_path, "erasor2_dataset")
    seq_dir = os.path.join(kitti_root, "dataset", "sequences", "00")
    velodyne_dir = os.path.join(seq_dir, "velodyne")
    labels_dir = os.path.join(seq_dir, "labels")

    os.makedirs(velodyne_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".pcd")])
    if not files:
        raise FileNotFoundError(f"frame/ 中没有 .pcd 文件: {frame_dir}")

    true_pose_lines = []
    compensated_pose_lines = []
    time_lines = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("转换为 KITTI 格式...", total=len(files))

        for i, file in enumerate(files):
            stem = f"{i:06d}"
            pcd_path = os.path.join(frame_dir, file)
            odom_path = pcd_path.replace(".pcd", ".odom")

            # 读取点云
            xyz, intensity = _read_pcd(pcd_path)

            # 写入 .bin (float32 x y z intensity)
            bin_data = np.column_stack([xyz, intensity]).astype(np.float32) if intensity is not None else np.column_stack([xyz, np.ones(len(xyz), dtype=np.float32)]).astype(np.float32)
            bin_data.tofile(os.path.join(velodyne_dir, f"{stem}.bin"))

            # 写入 .label (全零)
            np.zeros(len(xyz), dtype=np.uint32).tofile(os.path.join(labels_dir, f"{stem}.label"))

            # 读写姿态
            if os.path.exists(odom_path):
                try:
                    T_odom_base = np.loadtxt(odom_path, dtype=np.float64)
                    if T_odom_base.shape != (4, 4):
                        T_odom_base = np.eye(4, dtype=np.float64)
                except Exception:
                    T_odom_base = np.eye(4, dtype=np.float64)
            else:
                T_odom_base = np.eye(4, dtype=np.float64)

            compensated = TF_ORIGIN_INV @ T_odom_base @ KITTI_CAM2LIDAR_INV
            compensated_pose_lines.append(_mat3x4_line(compensated))
            true_pose_lines.append(_mat3x4_line(T_odom_base))
            time_lines.append(f"{i * 0.1:.9f}")  # 用帧序号估算时间戳

            progress.update(task, advance=1)

    # 写入文本文件
    (Path(seq_dir) / "poses_suma_optim.txt").write_text("\n".join(compensated_pose_lines) + "\n")
    (Path(seq_dir) / "poses_odom_base.txt").write_text("\n".join(true_pose_lines) + "\n")
    (Path(seq_dir) / "times.txt").write_text("\n".join(time_lines) + "\n")
    (Path(seq_dir) / "conversion_notes.txt").write_text(
        f"converted from: {frame_dir}\n"
        f"frames_written: {len(files)}\n"
        "cloud_frame: base_link (extracted frames)\n"
        "poses_suma_optim.txt is compensated for ERASOR2 SemanticKITTILoader.\n"
        "poses_odom_base.txt contains the true odom -> base_link matrices.\n"
        "labels/*.label are zero placeholders for size compatibility.\n"
    )

    print(f"KITTI 格式转换完成: {len(files)} 帧 → {seq_dir}")
    return kitti_root, len(files)


def _find_bag_storage(map_path):
    bag_dir = os.path.join(map_path, "bag")
    for root, _, files in os.walk(bag_dir):
        for name in files:
            if name.endswith(".db3"):
                return bag_dir, "sqlite3", os.path.join(root, name)
            if name.endswith(".mcap"):
                return bag_dir, "mcap", os.path.join(root, name)
    raise FileNotFoundError(f"未在 {bag_dir} 下找到 .db3 或 .mcap")


def _collect_bag_metadata(bag_dir, storage_id, pointcloud_topic, config):
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    if pointcloud_topic not in type_map:
        raise ValueError(f"bag 中没有点云话题: {pointcloud_topic}")

    tf_type = next((typ for name, typ in type_map.items() if name in ("/tf", "/tf_static")), None)
    tf_msg_cls = get_message(tf_type) if tf_type else None
    dynamic_tf = {}
    static_tf = {}
    total_cloud_msgs = 0

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == pointcloud_topic:
            total_cloud_msgs += 1
        elif tf_msg_cls and topic in ("/tf", "/tf_static"):
            tf_msg = deserialize_message(data, tf_msg_cls)
            for transform in tf_msg.transforms:
                parent = transform.header.frame_id
                child = transform.child_frame_id
                sec = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
                matrix = _transform_to_matrix(transform)
                key = (parent, child)
                if topic == "/tf_static":
                    static_tf[key] = (sec, matrix)
                else:
                    dynamic_tf.setdefault(key, []).append((sec, matrix))

    from .config import build_fixed_transforms

    for key, value in build_fixed_transforms(config).items():
        if key not in static_tf:
            static_tf[key] = value

    for key in dynamic_tf:
        dynamic_tf[key].sort(key=lambda x: x[0])

    if total_cloud_msgs == 0:
        raise ValueError(f"bag 中没有点云消息: {pointcloud_topic}")

    return storage_options, converter_options, type_map, {"dynamic": dynamic_tf, "static": static_tf}, total_cloud_msgs


def _lookup_or_identity(tf_buffer, parent, child, timestamp, warn_set):
    if parent == child:
        return np.eye(4, dtype=np.float64)

    transform = lookup_transform(tf_buffer, parent, child, timestamp)
    if transform is not None:
        return transform

    tag = (parent, child)
    if tag not in warn_set:
        warn_set.add(tag)
        console.print(f"[yellow]警告: 缺少 TF {parent} -> {child}，使用单位阵代替。[/yellow]")
    return np.eye(4, dtype=np.float64)


def _message_time_sec(msg, bag_timestamp_ns=None):
    sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if sec > 0.0:
        return sec, "header"
    if bag_timestamp_ns is not None:
        return bag_timestamp_ns * 1e-9, "bag"
    return sec, "header_zero"


def _reconstruct_legacy_frame_timestamps(map_path, config):
    """Rebuild frame reference times for datasets extracted before they were saved."""
    frame_dir = Path(map_path) / "frame"
    frame_ids = sorted(
        int(path.stem) for path in frame_dir.glob("*.odom") if path.stem.isdigit()
    )
    if not frame_ids:
        raise RuntimeError("frame/ 中没有 .odom，无法重建 frame 时间戳。")

    cfg = config["config"]
    pointcloud_topic = cfg["pointcloud_topic"]
    bag_dir, storage_id, _ = _find_bag_storage(map_path)
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )

    while True:
        interval_text = questionary.text(
            "旧版 frame 缺少时间戳，请输入当时 Frame Extractor 使用的累计间隔（秒）:",
            default="1.0",
        ).ask()
        if interval_text is None:
            raise RuntimeError("已取消旧版 frame 时间戳重建。")
        try:
            interval = float(interval_text)
            if interval <= 0:
                raise ValueError
        except ValueError:
            console.print("[yellow]累计间隔必须是大于 0 的数字。[/yellow]")
            continue

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
        if pointcloud_topic not in type_map:
            raise RuntimeError(f"bag 中没有点云话题: {pointcloud_topic}")
        cloud_msg_type = get_message(type_map[pointcloud_topic])

        reference_times = []
        window_start = None
        while reader.has_next():
            topic, data, bag_timestamp_ns = reader.read_next()
            if topic != pointcloud_topic:
                continue
            msg = deserialize_message(data, cloud_msg_type)
            sec, _ = _message_time_sec(msg, bag_timestamp_ns)
            if window_start is None:
                window_start = sec
            if sec - window_start >= interval:
                reference_times.append(window_start)
                window_start = None
        if window_start is not None:
            reference_times.append(window_start)

        if len(reference_times) != len(frame_ids):
            console.print(
                f"[yellow]按 {interval:g} 秒重建得到 {len(reference_times)} 帧，"
                f"但 frame/ 中有 {len(frame_ids)} 帧，请重新输入原累计间隔。[/yellow]"
            )
            continue

        timestamps_path = frame_dir / "timestamps.txt"
        with timestamps_path.open("w", encoding="utf-8") as f:
            f.write("# frame_id reference_time_sec\n")
            for frame_id, timestamp in zip(frame_ids, reference_times):
                f.write(f"{frame_id:06d} {timestamp:.9f}\n")
        console.print(f"[green]✓ 已重建 {len(frame_ids)} 个 frame 参考时间戳。[/green]")
        return


def _load_interactive_slam_correction(map_path, config=None, tf_buffer=None):
    """Load dense frame corrections as time-indexed SE(3) control points.

    ``frame`` contains the interpolated corrected poses.  Each is compared with
    the original bag TF at the frame reference time so the resulting delta also
    includes an optional planar constraint applied before Interactive SLAM.
    The accumulated frame point clouds themselves are not used here.
    """
    frame_dir = Path(map_path) / "frame"
    corrected_dir = Path(map_path) / "interactive_slam" / "corrected"
    timestamps_path = frame_dir / "timestamps.txt"

    if not corrected_dir.is_dir() or not any(corrected_dir.iterdir()):
        raise RuntimeError(
            "未找到 Interactive SLAM corrected 结果。请先完成位姿修正和插值回填。"
        )
    if not timestamps_path.exists():
        raise RuntimeError(
            "缺少 frame/timestamps.txt，无法将修正轨迹与 bag 扫描按时间对齐。"
            "请使用新版 Frame Extractor 重新提取，然后重新执行 Interactive SLAM。"
        )

    try:
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp
    except ImportError as exc:
        raise RuntimeError("插值修正轨迹需要 scipy。") from exc

    frame_times = {}
    with timestamps_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                frame_times[int(parts[0])] = float(parts[1])
            except ValueError:
                continue

    use_bag_tf = config is not None and tf_buffer is not None
    backup_dir = Path(map_path) / "frame_backup"
    if not use_bag_tf and not backup_dir.is_dir():
        raise RuntimeError("缺少原始 bag TF 或 frame_backup，无法计算位姿修正量。")

    fixed_frame = config["config"]["fixed_frame"] if use_bag_tf else None
    base_link_frame = config["config"]["base_link_frame"] if use_bag_tf else None
    warn_set = set()
    controls = []
    for frame_id, timestamp in frame_times.items():
        name = f"{frame_id:06d}.odom"
        corrected_path = frame_dir / name
        if not corrected_path.exists():
            continue
        try:
            corrected_pose = np.loadtxt(corrected_path, dtype=np.float64)
        except (OSError, ValueError):
            continue
        if corrected_pose.shape != (4, 4):
            continue
        if use_bag_tf:
            raw_pose = _lookup_or_identity(
                tf_buffer, fixed_frame, base_link_frame, timestamp, warn_set
            )
        else:
            raw_path = backup_dir / name
            if not raw_path.exists():
                continue
            try:
                raw_pose = np.loadtxt(raw_path, dtype=np.float64)
            except (OSError, ValueError):
                continue
            if raw_pose.shape != (4, 4):
                continue
        delta = corrected_pose @ np.linalg.inv(raw_pose)
        controls.append((timestamp, delta))

    controls.sort(key=lambda item: item[0])
    # Slerp requires strictly increasing control times.
    unique_controls = []
    for timestamp, delta in controls:
        if unique_controls and abs(timestamp - unique_controls[-1][0]) < 1e-9:
            unique_controls[-1] = (timestamp, delta)
        else:
            unique_controls.append((timestamp, delta))

    if len(unique_controls) < 2:
        raise RuntimeError(
            "可用的修正轨迹控制点不足 2 个，请检查 corrected frame 位姿和 timestamps.txt。"
        )

    times = np.asarray([item[0] for item in unique_controls], dtype=np.float64)
    deltas = [item[1] for item in unique_controls]
    translations = np.asarray([delta[:3, 3] for delta in deltas], dtype=np.float64)
    rotations = R.from_matrix(np.asarray([delta[:3, :3] for delta in deltas]))
    return {
        "times": times,
        "translations": translations,
        "rotations": rotations,
        "slerp": Slerp(times, rotations),
        "count": len(unique_controls),
    }


def _apply_interactive_slam_correction(raw_pose, timestamp, correction):
    """Apply the time-interpolated frame correction to one raw bag pose."""
    times = correction["times"]
    translations = correction["translations"]
    rotations = correction["rotations"]

    if timestamp <= times[0]:
        translation = translations[0]
        rotation = rotations[0]
    elif timestamp >= times[-1]:
        translation = translations[-1]
        rotation = rotations[-1]
    else:
        right = int(np.searchsorted(times, timestamp))
        left = right - 1
        alpha = (timestamp - times[left]) / (times[right] - times[left])
        translation = (
            (1.0 - alpha) * translations[left] + alpha * translations[right]
        )
        rotation = correction["slerp"]([timestamp])[0]

    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = rotation.as_matrix()
    delta[:3, 3] = translation
    return delta @ raw_pose


def convert_bag_to_kitti(map_path, config):
    """从 bag 逐扫描点云和 Interactive SLAM 修正轨迹生成 KITTI。

    写出的 velodyne/*.bin 必须是 base_link 局部帧；如果输入点云是 /cloud_registered
    这类 odom/global 点云，会先用原始 TF 还原到局部帧。输出轨迹则使用按时间插值后的
    Interactive SLAM 修正位姿，不能用修正位姿反变换原始 registered 点云。
    """
    if rosbag2_py is None:
        raise RuntimeError("无法导入 rosbag2_py。请在 ROS2 环境中运行 ERASOR2 转换。")

    cfg = config["config"]
    fixed_frame = cfg["fixed_frame"]
    base_link_frame = cfg["base_link_frame"]
    pointcloud_topic = cfg["pointcloud_topic"]

    bag_dir, storage_id, db_file = _find_bag_storage(map_path)
    print(f"从 bag 逐帧转换为 KITTI: {db_file}")
    print(
        f"  fixed_frame={fixed_frame}, base_link_frame={base_link_frame}, "
        f"pointcloud_topic={pointcloud_topic}"
    )
    storage_options, converter_options, type_map, tf_buffer, total_cloud_msgs = _collect_bag_metadata(
        bag_dir, storage_id, pointcloud_topic, config
    )
    timestamps_path = Path(map_path) / "frame" / "timestamps.txt"
    if not timestamps_path.exists():
        _reconstruct_legacy_frame_timestamps(map_path, config)
    correction = _load_interactive_slam_correction(map_path, config, tf_buffer)
    print(f"  pose_source=interactive_slam_corrected, 控制点={correction['count']}")

    kitti_root = os.path.join(map_path, "erasor2_dataset")
    seq_dir = os.path.join(kitti_root, "dataset", "sequences", "00")
    velodyne_dir = os.path.join(seq_dir, "velodyne")
    labels_dir = os.path.join(seq_dir, "labels")

    if os.path.isdir(seq_dir):
        shutil.rmtree(seq_dir)
    os.makedirs(velodyne_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    true_pose_lines = []
    compensated_pose_lines = []
    time_lines = []
    warn_set = set()
    frame_count = 0
    first_cloud_frame = None
    max_points = 0
    time_sources = set()

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    cloud_msg_type = get_message(type_map[pointcloud_topic])

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("从 bag 逐帧转换为 KITTI...", total=total_cloud_msgs)

        while reader.has_next():
            topic, data, bag_timestamp_ns = reader.read_next()
            if topic != pointcloud_topic:
                continue

            msg = deserialize_message(data, cloud_msg_type)
            sec, time_source = _message_time_sec(msg, bag_timestamp_ns)
            time_sources.add(time_source)
            cloud_frame = msg.header.frame_id if msg.header.frame_id else base_link_frame
            if first_cloud_frame is None:
                first_cloud_frame = cloud_frame

            xyz, intensity = parse_pc2_msg(msg)
            if len(xyz) == 0:
                progress.update(task, advance=1)
                continue
            if intensity is None:
                intensity = np.ones(len(xyz), dtype=np.float32)

            T_raw_odom_base = _lookup_or_identity(
                tf_buffer, fixed_frame, base_link_frame, sec, warn_set
            )

            T_cloud_to_fixed = _lookup_or_identity(
                tf_buffer, fixed_frame, cloud_frame, sec, warn_set
            )
            # The registered cloud was produced with the original trajectory,
            # so only the original pose may be used to recover local points.
            T_fixed_to_base = np.linalg.inv(T_raw_odom_base)
            T_cloud_to_base = T_fixed_to_base @ T_cloud_to_fixed
            if not np.allclose(T_cloud_to_base, np.eye(4)):
                pts_h = np.ones((len(xyz), 4), dtype=np.float64)
                pts_h[:, :3] = xyz
                xyz = (T_cloud_to_base @ pts_h.T).T[:, :3].astype(np.float32)

            stem = f"{frame_count:06d}"
            bin_data = np.column_stack([xyz, intensity]).astype(np.float32)
            bin_data.tofile(os.path.join(velodyne_dir, f"{stem}.bin"))
            np.zeros(len(xyz), dtype=np.uint32).tofile(os.path.join(labels_dir, f"{stem}.label"))

            T_corrected_odom_base = _apply_interactive_slam_correction(
                T_raw_odom_base, sec, correction
            )
            compensated = TF_ORIGIN_INV @ T_corrected_odom_base @ KITTI_CAM2LIDAR_INV
            compensated_pose_lines.append(_mat3x4_line(compensated))
            true_pose_lines.append(_mat3x4_line(T_corrected_odom_base))
            time_lines.append(f"{sec:.9f}")
            max_points = max(max_points, len(xyz))
            frame_count += 1
            progress.update(task, advance=1)

    if frame_count == 0:
        raise RuntimeError("没有成功转换任何点云帧。")

    (Path(seq_dir) / "poses_suma_optim.txt").write_text("\n".join(compensated_pose_lines) + "\n")
    (Path(seq_dir) / "poses_odom_base.txt").write_text("\n".join(true_pose_lines) + "\n")
    (Path(seq_dir) / "times.txt").write_text("\n".join(time_lines) + "\n")
    (Path(seq_dir) / "conversion_notes.txt").write_text(
        f"source_bag: {bag_dir}\n"
        f"cloud_topic: {pointcloud_topic}\n"
        f"tf_edge: {fixed_frame} -> {base_link_frame}\n"
        f"cloud_frame_written: {base_link_frame}\n"
        f"source_cloud_frame: {first_cloud_frame}\n"
        f"time_source: {','.join(sorted(time_sources))}\n"
        f"point_transform: {first_cloud_frame} -> {fixed_frame} -> {base_link_frame}\n"
        "pose_source: interactive_slam_corrected\n"
        f"correction_control_points: {correction['count']}\n"
        f"frames_written: {frame_count}\n"
        f"max_points_per_frame: {max_points}\n"
        "poses_suma_optim.txt is compensated for ERASOR2 SemanticKITTILoader.\n"
        "poses_odom_base.txt contains Interactive SLAM-corrected odom -> base_link matrices.\n"
        "labels/*.label are zero placeholders for size compatibility, not ground truth.\n"
    )

    print(f"KITTI 格式转换完成: {frame_count} 帧 → {seq_dir}")
    return kitti_root, frame_count


