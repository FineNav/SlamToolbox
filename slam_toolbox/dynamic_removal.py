"""
ERASOR2 + Removert 动态障碍物去除模块
"""

import os
import sys
import subprocess
import tempfile
import textwrap
import shutil
import re
import json
import platform
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import questionary
import yaml
from questionary import Choice
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()

from .extractor import (
    _read_pcd,
    _write_pcd,
    _invert_transform,
    _transform_to_matrix,
    lookup_transform,
    parse_pc2_msg,
    rosbag2_py,
    deserialize_message,
    get_message,
)

# ---------------------------------------------------------------------------
# Docker 镜像（自包含，无需挂载主机文件）
# ---------------------------------------------------------------------------

_ERASOR2_IMAGE = "stevenmhy/slamtoolbox-erasor2:latest"
_REMOVERT_IMAGE = "stevenmhy/slamtoolbox-removert:latest"
_ERASOR2_MEMORY_LIMIT = "12g"
_ERASOR2_MEMORY_SWAP_LIMIT = "18g"
_ERASOR2_MEMORY_SWAPPINESS = "80"

# 容器内固定路径
_ERASOR2_BIN_DIR = "/opt/erasor2/bin"
_ERASOR2_SCRIPTS_DIR = "/opt/erasor2/scripts"
_REMOVERT_WS = "/opt/removert_ws"

# The bundled ERASOR2 SemanticKITTILoader uses a 2,000,000-float buffer.
# KITTI scans store xyzi (four float32 values), so 500,000 points is a hard
# loader limit.  Keep some headroom for future loader-side checks.
_ERASOR2_HARD_MAX_POINTS = 500_000
_ERASOR2_TARGET_MAX_POINTS = 450_000

_PKG_DIR = Path(__file__).resolve().parent  # slam_toolbox/

# ---------------------------------------------------------------------------
# ERASOR2 SemanticKITTILoader 补偿矩阵（来自上游 convert_ros2bag_to_erasor2_kitti.py）
# ---------------------------------------------------------------------------

TF_ORIGIN = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

KITTI_CAM2LIDAR = np.array(
    [
        [-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03],
        [-6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02],
        [9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

TF_ORIGIN_INV = np.linalg.inv(TF_ORIGIN)
KITTI_CAM2LIDAR_INV = np.linalg.inv(KITTI_CAM2LIDAR)


def _mat3x4_line(mat):
    """将 4×4 矩阵转为 12 个空格分隔的 float（ERASOR2 3×4 行主序格式）"""
    return " ".join(f"{v:.9f}" for v in mat[:3, :4].reshape(-1))


def _timestamped_output_dir(map_path, method_name):
    """Create a timestamped run directory for method outputs."""
    root = os.path.join(map_path, "runs", method_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, stamp)
    suffix = 1
    while os.path.exists(out):
        out = os.path.join(root, f"{stamp}_{suffix:02d}")
        suffix += 1
    os.makedirs(out, exist_ok=True)
    return out


_RESOURCE_SAMPLE_INTERVAL = 0.5
_SIZE_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000 ** 2,
    "gb": 1000 ** 3,
    "tb": 1000 ** 4,
    "kib": 1024,
    "mib": 1024 ** 2,
    "gib": 1024 ** 3,
    "tib": 1024 ** 4,
}


def _parse_resource_size(value):
    """Parse Docker size strings such as ``512MiB`` or ``1.2GB``."""
    if value is None:
        return 0
    text_value = str(value).strip()
    if not text_value or text_value == "--":
        return 0
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?", text_value)
    if not match:
        return 0
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    return int(number * _SIZE_UNITS.get(unit, 1))


def _parse_resource_pair(value):
    parts = str(value or "").split("/", 1)
    first = _parse_resource_size(parts[0]) if parts else 0
    second = _parse_resource_size(parts[1]) if len(parts) > 1 else 0
    return first, second


class _ProcessResourceMonitor:
    """Sample one process tree and, when present, its Docker container."""

    def __init__(self, process, output_dir, method_name, docker_cidfile=None):
        self.process = process
        self.output_dir = Path(output_dir)
        self.method_name = method_name
        self.docker_cidfile = Path(docker_cidfile) if docker_cidfile else None
        self.started_wall = time.time()
        self.started_monotonic = time.monotonic()
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.cpu_times = {}
        self.io_counters = {}
        self.peak_rss_bytes = 0
        self.container_id = None
        self.container_pid = None
        self.container_cpu_times = {}
        self.container_io_counters = {}
        self.container_process_peak_rss_bytes = 0
        self.container_peak_memory_bytes = 0
        self.container_peak_cpu_percent = 0.0
        self.container_cpu_core_seconds_estimate = 0.0
        self.container_block_read_bytes = 0
        self.container_block_write_bytes = 0
        self._last_container_sample = None

    def start(self):
        self.thread.start()
        return self

    def _sample_process_tree(self):
        try:
            import psutil

            root = psutil.Process(self.process.pid)
            processes = [root] + root.children(recursive=True)
        except Exception:
            return

        current_rss = 0
        for proc in processes:
            try:
                pid = proc.pid
                cpu = proc.cpu_times()
                previous = self.cpu_times.get(pid, (0.0, 0.0))
                self.cpu_times[pid] = (
                    max(previous[0], float(cpu.user)),
                    max(previous[1], float(cpu.system)),
                )
                current_rss += int(proc.memory_info().rss)
                io = proc.io_counters()
                previous_io = self.io_counters.get(pid, (0, 0))
                self.io_counters[pid] = (
                    max(previous_io[0], int(io.read_bytes)),
                    max(previous_io[1], int(io.write_bytes)),
                )
            except Exception:
                continue
        self.peak_rss_bytes = max(self.peak_rss_bytes, current_rss)

    def _read_container_id(self):
        if self.container_id or self.docker_cidfile is None:
            return self.container_id
        try:
            container_id = self.docker_cidfile.read_text().strip()
        except OSError:
            return None
        if container_id:
            self.container_id = container_id
        return self.container_id

    def _read_container_pid(self):
        if self.container_pid:
            return self.container_pid
        container_id = self._read_container_id()
        if not container_id:
            return None
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        try:
            pid = int(result.stdout.strip())
        except ValueError:
            return None
        if pid > 0:
            self.container_pid = pid
        return self.container_pid

    def _sample_container_process_tree(self):
        pid = self._read_container_pid()
        if not pid:
            return
        try:
            import psutil

            root = psutil.Process(pid)
            processes = [root] + root.children(recursive=True)
        except Exception:
            return

        current_rss = 0
        for proc in processes:
            try:
                proc_pid = proc.pid
                cpu = proc.cpu_times()
                previous = self.container_cpu_times.get(proc_pid, (0.0, 0.0))
                self.container_cpu_times[proc_pid] = (
                    max(previous[0], float(cpu.user)),
                    max(previous[1], float(cpu.system)),
                )
                current_rss += int(proc.memory_info().rss)
                io = proc.io_counters()
                previous_io = self.container_io_counters.get(proc_pid, (0, 0))
                self.container_io_counters[proc_pid] = (
                    max(previous_io[0], int(io.read_bytes)),
                    max(previous_io[1], int(io.write_bytes)),
                )
            except Exception:
                continue
        self.container_process_peak_rss_bytes = max(
            self.container_process_peak_rss_bytes,
            current_rss,
        )

    def _sample_container(self):
        container_id = self._read_container_id()
        if not container_id:
            return
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return
        try:
            stats = json.loads(result.stdout.splitlines()[-1])
        except (ValueError, TypeError):
            return

        memory_used, _ = _parse_resource_pair(stats.get("MemUsage"))
        self.container_peak_memory_bytes = max(self.container_peak_memory_bytes, memory_used)
        try:
            cpu_percent = float(str(stats.get("CPUPerc", "0")).strip().rstrip("%"))
        except ValueError:
            cpu_percent = 0.0
        self.container_peak_cpu_percent = max(self.container_peak_cpu_percent, cpu_percent)

        now = time.monotonic()
        if self._last_container_sample is not None:
            previous_time, previous_cpu_percent = self._last_container_sample
            self.container_cpu_core_seconds_estimate += (
                previous_cpu_percent / 100.0 * max(0.0, now - previous_time)
            )
        self._last_container_sample = (now, cpu_percent)

        block_read, block_write = _parse_resource_pair(stats.get("BlockIO"))
        self.container_block_read_bytes = max(self.container_block_read_bytes, block_read)
        self.container_block_write_bytes = max(self.container_block_write_bytes, block_write)

    def _sample_loop(self):
        while not self.stop_event.is_set():
            self._sample_process_tree()
            self._sample_container_process_tree()
            self._sample_container()
            self.stop_event.wait(_RESOURCE_SAMPLE_INTERVAL)
        self._sample_process_tree()
        self._sample_container_process_tree()
        self._sample_container()

    def finish(self, exit_code):
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        wall_time = max(0.0, time.monotonic() - self.started_monotonic)
        cpu_user = sum(value[0] for value in self.cpu_times.values())
        cpu_system = sum(value[1] for value in self.cpu_times.values())
        host_read = sum(value[0] for value in self.io_counters.values())
        host_write = sum(value[1] for value in self.io_counters.values())
        is_docker = self.docker_cidfile is not None
        container_cpu_user = sum(value[0] for value in self.container_cpu_times.values())
        container_cpu_system = sum(value[1] for value in self.container_cpu_times.values())
        container_cpu_total = container_cpu_user + container_cpu_system
        container_read = sum(value[0] for value in self.container_io_counters.values())
        container_write = sum(value[1] for value in self.container_io_counters.values())
        effective_cpu = container_cpu_total if is_docker else cpu_user + cpu_system
        effective_peak_memory = self.peak_rss_bytes
        if is_docker:
            effective_peak_memory = max(
                self.container_peak_memory_bytes,
                self.container_process_peak_rss_bytes,
            )
        report = {
            "method": self.method_name,
            "scope": "core_algorithm_process",
            "started_at": self.started_at,
            "finished_at": finished_at,
            "wall_time_seconds": round(wall_time, 3),
            "exit_code": int(exit_code) if exit_code is not None else None,
            "effective_cpu_seconds": round(effective_cpu, 3),
            "effective_peak_memory_bytes": int(effective_peak_memory),
            "host_process_tree": {
                "cpu_user_seconds": round(cpu_user, 3),
                "cpu_system_seconds": round(cpu_system, 3),
                "cpu_total_seconds": round(cpu_user + cpu_system, 3),
                "peak_rss_bytes": int(self.peak_rss_bytes),
                "disk_read_bytes": int(host_read),
                "disk_write_bytes": int(host_write),
            },
            "docker_container": None,
            "sampling_interval_seconds": _RESOURCE_SAMPLE_INTERVAL,
            "system": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
            },
        }
        if is_docker:
            report["docker_container"] = {
                "container_id": self.container_id,
                "host_pid": self.container_pid,
                "cpu_user_seconds": round(container_cpu_user, 3),
                "cpu_system_seconds": round(container_cpu_system, 3),
                "cpu_total_seconds": round(container_cpu_total, 3),
                "cpu_core_seconds_estimate": round(
                    self.container_cpu_core_seconds_estimate, 3
                ),
                "peak_cpu_percent": round(self.container_peak_cpu_percent, 3),
                "peak_memory_bytes": int(self.container_peak_memory_bytes),
                "process_tree_peak_rss_bytes": int(
                    self.container_process_peak_rss_bytes
                ),
                "process_tree_disk_read_bytes": int(container_read),
                "process_tree_disk_write_bytes": int(container_write),
                "block_read_bytes": int(self.container_block_read_bytes),
                "block_write_bytes": int(self.container_block_write_bytes),
            }

        report_path = self.output_dir / "resource_usage.yaml"
        report_path.write_text(
            yaml.safe_dump(report, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        if self.docker_cidfile is not None:
            self.docker_cidfile.unlink(missing_ok=True)
        console.print(
            f"[dim]资源统计: 用时 {wall_time:.1f}s, CPU {effective_cpu:.1f}s, "
            f"峰值内存 {effective_peak_memory / (1024 ** 2):.1f} MiB[/dim]"
        )
        print(f"  资源报告: {report_path}")
        return report


def _docker_cidfile(output_dir, method_name):
    return os.path.join(output_dir, f".{method_name}_container.cid")


def _add_docker_cidfile(cmd, cidfile):
    if len(cmd) >= 2 and cmd[0:2] == ["docker", "run"]:
        return cmd[:2] + ["--cidfile", cidfile] + cmd[2:]
    return cmd


def _run_measured_command(cmd, output_dir, method_name, docker_cidfile=None):
    if docker_cidfile:
        Path(docker_cidfile).unlink(missing_ok=True)
    process = subprocess.Popen(cmd)
    monitor = _ProcessResourceMonitor(
        process,
        output_dir,
        method_name,
        docker_cidfile=docker_cidfile,
    ).start()
    try:
        returncode = process.wait()
    finally:
        returncode = process.poll()
        monitor.finish(returncode)
    return returncode


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


# ---------------------------------------------------------------------------
# Removert 动态障碍物去除
# ---------------------------------------------------------------------------

def _load_map_config(map_path):
    config_path = os.path.join(map_path, "config.yaml")
    try:
        from .config import DEFAULT_CONFIG, load_config
        return load_config(config_path) if os.path.exists(config_path) else DEFAULT_CONFIG
    except Exception as e:
        raise RuntimeError(f"读取 config.yaml 失败: {e}") from e


def _has_current_bag_local_transform(seq_dir):
    notes_path = os.path.join(seq_dir, "conversion_notes.txt")
    if not os.path.exists(notes_path):
        return False
    notes = Path(notes_path).read_text(errors="replace")
    return (
        "source_bag:" in notes
        and "point_transform:" in notes
        and "cloud_frame_written: base_link" in notes
        and "time_source:" in notes
        and "pose_source: interactive_slam_corrected" in notes
    )


def _ensure_kitti_dataset(map_path):
    """确保 KITTI 使用逐扫描局部点云和 Interactive SLAM 修正轨迹。"""
    kitti_root = os.path.join(map_path, "erasor2_dataset")
    seq_dir = os.path.join(kitti_root, "dataset", "sequences", "00")
    velodyne_dir = os.path.join(seq_dir, "velodyne")

    if os.path.isdir(velodyne_dir) and os.listdir(velodyne_dir):
        bin_files = [f for f in os.listdir(velodyne_dir) if f.endswith(".bin")]
        if _has_current_bag_local_transform(seq_dir):
            frame_count = len(bin_files)
            print(f"复用已有 corrected bag-local KITTI 数据集: {velodyne_dir} ({frame_count} 帧)")
            return kitti_root, frame_count
        print("检测到未使用 Interactive SLAM 修正轨迹的 KITTI 数据集，将重新生成。")

    bag_dir = os.path.join(map_path, "bag")
    if not os.path.isdir(bag_dir):
        raise RuntimeError(f"bag 目录不存在，无法生成 KITTI 数据集: {bag_dir}")

    print("从 bag 逐扫描点云和 Interactive SLAM 修正轨迹生成 KITTI...")
    return convert_bag_to_kitti(map_path, _load_map_config(map_path))


def _prepare_z_limited_kitti_dataset(kitti_root, map_path, method_name, z_min, z_max):
    """Create a method-specific KITTI dataset with local-frame z filtering."""
    source_root = Path(kitti_root)
    source_seq = source_root / "dataset" / "sequences" / "00"
    source_velodyne = source_seq / "velodyne"
    scan_paths = sorted(source_velodyne.glob("*.bin"))
    if not scan_paths:
        raise FileNotFoundError(f"KITTI 输入目录中没有点云: {source_velodyne}")

    limited_root = Path(map_path) / f"{method_name}_dataset_z_limited"
    limited_seq = limited_root / "dataset" / "sequences" / "00"
    limited_velodyne = limited_seq / "velodyne"
    limited_labels = limited_seq / "labels"
    if limited_root.exists():
        shutil.rmtree(limited_root)
    limited_velodyne.mkdir(parents=True)
    limited_labels.mkdir(parents=True)

    reports = []
    total_before = 0
    total_after = 0
    for scan_path in scan_paths:
        scan = np.fromfile(scan_path, dtype=np.float32)
        if scan.size % 4 != 0:
            raise ValueError(f"KITTI 点云不是 xyzi 格式: {scan_path}")
        scan = scan.reshape(-1, 4)
        keep = np.ones(len(scan), dtype=bool)
        keep &= scan[:, 2] >= z_min
        keep &= scan[:, 2] <= z_max
        filtered = scan[keep].astype(np.float32, copy=False)
        filtered.tofile(limited_velodyne / scan_path.name)
        np.zeros(len(filtered), dtype=np.uint32).tofile(
            limited_labels / f"{scan_path.stem}.label"
        )

        before_count = len(scan)
        after_count = len(filtered)
        total_before += before_count
        total_after += after_count
        reports.append(
            {
                "frame": scan_path.stem,
                "before_points": before_count,
                "after_points": after_count,
            }
        )

    for source_path in source_seq.iterdir():
        if source_path.is_file():
            shutil.copy2(source_path, limited_seq / source_path.name)

    (limited_seq / "z_filter_report.yaml").write_text(
        yaml.safe_dump(
            {
                "source_dataset": str(source_root.resolve()),
                "method": method_name,
                "local_z_min": z_min,
                "local_z_max": z_max,
                "input_frames": len(scan_paths),
                "input_points": total_before,
                "output_points": total_after,
                "frames": reports,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    console.print(
        f"[dim]{method_name} 局部 Z 过滤: [{z_min:g}, {z_max:g}] m, "
        f"{total_before:,} -> {total_after:,} 点[/dim]"
    )
    return str(limited_root), len(scan_paths)


def _ensure_local_kitti_dataset(map_path):
    """确保存在适合 local hash voxel / raycasting 的逐帧 local KITTI 数据集。"""
    kitti_root = os.path.join(map_path, "erasor2_dataset")
    seq_dir = os.path.join(kitti_root, "dataset", "sequences", "00")
    velodyne_dir = os.path.join(seq_dir, "velodyne")
    pose_path = os.path.join(seq_dir, "poses_odom_base.txt")

    if os.path.isdir(velodyne_dir) and os.path.exists(pose_path):
        bin_files = [f for f in os.listdir(velodyne_dir) if f.endswith(".bin")]
        if bin_files and _has_current_bag_local_transform(seq_dir):
            _require_sensor_trajectory(seq_dir)
            print(f"复用已有 corrected 逐帧 KITTI 数据集: {velodyne_dir} ({len(bin_files)} 帧)")
            return kitti_root, len(bin_files)
        if bin_files:
            print("检测到未使用 Interactive SLAM 修正轨迹的 KITTI 数据集，将重新生成。")

    print("生成 bag 逐扫描点云 + corrected 插值轨迹 KITTI 数据集...")
    kitti_root, frame_count = convert_bag_to_kitti(map_path, _load_map_config(map_path))
    _require_sensor_trajectory(seq_dir)
    return kitti_root, frame_count


def _require_sensor_trajectory(seq_dir):
    pose_path = os.path.join(seq_dir, "poses_odom_base.txt")
    identity_path = os.path.join(seq_dir, "poses_identity.txt")

    if not os.path.exists(pose_path):
        if os.path.exists(identity_path):
            raise RuntimeError(
                "当前数据集只有 poses_identity.txt，没有真实传感器轨迹。"
                "local hash voxel 和 raycasting 需要逐帧传感器位姿，YunJingFull 这类数据暂不支持。"
            )
        raise RuntimeError(f"缺少真实传感器轨迹文件: {pose_path}")

    translations = []
    with open(pose_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) == 12:
                translations.append((vals[3], vals[7], vals[11]))
            elif len(vals) == 16:
                translations.append((vals[3], vals[7], vals[11]))
            else:
                raise RuntimeError(f"不支持的 pose 格式: {pose_path}")

    if len(translations) < 2:
        raise RuntimeError("真实传感器轨迹少于 2 帧，无法进行 local hash voxel/raycasting。")

    arr = np.asarray(translations, dtype=np.float64)
    movement = np.linalg.norm(arr - arr[0], axis=1).max()
    if movement < 1e-3:
        raise RuntimeError(
            "检测到传感器轨迹几乎全为同一位姿，无法进行 local hash voxel/raycasting。"
            "YunJingFull 这类无真实传感器轨迹的数据请先跳过。"
        )


def _run_python_script(script_name, args, output_dir, method_name):
    script_path = _PKG_DIR / "algorithms" / script_name
    cmd = [sys.executable, str(script_path)] + args
    return _run_measured_command(cmd, output_dir, method_name)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07")
_READ_PROGRESS_RE = re.compile(r"\[progress\]\s+readValidScans\s+(\d+)/(\d+)\s+\((\d+)%\)")
_MERGE_PROGRESS_RE = re.compile(r"\[progress\]\s+mergeScansWithinGlobalCoord\s+(\d+)/(\d+)\s+\((\d+)%\)")
_MAP_SIDE_PROGRESS_RE = re.compile(r"\[progress\]\s+map-side scan loop\s+(\d+)/(\d+)\s+\((\d+)%\)")
_MAP2RANGE_PROGRESS_RE = re.compile(r"\[progress\]\s+map2RangeImg\b.*?(\d+)%")
_REMOVE_ITER_RE = re.compile(r"\[progress\]\s+remove iteration\s+(\d+)/(\d+)")
_SAVE_PCD_RE = re.compile(
    r"(removert_(?:after|dynamic)(?:_local)?\.pcd|"
    r"(?:Dynamic|Static)MapMapside(?:Global|Local)ResX[0-9.]+\.pcd)"
)


def _strip_ansi(text):
    return _ANSI_RE.sub("", text)


def _run_removert_with_progress(
    cmd,
    log_path,
    output_dir,
    frame_count,
    docker_cidfile=None,
):
    """Run Removert while converting its stable progress log markers to stage bars."""
    saved_files = set()
    recent_errors = []
    remove_has_progress = False

    def _get_task(progress, task_id):
        return next(task for task in progress.tasks if task.id == task_id)

    def _remember_error(line):
        lower = line.lower()
        if any(token in lower for token in ("error", "failed", "exception", "abort", "terminate")):
            recent_errors.append(line.strip())
            del recent_errors[:-8]

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        monitor = _ProcessResourceMonitor(
            process,
            output_dir,
            "removert",
            docker_cidfile=docker_cidfile,
        ).start()

        try:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as progress:
                read_task = progress.add_task("读取点云", total=frame_count)
                merge_task = progress.add_task("构建地图", total=frame_count)
                remove_task = progress.add_task("动态清除", total=frame_count)
                write_task = progress.add_task("写出结果", total=4)

                assert process.stdout is not None
                for raw_line in process.stdout:
                    log_file.write(raw_line)
                    log_file.flush()

                    line = _strip_ansi(raw_line).strip()
                    if not line:
                        continue
                    _remember_error(line)

                    match = _READ_PROGRESS_RE.search(line)
                    if match:
                        done, total, _ = match.groups()
                        progress.update(read_task, completed=int(done), total=int(total))
                        continue

                    match = _MERGE_PROGRESS_RE.search(line)
                    if match:
                        done, total, _ = match.groups()
                        progress.update(merge_task, completed=int(done), total=int(total))
                        continue

                    match = _MAP_SIDE_PROGRESS_RE.search(line)
                    if match:
                        done, total, _ = match.groups()
                        remove_has_progress = True
                        progress.update(remove_task, completed=int(done), total=int(total))
                        continue

                    match = _MAP2RANGE_PROGRESS_RE.search(line)
                    if match and not remove_has_progress:
                        pct = max(0, min(100, int(match.group(1))))
                        progress.update(remove_task, completed=round(frame_count * pct / 100))
                        continue

                    match = _REMOVE_ITER_RE.search(line)
                    if match and not remove_has_progress:
                        done, total = (int(v) for v in match.groups())
                        remove_has_progress = True
                        progress.update(remove_task, completed=done, total=total)
                        continue

                    for filename in _SAVE_PCD_RE.findall(line):
                        saved_files.add(filename)
                    if saved_files:
                        progress.update(write_task, completed=min(len(saved_files), 4))

                returncode = process.wait()

                for filename in (
                    "removert_after.pcd",
                    "removert_dynamic.pcd",
                    "removert_after_local.pcd",
                    "removert_dynamic_local.pcd",
                ):
                    if os.path.exists(os.path.join(output_dir, filename)):
                        saved_files.add(filename)
                progress.update(write_task, completed=min(len(saved_files), 4))

                if returncode == 0:
                    for task_id in (read_task, merge_task, remove_task):
                        task = _get_task(progress, task_id)
                        if task.total is not None and task.completed < task.total:
                            progress.update(task_id, completed=task.total)
        finally:
            returncode = process.poll()
            if returncode is None:
                returncode = process.wait()
            monitor.finish(returncode)

        if returncode != 0 and recent_errors:
            console.print("[yellow]Removert 关键错误日志:[/yellow]")
            for line in recent_errors:
                print(f"  {line}")

        return returncode


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


def start_local_hash_voxel(map_path):
    """Local hash voxel 动态障碍物清除。"""
    try:
        kitti_root, frame_count = _ensure_local_kitti_dataset(map_path)
    except Exception as e:
        console.print(f"[red]Local Hash Voxel 无法运行: {e}[/red]")
        return

    print(f"\n检测到 {frame_count} 帧，准备运行 Local Hash Voxel 动态障碍物清除。\n")

    voxel_size = questionary.text("Hash voxel 尺寸 (米):", default="0.4").ask() or "0.4"
    max_range = questionary.text("最大水平距离 (米):", default="30.0").ask() or "30.0"
    local_z_min = questionary.text("局部 Z 下限 (米):", default="-2.5").ask() or "-2.5"
    local_z_max = questionary.text("局部 Z 上限 (米):", default="3.0").ask() or "3.0"
    ground_z = questionary.text(
        "地面保护局部 Z 上限 (留空关闭，常用 0.0 或 -0.1):",
        default="0.0",
    ).ask()
    unknown_policy = questionary.select(
        "unknown 体素处理策略:",
        choices=["keep", "drop"],
        default="keep",
    ).ask() or "keep"

    output_dir = _timestamped_output_dir(map_path, "local_hash_voxel")

    args = [
        "--dataset", kitti_root,
        "--out", output_dir,
        "--seq", "00",
        "--voxel-size", voxel_size,
        "--max-range", max_range,
        "--local-z-min", local_z_min,
        "--local-z-max", local_z_max,
        "--unknown-policy", unknown_policy,
        "--deduplicate", "quantized",
        "--interleave-after",
        "--save-dynamic-frames",
    ]
    if ground_z:
        args.extend(["--ground-protect-local-z-max", ground_z])

    ret = _run_python_script(
        "local_hash_voxel_filter.py",
        args,
        output_dir,
        "local_hash_voxel",
    )
    if ret != 0:
        console.print(f"[yellow]Local Hash Voxel 返回非零退出码: {ret}[/yellow]")
        return

    console.print("\n[bold green]Local Hash Voxel 处理完成！[/bold green]")
    print(f"  完整输出目录: {output_dir}")
    run_before = _publish_run_result(os.path.join(output_dir, "local_hash_voxel_before.pcd"), output_dir, "before.pcd", map_path, "map_local_hash_voxel_before.pcd", "Local Hash 移除前")
    run_static = _publish_run_result(os.path.join(output_dir, "local_hash_voxel_after.pcd"), output_dir, "static.pcd", map_path, "map_local_hash_voxel_static.pcd", "Local Hash 移除后")
    run_dynamic = _publish_run_result(os.path.join(output_dir, "local_hash_voxel_dynamic.pcd"), output_dir, "dynamic.pcd", map_path, "map_local_hash_voxel_dynamic.pcd", "Local Hash 被移除动态")
    _keep_only_standard_run_pcds(output_dir)
    _prompt_generate_dynamic_gif(
        map_path,
        output_dir,
        "local_hash_voxel",
        "map_local_hash_voxel_static.pcd",
        dynamic_reference=run_dynamic,
        before_reference=run_before,
        static_reference=run_static,
        z_min=float(local_z_min),
        z_max=float(local_z_max),
    )


def start_raycast_voxel(map_path):
    """Raycast voxel cleanup 动态障碍物清除。"""
    try:
        kitti_root, frame_count = _ensure_local_kitti_dataset(map_path)
    except Exception as e:
        console.print(f"[red]Raycast Voxel 无法运行: {e}[/red]")
        return

    print(f"\n检测到 {frame_count} 帧，准备运行 Raycast Voxel Cleanup。\n")

    voxel_size = questionary.text("Raycast voxel 尺寸 (米):", default="0.30").ask() or "0.30"
    max_range = questionary.text("最大水平距离 (米):", default="35.0").ask() or "35.0"
    body_radius = questionary.text("车体半径过滤 (米):", default="0.8").ask() or "0.8"
    local_z_min = questionary.text("Raycast 局部 Z 下限 (米):", default="-2.5").ask() or "-2.5"
    local_z_max = questionary.text("Raycast 局部 Z 上限 (米):", default="1.5").ask() or "1.5"
    ray_stride = questionary.text("Ray point stride:", default="8").ask() or "8"
    ground_z = questionary.text(
        "地面保护局部 Z 上限 (留空关闭，常用 0.0 或 -0.1):",
        default="0.0",
    ).ask()

    output_dir = _timestamped_output_dir(map_path, "raycast_voxel")

    args = [
        "--dataset", kitti_root,
        "--out", output_dir,
        "--seq", "00",
        "--voxel-size", voxel_size,
        "--max-range", max_range,
        "--body-radius", body_radius,
        "--local-z-min", local_z_min,
        "--local-z-max", local_z_max,
        "--ray-point-stride", ray_stride,
        "--write-before",
        "--deduplicate", "quantized",
        "--interleave-after",
        "--save-dynamic-frames",
    ]
    if ground_z:
        args.extend(["--ground-protect-local-z-max", ground_z])

    try:
        local_z_min_value = float(local_z_min)
    except ValueError:
        local_z_min_value = -2.5
    try:
        local_z_max_value = float(local_z_max)
    except ValueError:
        local_z_max_value = 1.5

    ret = _run_python_script(
        "raycast_voxel_cleanup.py",
        args,
        output_dir,
        "raycast_voxel",
    )
    if ret != 0:
        console.print(f"[yellow]Raycast Voxel 返回非零退出码: {ret}[/yellow]")
        return

    console.print("\n[bold green]Raycast Voxel 处理完成！[/bold green]")
    print(f"  完整输出目录: {output_dir}")
    run_before = _publish_run_result(os.path.join(output_dir, "raycast_before.pcd"), output_dir, "before.pcd", map_path, "map_raycast_voxel_before.pcd", "Raycast 移除前")
    run_static = _publish_run_result(os.path.join(output_dir, "raycast_after.pcd"), output_dir, "static.pcd", map_path, "map_raycast_voxel_static.pcd", "Raycast 移除后")
    run_dynamic = _publish_run_result(os.path.join(output_dir, "raycast_removed.pcd"), output_dir, "dynamic.pcd", map_path, "map_raycast_voxel_dynamic.pcd", "Raycast 被移除动态")
    _keep_only_standard_run_pcds(output_dir)
    _prompt_generate_dynamic_gif(
        map_path,
        output_dir,
        "raycast_voxel",
        "map_raycast_voxel_static.pcd",
        dynamic_reference=run_dynamic,
        before_reference=run_before,
        static_reference=run_static,
        z_min=local_z_min_value,
        z_max=local_z_max_value,
    )


def _ensure_or_pull_image(image, fallback=None):
    """检查 Docker 镜像是否存在，否则拉取。返回实际的 image tag。"""
    local = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if local.returncode == 0:
        print(f"本地已有镜像: {image}")
        return image

    if fallback:
        local2 = subprocess.run(
            ["docker", "image", "inspect", fallback],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if local2.returncode == 0:
            print(f"使用本地镜像: {fallback}")
            return fallback

    print(f"本地未找到镜像，正在从 Docker Hub 拉取 {image}...")
    subprocess.run(["docker", "pull", image], check=True)
    return image


def start_removert(map_path):
    """Removert 动态障碍物去除主流程。"""

    bag_dir = os.path.join(map_path, "bag")
    if not os.path.isdir(bag_dir):
        print(f"bag 目录 {bag_dir} 不存在。Removert 需要 bag 逐扫描点云。")
        return

    try:
        full_kitti_root, frame_count = _ensure_kitti_dataset(map_path)
    except Exception as e:
        console.print(f"[red]KITTI 转换失败: {e}[/red]")
        return

    print(f"\n检测到 {frame_count} 帧，准备运行 Removert 动态障碍物去除。\n")

    vfov_str = questionary.text("垂直 FOV (度):", default="50").ask()
    hfov_str = questionary.text("水平 FOV (度):", default="360").ask()
    local_z_min_str = questionary.text("Removert 局部 Z 下限 (米):", default="-2.5").ask() or "-2.5"
    local_z_max_str = questionary.text("Removert 局部 Z 上限 (米):", default="1.5").ask() or "1.5"
    batch_str = questionary.text("批处理大小:", default="150").ask()
    omp_str = questionary.text("OpenMP 核心数:", default="4").ask()

    try:
        vfov = float(vfov_str)
    except ValueError:
        vfov = 50
    try:
        hfov = float(hfov_str)
    except ValueError:
        hfov = 360
    try:
        local_z_min = float(local_z_min_str)
    except ValueError:
        local_z_min = -2.5
    try:
        local_z_max = float(local_z_max_str)
    except ValueError:
        local_z_max = 1.5
    if local_z_min > local_z_max:
        local_z_min, local_z_max = local_z_max, local_z_min
    try:
        batch_size = int(batch_str)
    except ValueError:
        batch_size = 150
    try:
        omp_cores = int(omp_str)
    except ValueError:
        omp_cores = 4

    # 输出目录
    output_dir = _timestamped_output_dir(map_path, "removert")

    try:
        kitti_root, frame_count = _prepare_z_limited_kitti_dataset(
            full_kitti_root,
            map_path,
            "removert",
            local_z_min,
            local_z_max,
        )
    except Exception as e:
        console.print(f"[red]Removert Z 过滤数据集生成失败: {e}[/red]")
        return

    scan_dir = os.path.join(kitti_root, "dataset", "sequences", "00", "velodyne")
    pose_path = os.path.join(kitti_root, "dataset", "sequences", "00", "poses_odom_base.txt")
    if not os.path.exists(pose_path):
        pose_path = os.path.join(kitti_root, "dataset", "sequences", "00", "poses_suma_optim.txt")

    # 生成配置文件
    params_text = f"""removert:
  isScanFileKITTIFormat: true

  saveMapPCD: true
  saveCleanScansPCD: false
  save_pcd_directory: "{output_dir}"

  sequence_scan_dir: "{scan_dir}"
  sequence_pose_path: "{pose_path}"

  sequence_vfov: {vfov}
  sequence_hfov: {hfov}

  ExtrinsicLiDARtoPoseBase: [1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0]

  use_keyframe_gap: true
  keyframe_gap: 1

  start_idx: 0
  end_idx: {frame_count - 1}

  clean_for_all_scan: false
  batch_size: {batch_size}
  valid_ratio_to_save: 0.75

  remove_resolution_list: [2.5, 2.0, 1.5]
  revert_resolution_list: [1.0, 0.9, 0.8, 0.7]

  downsample_voxel_size: 0.0

  num_nn_points_within: 2
  dist_nn_points_within: 0.1

  num_omp_cores: {omp_cores}

  rimg_color_min: 0.0
  rimg_color_max: 20.0
"""
    params_path = os.path.join(output_dir, "removert_params.yaml")
    Path(params_path).write_text(params_text)
    print(f"配置文件已生成: {params_path}")

    # ---- Docker 运行（workspace 已预编译在镜像内）----
    image = _ensure_or_pull_image(_REMOVERT_IMAGE)

    print("正在 Docker 容器中运行 Removert（可能需要数分钟）...")
    print(f"输出目录: {output_dir}")
    log_path = os.path.join(output_dir, "removert_docker.log")
    print(f"详细日志: {log_path}")

    docker_cmd = [
        "docker", "run", "--rm",
        "--memory=8g", "--cpus=4",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{kitti_root}:{kitti_root}:ro",
        "-v", f"{output_dir}:{output_dir}",
        "-w", _REMOVERT_WS,
        image,
        "bash", "-lc",
        "set -euo pipefail; "
        "source /opt/ros/noetic/setup.bash; "
        "source /opt/removert_ws/devel/setup.bash; "
        "roscore >/tmp/roscore.log 2>&1 & "
        "ROSCORE_PID=$!; "
        "trap 'kill $ROSCORE_PID 2>/dev/null' EXIT; "
        "for i in $(seq 1 30); do "
        "  if rosparam list >/dev/null 2>&1; then break; fi; "
        "  sleep 1; "
        "done; "
        f"rosparam load {params_path}; "
        "rosrun removert removert_removert",
    ]
    cidfile = _docker_cidfile(output_dir, "removert")
    docker_cmd = _add_docker_cidfile(docker_cmd, cidfile)

    returncode = _run_removert_with_progress(
        docker_cmd,
        log_path,
        output_dir,
        frame_count,
        docker_cidfile=cidfile,
    )
    if returncode != 0:
        console.print(f"[yellow]Docker 返回非零退出码: {returncode}，请检查日志: {log_path}[/yellow]")

    # ---- 整理结果 ----
    # Removert outputs:
    #   final maps: removert_after.pcd / _local.pcd, removert_dynamic.pcd / _local.pcd
    #   original maps may also be generated as removert_before.pcd / _local.pcd
    after_pcd = os.path.join(output_dir, "removert_after.pcd")
    before_pcd = os.path.join(output_dir, "removert_before.pcd")
    dynamic_pcd = os.path.join(output_dir, "removert_dynamic.pcd")

    run_before = os.path.join(output_dir, "before.pcd")
    run_static = os.path.join(output_dir, "static.pcd")
    run_dynamic = os.path.join(output_dir, "dynamic.pcd")

    copied = []
    limited_before_path = before_pcd if os.path.exists(before_pcd) else None

    full_before_path = run_before
    full_before_ready = False
    try:
        generated_before = _write_accumulated_kitti_map(full_kitti_root, full_before_path)
    except Exception as exc:
        console.print(f"[yellow]Removert before.pcd 生成失败: {exc}[/yellow]")
        generated_before = None
    if generated_before and os.path.exists(generated_before):
        run_before = generated_before
        full_before_ready = True
        copied.append("removert_before (完整原始地图)")
    else:
        run_before = None

    dynamic_source = None
    if os.path.exists(dynamic_pcd):
        dynamic_source = dynamic_pcd
    elif os.path.exists(after_pcd):
        if limited_before_path is None:
            limited_before_fallback = os.path.join(output_dir, "_removert_before_limited.pcd")
            try:
                limited_before_path = _write_accumulated_kitti_map(kitti_root, limited_before_fallback)
            except Exception as exc:
                console.print(f"[yellow]Removert z 范围内 before.pcd 生成失败: {exc}[/yellow]")
                limited_before_path = None
        if limited_before_path and os.path.exists(limited_before_path):
            raw_dynamic_path = os.path.join(output_dir, "_removert_dynamic_raw.pcd")
            if _write_removed_difference(limited_before_path, after_pcd, raw_dynamic_path):
                dynamic_source = raw_dynamic_path

    if dynamic_source and _publish_run_result(dynamic_source, output_dir, "dynamic.pcd"):
        copied.append("removert_dynamic (z 范围内被移除动态点云)")
        if full_before_ready and run_before and os.path.exists(run_before) and _write_removed_difference(run_before, run_dynamic, run_static):
            copied.append("removert_after (保留 z 范围外点的静态地图)")
        elif not full_before_ready:
            console.print("[yellow]Removert 缺少完整 before，无法补回 Z 范围外点生成 static.pcd。[/yellow]")
    elif dynamic_source is None:
        console.print("[yellow]Removert 未找到可用 dynamic 输出，无法生成符合 Z ROI 语义的 static.pcd。[/yellow]")
    _keep_only_standard_run_pcds(output_dir)

    if copied:
        console.print(f"\n[bold green]Removert 处理完成！[/bold green]")
        for name in copied:
            print(f"  ✓ {name}")
        print(f"  完整输出目录: {output_dir}/")
        static_map_path = run_static
        if os.path.exists(static_map_path):
            _prompt_generate_dynamic_gif(
                map_path,
                output_dir,
                "removert",
                "static.pcd",
                kitti_root=kitti_root,
                dynamic_reference=run_dynamic if os.path.exists(run_dynamic) else None,
                before_reference=run_before,
                static_reference=run_static or static_map_path,
                z_min=local_z_min,
                z_max=local_z_max,
            )
    else:
        console.print(f"\n[yellow]未找到 Removert 输出文件，请检查 Docker 日志。[/yellow]")
        print(f"  输出目录: {output_dir}/")
