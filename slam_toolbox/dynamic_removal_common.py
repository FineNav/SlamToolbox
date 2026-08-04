"""Shared helpers for dynamic-removal workflows."""

import json
import os
import platform
import re
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console

console = Console()

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


