"""Removert dynamic object removal workflow."""

import os
import re
import shutil
import subprocess

import questionary

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from .dynamic_removal_common import (
    _REMOVERT_IMAGE,
    _REMOVERT_WS,
    _ProcessResourceMonitor,
    _add_docker_cidfile,
    _docker_cidfile,
    _timestamped_output_dir,
    console,
)
from .dynamic_removal_docker import _ensure_or_pull_image
from .dynamic_removal_dataset import (
    _ensure_kitti_dataset,
    _prepare_z_limited_kitti_dataset,
)
from .dynamic_removal_results import (
    _keep_only_standard_run_pcds,
    _prompt_generate_dynamic_gif,
    _publish_run_result,
    _voxelize_pcd_file,
    _write_removed_difference,
)

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
