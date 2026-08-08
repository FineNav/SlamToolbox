import os
import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import questionary
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from .extractor import _read_pcd

console = Console()

POINT_CLOUD_SUFFIXES = {".pcd"}


def _cloud_files(directory):
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in POINT_CLOUD_SUFFIXES
    )


def _filter_z_range(xyz, z_min=None, z_max=None):
    if xyz is None or len(xyz) == 0 or (z_min is None and z_max is None):
        return xyz
    mask = np.ones(len(xyz), dtype=bool)
    if z_min is not None:
        mask &= xyz[:, 2] >= z_min
    if z_max is not None:
        mask &= xyz[:, 2] <= z_max
    return xyz[mask]


def _load_xyz(path, apply_odom=False, z_min=None, z_max=None):
    xyz, _ = _read_pcd(str(path))
    if xyz is None or len(xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    odom_path = Path(path).with_suffix(".odom")
    if apply_odom and odom_path.exists():
        try:
            pose = np.loadtxt(odom_path)
            if pose.shape == (4, 4):
                pts_h = np.ones((len(xyz), 4), dtype=np.float64)
                pts_h[:, :3] = xyz
                xyz = (pose @ pts_h.T).T[:, :3].astype(np.float32)
        except Exception as exc:
            console.print(f"[yellow]跳过无效 odom: {odom_path} ({exc})[/yellow]")

    return _filter_z_range(xyz, z_min=z_min, z_max=z_max)


def _sample_xyz(xyz, max_points, seed):
    if max_points <= 0 or len(xyz) <= max_points:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), max_points, replace=False)
    return xyz[idx]


def _append_limited_history(history, current, max_points, seed):
    if len(current) == 0:
        return history
    if history is None or len(history) == 0:
        merged = current
    else:
        merged = np.vstack([history, current])
    return _sample_xyz(merged, max_points, seed)


def _padded_bounds(xy_min, xy_max, padding_ratio=0.05):
    mins = np.asarray(xy_min, dtype=np.float64)
    maxs = np.asarray(xy_max, dtype=np.float64)
    span = np.maximum(maxs - mins, 1.0)
    padding = span * padding_ratio
    return mins - padding, maxs + padding


def _set_axes_3d_equal(ax, xyz_min, xyz_max):
    center = (xyz_min + xyz_max) * 0.5
    span = np.maximum(xyz_max - xyz_min, 1.0)
    radius = float(np.max(span) * 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _fit_bev_bounds(xy_min, xy_max, width, height):
    xy_min, xy_max = _padded_bounds(xy_min, xy_max)
    center = (xy_min + xy_max) * 0.5
    span = np.maximum(xy_max - xy_min, 1.0)
    target_aspect = width / height
    if span[0] / span[1] < target_aspect:
        span[0] = span[1] * target_aspect
    else:
        span[1] = span[0] / target_aspect
    return center - span * 0.5, center + span * 0.5


def _paint_bev_points(image, xyz, xy_min, xy_max, color, radius=0):
    if xyz is None or len(xyz) == 0:
        return
    height, width = image.shape[:2]
    xy = xyz[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    if not finite.any():
        return
    xy = xy[finite]
    scale = np.array([width - 1, height - 1], dtype=np.float64) / np.maximum(xy_max - xy_min, 1e-9)
    pixels = np.rint((xy - xy_min) * scale).astype(np.int64)
    pixels[:, 1] = height - 1 - pixels[:, 1]
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    pixels = pixels[valid]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px = pixels[:, 0] + dx
            py = pixels[:, 1] + dy
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            image[py[inside], px[inside]] = color


def _create_fast_bev_gif(
    files,
    static_viz,
    output_path,
    duration,
    trail,
    apply_odom,
    max_dynamic_points,
    max_frames,
    frame_width,
    frame_height,
    render_scale,
    xy_min,
    xy_max,
    z_min,
    z_max,
    target_duration,
):
    from PIL import Image

    render_width = frame_width * render_scale
    render_height = frame_height * render_scale
    xy_min, xy_max = _fit_bev_bounds(xy_min, xy_max, render_width, render_height)
    background = np.zeros((render_height, render_width, 3), dtype=np.uint8)
    _paint_bev_points(background, static_viz, xy_min, xy_max, (88, 88, 88), radius=1)
    trajectory = background.copy() if trail < 0 else None
    trail_groups = []

    frame_stride = max(1, math.ceil(len(files) / max_frames)) if max_frames > 0 else 1
    output_count = math.ceil(len(files) / frame_stride)
    if duration is None:
        duration = target_duration / max(output_count, 1)
        console.print(f"[dim]GIF 自动播放时长: 约 {target_duration:g} 秒 ({duration:.4f} 秒/帧)[/dim]")
    if frame_stride > 1:
        console.print(
            f"[dim]GIF 观察模式: {len(files)} 个输入帧压缩为约 {output_count} 个渲染帧[/dim]"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("渲染 GIF 帧...", total=output_count)
            current_group = []
            output_index = 0
            for index, path in enumerate(files):
                current = _load_xyz(path, apply_odom=apply_odom, z_min=z_min, z_max=z_max)
                current = _sample_xyz(current, max_dynamic_points, seed=1000 + index)
                if len(current) > 0:
                    current_group.append(current)
                    if trail < 0:
                        _paint_bev_points(trajectory, current, xy_min, xy_max, (62, 124, 255), radius=2)

                group_done = (index + 1) % frame_stride == 0 or index == len(files) - 1
                if not group_done:
                    continue

                frame = trajectory.copy() if trail < 0 else background.copy()
                if trail > 0:
                    for old in trail_groups:
                        _paint_bev_points(frame, old, xy_min, xy_max, (62, 124, 255), radius=2)
                if current_group:
                    current_viz = np.vstack(current_group)
                    current_viz = _sample_xyz(current_viz, max_dynamic_points, seed=5000 + output_index)
                    _paint_bev_points(frame, current_viz, xy_min, xy_max, (31, 170, 255), radius=3)
                else:
                    current_viz = None
                output_frame = np.asarray(
                    Image.fromarray(frame).resize(
                        (frame_width, frame_height),
                        resample=Image.Resampling.LANCZOS,
                    )
                )
                writer.append_data(output_frame)
                if trail > 0 and current_viz is not None and len(current_viz) > 0:
                    trail_groups.append(current_viz)
                    trail_groups = trail_groups[-trail:]
                current_group.clear()
                output_index += 1
                progress.update(task, advance=1)


def create_dynamic_overlay_gif(
    static_path,
    dynamic_dir,
    output_path,
    duration=None,
    trail=-1,
    apply_odom=False,
    max_static_points=600_000,
    max_dynamic_points=80_000,
    max_trajectory_points=400_000,
    max_frames=180,
    frame_width=2560,
    frame_height=1440,
    render_scale=2,
    view_mode="bev",
    elev=35.264,
    azim=-45.0,
    rotate_degrees=0.0,
    z_min=None,
    z_max=None,
    target_duration=10.0,
    dpi=160,
):
    """Generate a bird's-eye GIF of dynamic point clouds over a static map.

    trail > 0 keeps the last N dynamic frames as a fading trail.
    trail < 0 keeps a persistent accumulated trajectory for the whole GIF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    static_path = Path(static_path)
    dynamic_dir = Path(dynamic_dir)
    output_path = Path(output_path)

    if not static_path.exists():
        raise FileNotFoundError(f"静态地图不存在: {static_path}")
    if not dynamic_dir.is_dir():
        raise FileNotFoundError(f"动态点云目录不存在: {dynamic_dir}")

    files = _cloud_files(dynamic_dir)
    if not files:
        raise FileNotFoundError(f"动态点云目录中没有 .pcd 文件: {dynamic_dir}")
    if view_mode not in ("bev", "3d"):
        raise ValueError("view_mode must be 'bev' or '3d'")

    console.print(f"[dim]加载静态地图: {static_path}[/dim]")
    static_xyz = _load_xyz(static_path, z_min=z_min, z_max=z_max)
    if len(static_xyz) == 0:
        raise RuntimeError(f"静态地图为空或 Z 范围内无点: {static_path}")
    if z_min is not None or z_max is not None:
        z_min_text = "-inf" if z_min is None else f"{z_min:g}"
        z_max_text = "inf" if z_max is None else f"{z_max:g}"
        console.print(f"[dim]GIF Z 过滤范围: [{z_min_text}, {z_max_text}] m[/dim]")
    finite_mask = np.isfinite(static_xyz).all(axis=1)
    finite_static = static_xyz if finite_mask.all() else static_xyz[finite_mask]
    if len(finite_static) == 0:
        raise RuntimeError(f"静态地图不包含有效坐标: {static_path}")
    xy_min = np.nanmin(finite_static[:, :2], axis=0)
    xy_max = np.nanmax(finite_static[:, :2], axis=0)
    xyz_min = np.nanmin(finite_static[:, :3], axis=0)
    xyz_max = np.nanmax(finite_static[:, :3], axis=0)
    static_viz = _sample_xyz(finite_static, max_static_points, seed=42)

    if view_mode == "bev":
        del static_xyz, finite_static
        _create_fast_bev_gif(
            files=files,
            static_viz=static_viz,
            output_path=output_path,
            duration=duration,
            trail=trail,
            apply_odom=apply_odom,
            max_dynamic_points=max_dynamic_points,
            max_frames=max_frames,
            frame_width=frame_width,
            frame_height=frame_height,
            render_scale=render_scale,
            xy_min=xy_min,
            xy_max=xy_max,
            z_min=z_min,
            z_max=z_max,
            target_duration=target_duration,
        )
        console.print(f"[green]✓ 动态障碍物 GIF 已生成 → {output_path}[/green]")
        return str(output_path)

    frame_stride = max(1, math.ceil(len(files) / max_frames)) if max_frames > 0 else 1
    render_files = files[::frame_stride]
    if render_files[-1] != files[-1]:
        render_files.append(files[-1])
    if duration is None:
        duration = target_duration / max(len(render_files), 1)
        console.print(f"[dim]GIF 自动播放时长: 约 {target_duration:g} 秒 ({duration:.4f} 秒/帧)[/dim]")
    if frame_stride > 1:
        console.print(
            f"[dim]GIF 3D 观察模式: {len(files)} 个输入帧压缩为 {len(render_files)} 个渲染帧[/dim]"
        )

    console.print(f"[dim]扫描动态点云范围: {dynamic_dir}[/dim]")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("扫描动态点云...", total=len(files))
        for index, path in enumerate(files):
            xyz = _load_xyz(path, apply_odom=apply_odom, z_min=z_min, z_max=z_max)
            xyz = _sample_xyz(xyz, max_dynamic_points, seed=1000 + index)
            if len(xyz) > 0:
                xy_min = np.minimum(xy_min, np.nanmin(xyz[:, :2], axis=0))
                xy_max = np.maximum(xy_max, np.nanmax(xyz[:, :2], axis=0))
                xyz_min = np.minimum(xyz_min, np.nanmin(xyz[:, :3], axis=0))
                xyz_max = np.maximum(xyz_max, np.nanmax(xyz[:, :3], axis=0))
            progress.update(task, advance=1)

    xy_min, xy_max = _padded_bounds(xy_min, xy_max)
    xyz_min, xyz_max = _padded_bounds(xyz_min, xyz_max)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output_path, mode="I", duration=duration) as writer:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("渲染 GIF 帧...", total=len(render_files))
            trail_frames = []
            trajectory_points = None

            for index, path in enumerate(render_files):
                current = _load_xyz(path, apply_odom=apply_odom, z_min=z_min, z_max=z_max)
                current = _sample_xyz(current, max_dynamic_points, seed=1000 + index)
                fig = plt.figure(figsize=(frame_width / dpi, frame_height / dpi), dpi=dpi)
                fig.patch.set_facecolor("black")
                if view_mode == "3d":
                    ax = fig.add_subplot(111, projection="3d")
                else:
                    ax = fig.add_subplot(111)
                ax.set_facecolor("black")

                if view_mode == "3d":
                    ax.scatter(
                        static_viz[:, 0],
                        static_viz[:, 1],
                        static_viz[:, 2],
                        c="#606060",
                        s=0.16,
                        alpha=0.46,
                        linewidths=0,
                        rasterized=True,
                        depthshade=False,
                    )
                else:
                    ax.scatter(
                        static_viz[:, 0],
                        static_viz[:, 1],
                        c="#606060",
                        s=0.12,
                        alpha=0.55,
                        linewidths=0,
                        rasterized=True,
                    )

                if trail < 0 and trajectory_points is not None and len(trajectory_points) > 0:
                    if view_mode == "3d":
                        ax.scatter(
                        trajectory_points[:, 0],
                        trajectory_points[:, 1],
                        trajectory_points[:, 2],
                        c="#4f8dff",
                        s=0.58,
                        alpha=0.70,
                            linewidths=0,
                            rasterized=True,
                            depthshade=False,
                        )
                    else:
                        ax.scatter(
                            trajectory_points[:, 0],
                            trajectory_points[:, 1],
                        c="#4f8dff",
                            s=0.38,
                            alpha=0.58,
                            linewidths=0,
                            rasterized=True,
                        )
                elif trail > 0:
                    for offset, old in enumerate(trail_frames):
                        if len(old) == 0:
                            continue
                        alpha = 0.15 + 0.35 * ((offset + 1) / max(len(trail_frames), 1))
                        if view_mode == "3d":
                            ax.scatter(
                                old[:, 0],
                                old[:, 1],
                                old[:, 2],
                                c="#4f8dff",
                                s=0.62,
                                alpha=alpha,
                                linewidths=0,
                                rasterized=True,
                                depthshade=False,
                            )
                        else:
                            ax.scatter(
                                old[:, 0],
                                old[:, 1],
                                c="#4f8dff",
                                s=0.45,
                                alpha=alpha,
                                linewidths=0,
                                rasterized=True,
                            )

                if len(current) > 0:
                    if view_mode == "3d":
                        ax.scatter(
                            current[:, 0],
                            current[:, 1],
                            current[:, 2],
                            c="#1fa6ff",
                            s=1.10,
                            alpha=0.98,
                            linewidths=0,
                            rasterized=True,
                            depthshade=False,
                        )
                    else:
                        ax.scatter(
                            current[:, 0],
                            current[:, 1],
                            c="#1fa6ff",
                            s=0.75,
                            alpha=0.95,
                            linewidths=0,
                            rasterized=True,
                        )

                if view_mode == "3d":
                    frame_azim = azim + rotate_degrees * index / max(len(render_files) - 1, 1)
                    _set_axes_3d_equal(ax, xyz_min, xyz_max)
                    ax.view_init(elev=elev, azim=frame_azim)
                    ax.grid(False)
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    ax.xaxis.line.set_color((0, 0, 0, 0))
                    ax.yaxis.line.set_color((0, 0, 0, 0))
                    ax.zaxis.line.set_color((0, 0, 0, 0))
                else:
                    ax.set_xlim(xy_min[0], xy_max[0])
                    ax.set_ylim(xy_min[1], xy_max[1])
                    ax.set_aspect("equal", adjustable="box")
                ax.axis("off")
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                writer.append_data(frame)
                plt.close(fig)

                if trail < 0:
                    trajectory_points = _append_limited_history(
                        trajectory_points,
                        current,
                        max_trajectory_points,
                        seed=2000 + index,
                    )
                elif trail > 0:
                    trail_frames.append(current)
                    trail_frames = trail_frames[-trail:]
                progress.update(task, advance=1)

    console.print(f"[green]✓ 动态障碍物 GIF 已生成 → {output_path}[/green]")
    return str(output_path)


def _default_static_map(map_path):
    candidates = [
        os.path.join(map_path, "map", "map_erasor2_static_cleaned.pcd"),
        os.path.join(map_path, "map", "map_erasor2_static.pcd"),
        os.path.join(map_path, "map", "map_removert_static.pcd"),
        os.path.join(map_path, "map", "map.pcd"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def start_dynamic_overlay_gif(map_path):
    static_default = _default_static_map(map_path)
    dynamic_default = os.path.join(map_path, "dynamic_frames")
    output_default = os.path.join(map_path, "visualize", "dynamic_overlay.gif")

    static_path = questionary.text("静态地图 PCD 路径:", default=static_default).ask()
    dynamic_dir = questionary.text(
        "动态障碍物逐帧 PCD 目录:",
        default=dynamic_default,
    ).ask()
    output_path = questionary.text("输出 GIF 路径:", default=output_default).ask()

    trail_str = questionary.text("残影帧数 (-1 表示永久轨迹):", default="-1").ask()
    apply_odom = questionary.confirm(
        "动态点云是否需要按同名 .odom 变换到全局坐标系？",
        default=False,
    ).ask()
    view_mode = questionary.select(
        "GIF 视角:",
        choices=["bev", "3d"],
        default="3d",
    ).ask() or "3d"
    rotate_degrees_str = questionary.text(
        "3D 相机旋转角度 (0 表示固定视角):",
        default="0",
    ).ask()
    z_min_str = questionary.text("GIF Z 过滤下限 (留空表示不限):", default="").ask()
    z_max_str = questionary.text("GIF Z 过滤上限 (留空表示不限):", default="").ask()

    try:
        trail = int(trail_str)
    except (TypeError, ValueError):
        trail = 5
    try:
        rotate_degrees = float(rotate_degrees_str)
    except (TypeError, ValueError):
        rotate_degrees = 0.0
    try:
        z_min = float(z_min_str) if z_min_str not in (None, "") else None
    except ValueError:
        z_min = None
    try:
        z_max = float(z_max_str) if z_max_str not in (None, "") else None
    except ValueError:
        z_max = None

    try:
        create_dynamic_overlay_gif(
            static_path=static_path,
            dynamic_dir=dynamic_dir,
            output_path=output_path,
            duration=None,
            trail=trail,
            apply_odom=bool(apply_odom),
            view_mode=view_mode,
            rotate_degrees=rotate_degrees,
            z_min=z_min,
            z_max=z_max,
        )
    except Exception as exc:
        console.print(f"[red]生成动态障碍物 GIF 失败: {exc}[/red]")


def main():
    parser = argparse.ArgumentParser(
        description="将动态障碍物逐帧点云叠加到静态地图上并生成 GIF。"
    )
    parser.add_argument("--static", required=True, help="静态地图 PCD 路径。")
    parser.add_argument("--dynamic-dir", required=True, help="动态障碍物逐帧 PCD 目录。")
    parser.add_argument("--output", required=True, help="输出 GIF 路径。")
    parser.add_argument("--duration", type=float, default=None, help="每帧时长，单位秒；默认按目标总时长自动计算。")
    parser.add_argument("--target-duration", type=float, default=10.0, help="自动速度的目标播放总时长，单位秒。")
    parser.add_argument("--trail", type=int, default=-1, help="保留多少帧橙色残影；-1 表示永久轨迹。")
    parser.add_argument("--view-mode", choices=("bev", "3d"), default="3d", help="GIF 视角。")
    parser.add_argument("--elev", type=float, default=35.264, help="3D 视角仰角。")
    parser.add_argument("--azim", type=float, default=-45.0, help="3D 视角方位角。")
    parser.add_argument("--rotate-degrees", type=float, default=0.0, help="整个 GIF 期间额外旋转的角度。")
    parser.add_argument("--z-min", type=float, default=None, help="GIF 渲染的 Z 下限，默认不限。")
    parser.add_argument("--z-max", type=float, default=None, help="GIF 渲染的 Z 上限，默认不限。")
    parser.add_argument(
        "--apply-odom",
        action="store_true",
        help="对动态点云应用同名 .odom 位姿，将局部帧变换到全局坐标系。",
    )
    parser.add_argument("--max-static-points", type=int, default=600_000)
    parser.add_argument("--max-dynamic-points", type=int, default=80_000)
    parser.add_argument("--max-trajectory-points", type=int, default=400_000)
    args = parser.parse_args()

    create_dynamic_overlay_gif(
        static_path=args.static,
        dynamic_dir=args.dynamic_dir,
        output_path=args.output,
        duration=args.duration,
        trail=args.trail,
        apply_odom=args.apply_odom,
        max_static_points=args.max_static_points,
        max_dynamic_points=args.max_dynamic_points,
        max_trajectory_points=args.max_trajectory_points,
        view_mode=args.view_mode,
        elev=args.elev,
        azim=args.azim,
        rotate_degrees=args.rotate_degrees,
        z_min=args.z_min,
        z_max=args.z_max,
        target_duration=args.target_duration,
    )


if __name__ == "__main__":
    main()
