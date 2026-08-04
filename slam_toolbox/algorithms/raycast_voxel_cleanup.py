#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


class OctreeNode:
    __slots__ = (
        "origin",
        "size",
        "children",
        "log_odds",
        "hit_frames",
        "miss_frames",
        "hit_points",
    )

    def __init__(self, origin: tuple[int, int, int], size: int) -> None:
        self.origin = origin
        self.size = size
        self.children: list[OctreeNode | None] | None = None
        self.log_odds = 0.0
        self.hit_frames = 0
        self.miss_frames = 0
        self.hit_points = 0

    @property
    def is_leaf_cell(self) -> bool:
        return self.size == 1


class OctreeMap:
    def __init__(self) -> None:
        self.root: OctreeNode | None = None
        self.leaf_count = 0
        self._leaves_by_coord: dict[tuple[int, int, int], OctreeNode] = {}

    @staticmethod
    def _contains(node: OctreeNode, coord: tuple[int, int, int]) -> bool:
        ox, oy, oz = node.origin
        size = node.size
        return (
            ox <= coord[0] < ox + size
            and oy <= coord[1] < oy + size
            and oz <= coord[2] < oz + size
        )

    def _ensure_contains(self, coord: tuple[int, int, int]) -> None:
        if self.root is None:
            self.root = OctreeNode(coord, 1)
            return

        while not self._contains(self.root, coord):
            old = self.root
            old_origin = old.origin
            old_size = old.size
            new_size = old_size * 2
            new_origin = (
                old_origin[0] - old_size if coord[0] < old_origin[0] else old_origin[0],
                old_origin[1] - old_size if coord[1] < old_origin[1] else old_origin[1],
                old_origin[2] - old_size if coord[2] < old_origin[2] else old_origin[2],
            )
            new_root = OctreeNode(new_origin, new_size)
            new_root.children = [None] * 8
            ix = 1 if old_origin[0] >= new_origin[0] + old_size else 0
            iy = 1 if old_origin[1] >= new_origin[1] + old_size else 0
            iz = 1 if old_origin[2] >= new_origin[2] + old_size else 0
            new_root.children[ix | (iy << 1) | (iz << 2)] = old
            self.root = new_root

    def get_or_create_leaf(self, coord: tuple[int, int, int]) -> OctreeNode:
        existing = self._leaves_by_coord.get(coord)
        if existing is not None:
            return existing

        self._ensure_contains(coord)
        assert self.root is not None
        node = self.root
        while node.size > 1:
            half = node.size // 2
            ox, oy, oz = node.origin
            ix = 1 if coord[0] >= ox + half else 0
            iy = 1 if coord[1] >= oy + half else 0
            iz = 1 if coord[2] >= oz + half else 0
            child_idx = ix | (iy << 1) | (iz << 2)
            child_origin = (ox + ix * half, oy + iy * half, oz + iz * half)
            if node.children is None:
                node.children = [None] * 8
            child = node.children[child_idx]
            if child is None:
                child = OctreeNode(child_origin, half)
                node.children[child_idx] = child
                if half == 1:
                    self.leaf_count += 1
            node = child
        if self.leaf_count == 0 and self.root is node:
            self.leaf_count = 1
        self._leaves_by_coord[coord] = node
        return node

    def get_leaf(self, coord: tuple[int, int, int]) -> OctreeNode | None:
        return self._leaves_by_coord.get(coord)

    def update_hit(self, coord: tuple[int, int, int], count: int, hit_log_odds: float) -> None:
        leaf = self.get_or_create_leaf(coord)
        leaf.hit_frames += 1
        leaf.hit_points += count
        leaf.log_odds += hit_log_odds

    def update_miss(self, coord: tuple[int, int, int], miss_log_odds: float) -> None:
        leaf = self.get_leaf(coord)
        if leaf is None:
            return
        leaf.miss_frames += 1
        leaf.log_odds -= miss_log_odds

    def is_occupied(self, coord: tuple[int, int, int], occupied_threshold: float) -> bool:
        leaf = self.get_leaf(coord)
        return bool(leaf is not None and leaf.log_odds >= occupied_threshold)

    def free_coords(self, free_threshold: float) -> set[tuple[int, int, int]]:
        return {
            coord
            for coord, leaf in self._leaves_by_coord.items()
            if leaf.log_odds <= free_threshold
        }

    def count_occupied(self, occupied_threshold: float) -> int:
        return len(self.occupied_coords(occupied_threshold))

    def occupied_coords(self, occupied_threshold: float) -> set[tuple[int, int, int]]:
        return {
            coord
            for coord, leaf in self._leaves_by_coord.items()
            if leaf.log_odds >= occupied_threshold
        }

    def iter_leaves(self):
        yield from self._leaves_by_coord.values()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Raycast voxel cleanup for converted KITTI-style lidar sequences. "
            "Use local scans plus true poses_odom_base.txt; do not use identity global scans."
        )
    )
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--seq", default="00")
    p.add_argument("--pose", type=Path, help="Default: dataset/sequences/<seq>/poses_odom_base.txt")
    p.add_argument("--out", type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    p.add_argument("--voxel-size", type=float, default=0.5)
    p.add_argument("--max-range", type=float, default=30.0, help="Horizontal range in local frame; 0 disables.")
    p.add_argument("--body-radius", type=float, default=0.5, help="Drop points near the vehicle in local xy; 0 disables.")
    p.add_argument("--local-z-min", type=float, default=0.0, help="Only points within this local z ROI can be removed as dynamic.")
    p.add_argument("--local-z-max", type=float, default=3.0, help="Only points within this local z ROI can be removed as dynamic.")
    p.add_argument(
        "--ground-protect-local-z-max",
        type=float,
        help=(
            "Force-keep points whose original local/base_link z is <= this value. "
            "Useful when raycasting removes floor points; try 0.0 or -0.1."
        ),
    )
    p.add_argument("--ray-point-stride", type=int, default=4, help="Use every Nth point for free-space raycasting.")
    p.add_argument("--ray-step-factor", type=float, default=0.75, help="Ray sample step = voxel_size * factor.")
    p.add_argument("--endpoint-margin", type=float, default=0.10, help="Meters before endpoint left untouched by free rays.")
    p.add_argument("--hit-log-odds", type=float, default=0.40)
    p.add_argument("--miss-log-odds", type=float, default=0.80)
    p.add_argument("--occupied-threshold", type=float, default=1.5)
    p.add_argument("--free-threshold", type=float, default=-1.0)
    p.add_argument("--unknown-policy", choices=("keep", "drop"), default="keep")
    p.add_argument("--write-before", action="store_true", help="Also write raycast_before.pcd.")
    p.add_argument(
        "--save-dynamic-frames",
        action="store_true",
        help="Also write per-frame removed point clouds under <out>/dynamic_frames for GIF visualization.",
    )
    p.add_argument("--progress-interval", type=int, default=25)
    return p.parse_args()


def load_poses(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) == 12:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
        elif len(vals) == 16:
            mat = np.asarray(vals, dtype=np.float64).reshape(4, 4)
        else:
            raise ValueError(f"unsupported pose line {line_no}: {len(vals)} values in {path}")
        poses.append(mat)
    return poses


def transform_scan(
    scan_path: Path,
    pose: np.ndarray,
    body_radius: float,
    max_range: float,
    local_z_min: float | None,
    local_z_max: float | None,
    apply_local_z: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    scan = scan[np.isfinite(scan).all(axis=1)]
    if scan.size == 0:
        return scan, scan
    dist2 = scan[:, 0].astype(np.float64) ** 2 + scan[:, 1].astype(np.float64) ** 2
    keep = np.ones(scan.shape[0], dtype=bool)
    if body_radius > 0:
        keep &= dist2 >= body_radius * body_radius
    if max_range > 0:
        keep &= dist2 <= max_range * max_range
    if apply_local_z:
        if local_z_min is not None:
            keep &= scan[:, 2] >= local_z_min
        if local_z_max is not None:
            keep &= scan[:, 2] <= local_z_max
    scan = scan[keep]
    if scan.size == 0:
        return scan, scan
    xyz = scan[:, :3].astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
    out = np.empty_like(scan, dtype=np.float32)
    out[:, :3] = xyz.astype(np.float32)
    out[:, 3] = scan[:, 3]
    return out, scan


def voxel_coords_for_points(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(xyz.astype(np.float64) / voxel_size).astype(np.int64)


def ray_free_coords(
    origin: np.ndarray,
    endpoints: np.ndarray,
    voxel_size: float,
    step_factor: float,
    endpoint_margin: float,
) -> set[tuple[int, int, int]]:
    free: set[tuple[int, int, int]] = set()
    step = max(voxel_size * step_factor, voxel_size * 0.25)
    for endpoint in endpoints:
        vec = endpoint.astype(np.float64) - origin
        dist = float(np.linalg.norm(vec))
        usable = dist - endpoint_margin
        if usable <= step:
            continue
        samples = max(1, int(math.floor(usable / step)))
        ts = (np.arange(1, samples + 1, dtype=np.float64) * step) / dist
        pts = origin[None, :] + ts[:, None] * vec[None, :]
        coords = voxel_coords_for_points(pts, voxel_size)
        if coords.shape[0] > 1:
            # Along one straight ray, repeated visits to a voxel are contiguous.
            # Removing adjacent duplicates preserves the voxel set without sorting.
            keep = np.empty(coords.shape[0], dtype=bool)
            keep[0] = True
            keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
            coords = coords[keep]
        free.update((int(c[0]), int(c[1]), int(c[2])) for c in coords)
    return free


def write_pcd_header(out_path: Path, point_count: int) -> None:
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {point_count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {point_count}\n"
        "DATA binary\n"
    ).encode("ascii")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(header)


def write_pcd_from_payload(payload_path: Path, point_count: int, out_path: Path) -> None:
    write_pcd_header(out_path, point_count)
    with out_path.open("ab") as dst, payload_path.open("rb") as src:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)


def write_pcd_from_points(points: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    points = points.astype(np.float32, copy=False).reshape(-1, 4)
    write_pcd_header(out_path, points.shape[0])
    with out_path.open("ab") as dst:
        points.tofile(dst)


def choose_out_dir(dataset: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    workspace = Path(__file__).resolve().parent.parent
    run_root = workspace / "run_results" / dataset.resolve().name / "raycast_voxel_runs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_raycast_voxel"
    out = run_root / stamp
    suffix = 1
    while out.exists():
        out = run_root / f"{stamp}_{suffix}"
        suffix += 1
    return out


def print_progress(phase: str, done: int, total: int, started: float, detail: str = "") -> None:
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, done / total))
    percent = 100.0 * fraction
    elapsed = time.time() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    bar_width = 30
    filled = int(bar_width * fraction)
    bar = "=" * filled + "-" * (bar_width - filled)
    suffix = f" | {detail}" if detail else ""
    print(
        f"[{phase:<9}] [{bar}] {percent:6.2f}% "
        f"({done}/{total}) elapsed={elapsed:.1f}s eta={eta:.1f}s{suffix}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be > 0")
    if args.stride <= 0 or args.ray_point_stride <= 0:
        raise ValueError("--stride and --ray-point-stride must be > 0")
    if args.ray_step_factor <= 0:
        raise ValueError("--ray-step-factor must be > 0")
    seq_dir = args.dataset / "dataset" / "sequences" / args.seq
    velodyne_dir = seq_dir / "velodyne"
    scan_paths = sorted(velodyne_dir.glob("*.bin"))
    if not scan_paths:
        raise FileNotFoundError(f"no .bin scans found in {velodyne_dir}")
    pose_path = args.pose or (seq_dir / "poses_odom_base.txt")
    if not pose_path.exists():
        raise FileNotFoundError(f"pose file not found: {pose_path}")
    poses = load_poses(pose_path)

    end = args.end if args.end is not None else len(scan_paths) - 1
    frame_ids = list(range(args.start, end + 1, args.stride))
    if args.start < 0 or end >= len(scan_paths) or not frame_ids:
        raise ValueError(f"invalid frame range {args.start}..{end}; scan count={len(scan_paths)}")
    if len(poses) <= end:
        raise ValueError(f"pose count {len(poses)} is smaller than end frame {end}")
    if pose_path.name == "poses_identity.txt":
        raise ValueError(
            "poses_identity.txt does not provide real sensor origins. "
            "Use datasets with poses_odom_base.txt for raycasting."
        )

    out_dir = choose_out_dir(args.dataset, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="raycast_voxel_", dir=str(out_dir)))

    octree = OctreeMap()

    started = time.time()
    total_hit_points = 0
    total_ray_points = 0
    print(
        f"[raycast] dataset={args.dataset} seq={args.seq} frames={args.start}..{end} "
        f"step={args.stride} pose={pose_path} out={out_dir} "
        f"z=[{args.local_z_min}, {args.local_z_max}]",
        flush=True,
    )

    for done, frame_id in enumerate(frame_ids, 1):
        pose = poses[frame_id]
        origin = pose[:3, 3].astype(np.float64)
        cloud, _local_cloud = transform_scan(
            scan_paths[frame_id],
            pose,
            args.body_radius,
            args.max_range,
            args.local_z_min,
            args.local_z_max,
            True,
        )
        if cloud.size == 0:
            continue

        xyz = cloud[:, :3]
        hit_coords_all = voxel_coords_for_points(xyz, args.voxel_size)
        unique_hit_coords, counts = np.unique(hit_coords_all, axis=0, return_counts=True)
        total_hit_points += int(cloud.shape[0])

        ray_xyz = xyz[:: args.ray_point_stride]
        total_ray_points += int(ray_xyz.shape[0])
        free_coords = ray_free_coords(
            origin,
            ray_xyz,
            args.voxel_size,
            args.ray_step_factor,
            args.endpoint_margin,
        )
        hit_coord_set = {(int(c[0]), int(c[1]), int(c[2])) for c in unique_hit_coords}
        free_coords.difference_update(hit_coord_set)

        for coord in free_coords:
            octree.update_miss(coord, args.miss_log_odds)

        for coord_np, count_np in zip(unique_hit_coords, counts):
            coord = (int(coord_np[0]), int(coord_np[1]), int(coord_np[2]))
            octree.update_hit(coord, int(count_np), args.hit_log_odds)

        if args.progress_interval and (
            done == 1 or done % args.progress_interval == 0 or done == len(frame_ids)
        ):
            print_progress(
                "integrate",
                done,
                len(frame_ids),
                started,
                f"octree_leaves={octree.leaf_count} hit_points={total_hit_points} ray_points={total_ray_points}",
            )

    occupied_coords = octree.occupied_coords(args.occupied_threshold)
    free_coords = octree.free_coords(args.free_threshold)
    unknown_coords = octree._leaves_by_coord.keys() - occupied_coords - free_coords
    if args.unknown_policy == "keep":
        kept_voxels = occupied_coords | unknown_coords
    else:
        kept_voxels = occupied_coords
    removed_voxels = free_coords if args.unknown_policy == "keep" else free_coords | unknown_coords
    free_leaf_count = len(free_coords)
    unknown_leaf_count = len(unknown_coords)
    kept_leaf_count = len(kept_voxels)
    removed_voxel_count = len(removed_voxels)

    after_payload = tmp_dir / "after.payload"
    removed_payload = tmp_dir / "removed.payload"
    before_payload = tmp_dir / "before.payload"
    after_count = 0
    removed_count = 0
    before_count = 0
    dynamic_frames_dir = out_dir / "dynamic_frames"
    if args.save_dynamic_frames:
        dynamic_frames_dir.mkdir(parents=True, exist_ok=True)

    second_started = time.time()
    with after_payload.open("wb") as after_f, removed_payload.open("wb") as removed_f:
        before_f = before_payload.open("wb") if args.write_before else None
        try:
            for done, frame_id in enumerate(frame_ids, 1):
                cloud, local_cloud = transform_scan(
                    scan_paths[frame_id],
                    poses[frame_id],
                    args.body_radius,
                    args.max_range,
                    args.local_z_min,
                    args.local_z_max,
                    False,
                )
                if cloud.size == 0:
                    if args.save_dynamic_frames:
                        write_pcd_from_points(
                            np.empty((0, 4), dtype=np.float32),
                            dynamic_frames_dir / f"{frame_id:06d}.pcd",
                        )
                    continue
                coords = voxel_coords_for_points(cloud[:, :3], args.voxel_size)
                if args.unknown_policy == "keep":
                    keep_mask = np.fromiter(
                        (
                            (coord := (int(c[0]), int(c[1]), int(c[2]))) in occupied_coords
                            or coord not in free_coords
                            for c in coords
                        ),
                        dtype=bool,
                        count=coords.shape[0],
                    )
                else:
                    keep_mask = np.fromiter(
                        (
                            (int(c[0]), int(c[1]), int(c[2])) in occupied_coords
                            for c in coords
                        ),
                        dtype=bool,
                        count=coords.shape[0],
                    )
                if args.ground_protect_local_z_max is not None:
                    keep_mask |= local_cloud[:, 2] <= args.ground_protect_local_z_max
                outside_roi = np.zeros(local_cloud.shape[0], dtype=bool)
                if args.local_z_min is not None:
                    outside_roi |= local_cloud[:, 2] < args.local_z_min
                if args.local_z_max is not None:
                    outside_roi |= local_cloud[:, 2] > args.local_z_max
                keep_mask |= outside_roi
                kept = cloud[keep_mask]
                removed = cloud[~keep_mask]
                if kept.size:
                    kept = kept.astype(np.float32, copy=False)
                    kept.tofile(after_f)
                    after_count += int(kept.shape[0])
                if removed.size:
                    removed = removed.astype(np.float32, copy=False)
                    removed.tofile(removed_f)
                    removed_count += int(removed.shape[0])
                if args.save_dynamic_frames:
                    write_pcd_from_points(removed, dynamic_frames_dir / f"{frame_id:06d}.pcd")
                if before_f is not None:
                    cloud.astype(np.float32, copy=False).tofile(before_f)
                    before_count += int(cloud.shape[0])
                if args.progress_interval and (
                    done == 1 or done % args.progress_interval == 0 or done == len(frame_ids)
                ):
                    print_progress(
                        "write",
                        done,
                        len(frame_ids),
                        second_started,
                        f"after={after_count} removed={removed_count}",
                    )
        finally:
            if before_f is not None:
                before_f.close()

    write_pcd_from_payload(after_payload, after_count, out_dir / "raycast_after.pcd")
    write_pcd_from_payload(removed_payload, removed_count, out_dir / "raycast_removed.pcd")
    if args.write_before:
        write_pcd_from_payload(before_payload, before_count, out_dir / "raycast_before.pcd")

    summary = [
        f"dataset: {args.dataset}",
        f"seq: {args.seq}",
        f"pose: {pose_path}",
        f"frames: {args.start}..{end}",
        f"stride: {args.stride}",
        f"voxel_size: {args.voxel_size}",
        f"max_range: {args.max_range}",
        f"body_radius: {args.body_radius}",
        f"local_z_min: {args.local_z_min}",
        f"local_z_max: {args.local_z_max}",
        f"ground_protect_local_z_max: {args.ground_protect_local_z_max}",
        f"ray_point_stride: {args.ray_point_stride}",
        f"ray_step_factor: {args.ray_step_factor}",
        f"endpoint_margin: {args.endpoint_margin}",
        f"hit_log_odds: {args.hit_log_odds}",
        f"miss_log_odds: {args.miss_log_odds}",
        f"classification_model: log_odds_only",
        f"occupied_threshold: {args.occupied_threshold}",
        f"free_threshold: {args.free_threshold}",
        f"unknown_policy: {args.unknown_policy}",
        "map_structure: octree",
        f"octree_root_origin: {octree.root.origin if octree.root is not None else None}",
        f"octree_root_size_voxels: {octree.root.size if octree.root is not None else 0}",
        f"octree_leaf_voxels: {octree.leaf_count}",
        f"kept_voxels: {kept_leaf_count}",
        f"occupied_voxels: {len(occupied_coords)}",
        f"free_voxels: {free_leaf_count}",
        f"removed_voxels: {removed_voxel_count}",
        f"unknown_voxels: {unknown_leaf_count}",
        f"input_points: {total_hit_points}",
        f"raycast_sample_points: {total_ray_points}",
        f"after_points: {after_count}",
        f"removed_points: {removed_count}",
        f"elapsed_sec: {time.time() - started:.3f}",
    ]
    (out_dir / "raycast_summary.txt").write_text("\n".join(summary) + "\n")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[done] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
