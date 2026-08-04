# Dynamic Removal Branch Notes

This document records the changes prepared for the dynamic-obstacle-removal
branch.

## Summary

The branch changes the dynamic removal workflow to avoid using accumulated
`frame/` point clouds as algorithm input. ERASOR2, Removert, Local Hash Voxel,
and Raycast Voxel use KITTI-style per-scan point clouds generated from the
source bag, while their poses come from the time-interpolated Interactive SLAM
correction.

The main goal is to keep each scan in the correct local sensor/base frame while
using the Interactive SLAM-corrected trajectory in `poses_odom_base.txt`.

## Data Conversion

- `frame/` is no longer used as the default source for KITTI conversion.
- KITTI data is generated directly from the source bag point cloud topic,
  usually `/cloud_registered`.
- If the source point cloud is already in an odom/global frame, conversion now
  transforms points through:

```text
source_cloud_frame -> fixed_frame/odom -> base_link
```

- The written `velodyne/*.bin` files are local `base_link` scans.
- `poses_odom_base.txt` stores the Interactive SLAM-corrected `odom -> base_link`
  trajectory, interpolated onto every bag scan.
- `poses_suma_optim.txt` remains the ERASOR2-compatible compensated trajectory.
- `conversion_notes.txt` now records `point_transform` so newer converted
  datasets can be distinguished from old or frame-based datasets.
- `frame/timestamps.txt` records the reference time of each accumulated frame,
  allowing its correction to be interpolated onto the original bag scans.
- Old KITTI datasets without the new bag-local conversion marker are discarded
  and regenerated from bag.

This fixes the previous Raycast Voxel failure mode where `/cloud_registered`
was treated as local scan data even though it was already in an odom/global
frame. Raycasting then applied `poses_odom_base.txt` again, which caused
double-transforming and scattered output.

## ERASOR2

- ERASOR2 conversion now uses bag-derived per-scan KITTI data instead of
  accumulated `frame/` PCD files.
- This avoids generating very large accumulated KITTI frames that can exceed
  ERASOR2 mapgen's fixed loader buffer.
- The earlier point/label mismatch issue is documented separately in
  `docs/issues/erasor2-large-frame-load-limit.md`.

## Removert

- Removert no longer requires `frame/` to exist before running.
- It uses the same bag-derived KITTI dataset path as the other dynamic removal
  methods.
- Existing old/frame-based KITTI data is regenerated from bag before use.
- Runtime output is now shown as stage-level progress bars:

```text
读取点云
构建地图
动态清除
写出结果
```

- The full Docker output is still written to `removert_docker.log`.

## Local Hash Voxel

- Added Local Hash Voxel dynamic removal as a selectable method.
- The method reads local KITTI scans and `poses_odom_base.txt`.
- It rejects datasets that only have identity/no-motion poses, because the
  method needs real sensor origins.
- This means YunJingFull-like datasets without real sensor trajectory are
  intentionally rejected with an explanatory error.

## Raycast Voxel

- Added Raycast Voxel Cleanup as a selectable dynamic removal method.
- The method reads local KITTI scans and `poses_odom_base.txt`.
- It uses raycasting from real sensor poses to classify occupied/free voxels.
- Outputs are copied to the map directory as:

```text
map_raycast_voxel_static.pcd
map_raycast_voxel_removed.pcd
```

- For the `BLGX_3.25` map, the KITTI dataset was regenerated after fixing the
  bag-to-local conversion. The old raycast output in `map/` must be regenerated
  by running Raycast Voxel again.

### Raycast Performance Optimization

The Python Raycast Voxel implementation was profiled on the first 200 frames of
`YunJing_521`. The main bottlenecks were per-ray `numpy.unique` calls, repeated
octree traversal, and point-by-point occupied-voxel lookup during the second
pass.

The implementation now:

- removes only adjacent duplicate voxels along each straight ray, which is
  equivalent to a full per-ray unique operation because a ray cannot leave and
  later re-enter the same convex voxel;
- keeps a direct coordinate-to-leaf index alongside the octree;
- builds the final occupied-coordinate set once and uses direct membership
  checks during output filtering.

The profiled 200-frame run changed from approximately `28.83 s` to `12.05 s`.
An unprofiled optimized run completed in approximately `7.68 s`. The generated
`raycast_after.pcd` and `raycast_removed.pcd` files were byte-for-byte identical
to the baseline, and all point and voxel counts matched.

## Sensor Timestamp and Pose Handling

- Some bags, including `YunJing_521`, contain valid `/tf` messages and a moving
  `odom -> base_link` trajectory, while `PointCloud2.header.stamp` is always
  zero.
- KITTI conversion now falls back to the rosbag message timestamp when the
  point-cloud header timestamp is zero.
- This prevents every frame from resolving to the same sensor pose and fixes
  the previous error:

```text
Raycast Voxel 无法运行: 检测到传感器轨迹几乎全为同一位姿
```

- After reconverting `YunJing_521`, `poses_odom_base.txt` contains 2220 poses
  with approximately 74.94 m maximum translation from the first pose.

## Pose-Correction Safety and ERASOR2 Input Limits

- Planar constraint and Interactive SLAM interpolation now create timestamped
  odometry snapshots under `pose_correction/backups/` before modifying
  `frame/*.odom`.
- Interactive SLAM correction deltas prefer the `odom` matrix saved alongside
  each optimized `estimate`, avoiding accidental replay against a changed
  frame trajectory.
- Dense correction interpolation uses `frame/timestamps.txt` when it is
  complete, and falls back to frame IDs only for legacy datasets.
- The latest correction summary and saved-odom mismatch diagnostics are
  written to `pose_correction/latest_report.yaml`.
- ERASOR2 now checks every KITTI scan against its 500,000-point loader limit.
  Oversized scans are voxel-limited to at most 450,000 points in the separate
  `erasor2_dataset_limited/` dataset, with exactly matching label counts. The
  shared KITTI dataset used by Removert and the Python filters is not changed.

## Dynamic-Obstacle GIF Workflow

All four dynamic-removal methods now offer the same post-processing prompt:

```text
是否生成动态障碍物轨迹 GIF？
```

Selecting no keeps all generated point-cloud results and ends the current
dynamic-removal workflow. Selecting yes opens a playback-speed selector:

```text
极速: about 67 frames/s (0.015 s/frame)
快速: 40 frames/s (0.025 s/frame, default)
正常: 20 frames/s (0.05 s/frame)
慢速: 10 frames/s (0.1 s/frame)
自定义: user-provided seconds/frame
```

Each dynamic-removal run also writes `resource_usage.yaml` in its timestamped
run directory. The report records the core algorithm process wall time, user
and system CPU time, peak resident memory, disk I/O, exit code, and host
metadata. ERASOR2 and Removert additionally record Docker container CPU,
peak memory, and block I/O. The report scope is the algorithm process and its
children, excluding dataset conversion and post-processing work.

Per-method dynamic-frame sources are:

- Local Hash Voxel: writes removed points directly to `dynamic_frames/` while
  processing each scan.
- Raycast Voxel: writes removed points directly to `dynamic_frames/` while
  processing each scan.
- Removert: reconstructs per-frame dynamic points by matching the aggregated
  `removert_dynamic.pcd` against the original KITTI scans transformed by
  `poses_odom_base.txt`.
- ERASOR2: derives removed voxels from the difference between the original and
  estimated static maps, then matches those voxels back to transformed KITTI
  scans.

The generated GIF uses a 2D bird's-eye view. Removed objects leave a persistent
orange trajectory, while points from the current displayed interval are red.
The GIF is written to the timestamped run directory and copied to the map-level
visualization directory:

```text
<map>/runs/<method>/<timestamp>/visualize/<method>_dynamic_overlay.gif
<map>/visualize/<method>_dynamic_overlay.gif
```

## GIF Rendering Optimization

The original renderer recreated a Matplotlib figure for every source frame and
could repeatedly draw up to 500,000 static points and 1,000,000 accumulated
trajectory points. On long sequences this appeared to freeze during
`渲染 GIF 帧`.

Automatic 2D GIF generation now uses a lightweight raster renderer:

- the static map is rasterized once instead of being redrawn for every frame;
- the trajectory is accumulated directly in an image buffer;
- source frames are grouped into at most 240 observation frames while every
  source frame still contributes to the persistent trajectory;
- static-map visualization is sampled to at most 220,000 points;
- each source dynamic frame is sampled to at most 20,000 points;
- output is `720 x 720`, rendered internally at `1440 x 1440`, then reduced
  with Lanczos antialiasing for smoother points and edges.

The visualization limits are defaults in
`slam_toolbox/visualization.py:create_dynamic_overlay_gif`:

```text
max_static_points = 220000
max_dynamic_points = 20000
max_frames = 240
frame_width = 720
frame_height = 720
render_scale = 2
```

On the complete 2220-frame `YunJing_521` Raycast result, the renderer produced
222 GIF frames in approximately `15.9 s`; the resulting GIF was approximately
15 MB. These limits affect visualization only and do not change the dynamic
removal result or saved PCD files.

The previous Matplotlib renderer remains available for the optional 3D view,
but automatic post-processing uses the faster 2D path.

## Rich CLI Output

Dynamic-removal status messages previously used Rich markup such as
`[bold green]...[/bold green]` with Python's plain `print`, causing the markup
names to appear literally in the terminal. ERASOR2, Removert, Local Hash
Voxel, Raycast Voxel, and GIF-related colored messages now use
`rich.console.Console.print`, so success, warning, error, and dim status styles
are rendered as actual terminal colors.

## Timestamped Outputs

Dynamic removal and map-building outputs are now stored in timestamped run
directories:

```text
<map>/runs/erasor2/YYYYmmdd_HHMMSS/
<map>/runs/removert/YYYYmmdd_HHMMSS/
<map>/runs/local_hash_voxel/YYYYmmdd_HHMMSS/
<map>/runs/raycast_voxel/YYYYmmdd_HHMMSS/
<map>/runs/map_builder/YYYYmmdd_HHMMSS/
```

The latest user-facing results are still copied into:

```text
<map>/map/
```

## Interactive SLAM Docker GUI

- The Docker GUI launch was updated for X11 access.
- The launcher grants local root X11 access with `xhost`.
- `DISPLAY` and `XAUTHORITY` are passed into the container.
- The host `~/Map` directory is mounted at the same absolute path inside the
  container (`/home/<user>/Map`) and is also available through `/Map`; the
  currently selected map remains available through the `/root/Map` shortcut.

## CLI Flow

- The map action menu now includes `更换地图`.
- Selecting it returns to the previous map-selection step instead of exiting the
  whole tool.

## Verification Performed

- `python3 -m py_compile slam_toolbox/dynamic_removal.py`
- Simulated Removert progress-log parsing.
- Regenerated `/home/timory/Map/BLGX_3.25/erasor2_dataset` from bag:
  4180 scans, 4180 poses, 4180 timestamps.
- Spot-checked regenerated `BLGX_3.25` scans and confirmed that sample frames
  are now in local coordinates rather than large odom/global coordinates.
- Profiled Raycast Voxel before and after optimization on 200 `YunJing_521`
  frames and compared both output PCD files byte-for-byte.
- Verified the GIF yes/no prompt, preset speed, and invalid custom-speed
  fallback branches.
- Verified aggregate-map-to-frame reconstruction with a synthetic two-frame
  KITTI dataset and moving poses.
- Rendered the complete 2220-frame `YunJing_521` Raycast dynamic sequence with
  the optimized 2D renderer.
- Scanned the Python package for Rich markup still sent through plain `print`
  and verified rendered output no longer contains literal markup tags.

## Push Notes

Before pushing, rerun the affected method on any map whose previous output was
generated from the old conversion logic. In particular, rerun Raycast Voxel for
`BLGX_3.25` because the existing `map_raycast_voxel_static.pcd` and
`map_raycast_voxel_removed.pcd` were produced before the conversion fix.
