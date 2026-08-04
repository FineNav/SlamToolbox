from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from slam_toolbox.algorithms.local_hash_voxel_filter import (
    build_native_model,
    build_python_model,
    pack_keys,
)
from slam_toolbox.native import native_available


@unittest.skipUnless(native_available(), "native extension is not built")
class LocalHashVoxelNativeTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            start=0,
            voxel_size=0.5,
            max_range=20.0,
            local_z_min=-1.0,
            local_z_max=2.5,
            body_radius=0.1,
            ground_protect_local_z_max=-0.5,
            lidar_hz=10.0,
            ray_stride=2,
            max_ray_endpoints=1000,
            min_visible_frames=2,
            min_visible_time=0.2,
            static_min_hit_ratio=0.5,
            dynamic_max_hit_ratio=0.2,
            dynamic_max_hit_time=0.3,
            unknown_policy="keep",
            progress_interval=0,
        )

    def test_native_matches_python_reference(self):
        scans = [
            np.array(
                [
                    [1.1, 0.1, 0.2, 10.0],
                    [1.2, 0.1, 0.2, 11.0],
                    [2.2, 0.7, 0.3, 12.0],
                    [3.1, -0.8, 1.2, 13.0],
                    [0.01, 0.01, 0.0, 14.0],
                    [25.0, 0.0, 0.0, 15.0],
                    [1.0, 0.0, 3.0, 16.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [1.05, 0.12, 0.2, 20.0],
                    [2.15, 0.72, 0.3, 21.0],
                    [3.0, -0.75, 1.2, 22.0],
                    [4.2, 1.1, 0.5, 23.0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [1.0, 0.15, 0.2, 30.0],
                    [2.1, 0.75, 0.3, 31.0],
                    [4.1, 1.15, 0.5, 32.0],
                    [5.3, -1.2, -0.7, 33.0],
                ],
                dtype=np.float32,
            ),
        ]
        poses = []
        for index in range(len(scans)):
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = np.float32(index * 0.1)
            poses.append(pose)
        times = np.arange(len(scans), dtype=np.float64) / self.args.lidar_hz

        with tempfile.TemporaryDirectory() as directory:
            scan_paths = []
            for index, scan in enumerate(scans):
                path = Path(directory) / f"{index:06d}.bin"
                scan.tofile(path)
                scan_paths.append(path)

            python_keys, python_stats = build_python_model(
                self.args, scan_paths, poses, times, len(scans) - 1
            )
            native_engine, native_stats = build_native_model(
                self.args, scan_paths, poses, times, len(scans) - 1
            )

        self.assertEqual(native_stats, python_stats)

        for scan, pose in zip(scans, poses):
            native_points, native_mask = native_engine.classify_frame(scan, pose)
            valid = np.isfinite(scan).all(axis=1)
            distance_squared = (
                scan[:, 0].astype(np.float64) ** 2
                + scan[:, 1].astype(np.float64) ** 2
            )
            valid &= distance_squared >= self.args.body_radius**2
            valid &= distance_squared <= self.args.max_range**2
            filtered = scan[valid]
            xyz = filtered[:, :3] @ pose[:3, :3].T + pose[:3, 3]
            keys = pack_keys(np.floor(xyz / self.args.voxel_size).astype(np.int64))
            classified_static = np.fromiter(
                (int(key) in python_keys for key in keys),
                dtype=bool,
                count=keys.shape[0],
            )
            outside_roi = (
                (filtered[:, 2] < self.args.local_z_min)
                | (filtered[:, 2] > self.args.local_z_max)
            )
            ground_protected = filtered[:, 2] <= self.args.ground_protect_local_z_max
            expected_mask = outside_roi | ground_protected | classified_static
            expected_points = np.column_stack((xyz, filtered[:, 3])).astype(np.float32)

            np.testing.assert_allclose(native_points, expected_points, rtol=0.0, atol=1e-6)
            np.testing.assert_array_equal(native_mask, expected_mask)


if __name__ == "__main__":
    unittest.main()
