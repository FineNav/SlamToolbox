import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from slam_toolbox.dynamic_removal import (
    _keep_only_standard_run_pcds,
    _parse_resource_pair,
    _parse_resource_size,
    _run_measured_command,
)
from slam_toolbox.dynamic_removal_local_voxel import (
    _load_algorithm_parameter_config,
    _save_algorithm_parameter_config,
)


class ResourceUsageTest(unittest.TestCase):
    def test_parse_resource_sizes(self):
        self.assertEqual(_parse_resource_size("512MiB"), 512 * 1024 * 1024)
        self.assertEqual(_parse_resource_size("1.5GB"), 1_500_000_000)
        self.assertEqual(
            _parse_resource_pair("12.5MB / 2GiB"),
            (12_500_000, 2 * 1024 * 1024 * 1024),
        )

    def test_local_process_writes_report(self):
        with tempfile.TemporaryDirectory() as output_dir:
            code = (
                "import time; "
                "data=bytearray(16*1024*1024); "
                "end=time.time()+1.2; "
                "\nwhile time.time()<end: sum(range(10000))"
            )
            returncode = _run_measured_command(
                [sys.executable, "-c", code],
                output_dir,
                "test_method",
            )
            self.assertEqual(returncode, 0)
            report_path = Path(output_dir) / "resource_usage.yaml"
            self.assertTrue(report_path.exists())
            report = yaml.safe_load(report_path.read_text())
            self.assertEqual(report["method"], "test_method")
            self.assertEqual(report["exit_code"], 0)
            self.assertGreater(report["wall_time_seconds"], 1.0)
            self.assertGreater(report["effective_cpu_seconds"], 0.0)
            self.assertGreater(report["effective_peak_memory_bytes"], 8 * 1024 * 1024)
            self.assertIsNone(report["docker_container"])

    def test_cleanup_keeps_only_standard_pcd_names(self):
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            for name in (
                "before.pcd",
                "static.pcd",
                "dynamic.pcd",
                "00_original.pcd",
                "00_estimated.pcd",
                "00_voxelized.pcd",
            ):
                (output_path / name).write_bytes(b"pcd")
            (output_path / "resource_usage.yaml").write_text("method: test\n")

            _keep_only_standard_run_pcds(output_dir)

            self.assertEqual(
                sorted(path.name for path in output_path.glob("*.pcd")),
                ["before.pcd", "dynamic.pcd", "static.pcd"],
            )
            self.assertTrue((output_path / "resource_usage.yaml").exists())

    def test_algorithm_parameter_config_round_trips(self):
        specs = [
            {"key": "start", "kind": "int", "min": 0},
            {"key": "end", "kind": "int", "min": 0, "optional": True},
            {"key": "local_z_min", "kind": "float"},
            {"key": "local_z_max", "kind": "float"},
            {"key": "write_before", "kind": "bool"},
            {"key": "unknown_policy", "kind": "choice", "choices": ["keep", "drop"]},
            {"key": "backend", "kind": "choice", "choices": ["auto", "native", "python"]},
        ]
        defaults = {
            "start": 0,
            "end": None,
            "local_z_min": 0.0,
            "local_z_max": 3.0,
            "write_before": True,
            "unknown_policy": "keep",
            "backend": "auto",
        }
        saved = {
            "start": 5,
            "end": 20,
            "local_z_min": -1.5,
            "local_z_max": 2.0,
            "write_before": False,
            "unknown_policy": "drop",
            "backend": "native",
        }

        with tempfile.TemporaryDirectory() as map_path:
            _save_algorithm_parameter_config(map_path, "raycast_voxel", saved)
            loaded = _load_algorithm_parameter_config(
                map_path,
                "raycast_voxel",
                specs,
                defaults,
            )

        self.assertEqual(loaded, saved)


if __name__ == "__main__":
    unittest.main()
