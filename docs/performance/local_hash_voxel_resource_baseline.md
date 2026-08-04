# Local Hash Voxel Latest Resource Baseline

Generated from the latest `/home/timory/Map/<map>/runs/local_hash_voxel/<run>` per map, excluding `YunJing_Full` / `YunJingfull`.

- Generated at: 2026-08-04T11:26:39+08:00
- Maps included: 7
- Latest runs with resource_usage.yaml: 7
- CSV: `docs/performance/local_hash_voxel_latest_resource_baseline.csv`

## Overall Resource Summary

| metric | value |
|---|---:|
| wall time min / median / max | 13.994s / 45.451s / 241.055s |
| CPU time min / median / max | 15.070s / 46.700s / 236.940s |
| peak memory min / median / max | 44.4 / 64.0 / 188.6 MiB |

## Latest Run Per Map

| map | run | frames | wall s | CPU s | peak MiB | read MiB | write MiB | raw pts | ROI pts | traced rays | visible updates | stats voxels | dynamic pts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Airport_1F | 20260804_110516 | 0..2003 | 58.918 | 59.860 | 63.6 | 272.4 | 510.4 | 17,535,844 | 11,461,777 | 466,455 | 6,593,841 | 88,204 | 89,841 |
| Airport_2F_N | 20260804_104048 | 0..1095 | 19.784 | 22.200 | 66.2 | 0.0 | 291.6 | 8,893,362 | 3,808,628 | 162,488 | 2,267,053 | 59,574 | 65,739 |
| Airport_2F_S | 20260804_110256 | 0..1232 | 21.957 | 23.530 | 53.6 | 148.9 | 273.0 | 9,550,495 | 3,543,967 | 169,026 | 2,456,850 | 56,233 | 42,601 |
| Airport_3F | 20260804_105130 | 0..11670 | 241.055 | 236.940 | 188.6 | 1556.3 | 5465.1 | 100,122,028 | 36,479,804 | 1,751,058 | 27,767,053 | 499,701 | 535,172 |
| BLGX_3.25 | 20260804_104522 | 0..4179 | 73.766 | 74.140 | 148.0 | 289.9 | 966.9 | 18,324,391 | 5,471,735 | 539,996 | 9,273,027 | 376,197 | 80,387 |
| WeiNeng_6.23 | 20260804_104914 | 0..1469 | 13.994 | 15.070 | 44.4 | 168.3 | 294.2 | 10,779,316 | 6,366,482 | 125,716 | 1,149,194 | 16,118 | 6,374 |
| YunJing_521 | 20260804_110043 | 0..2219 | 45.451 | 46.700 | 64.0 | 280.6 | 1064.7 | 17,453,804 | 8,100,470 | 368,066 | 4,507,596 | 89,308 | 16,654 |

## Parameters

| map | run | voxel | range | z min/max | ray stride | max endpoints | min visible frames/time | static/dynamic hit ratio | dynamic hit time | unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Airport_1F | 20260804_110516 | 0.500 | 30.000 | 0.000/3.000 | 4 | 25,000 | 10/1.000 | 0.800/0.600 | 2.000 | keep |
| Airport_2F_N | 20260804_104048 | 0.500 | 30.000 | 0.000/3.000 | 4 | 25,000 | 10/1.000 | 0.800/0.600 | 2.000 | keep |
| Airport_2F_S | 20260804_110256 | 0.500 | 30.000 | 0.000/3.000 | 4 | 25,000 | 10/1.000 | 0.800/0.600 | 2.000 | keep |
| Airport_3F | 20260804_105130 | 0.500 | 30.000 | 0.000/3.000 | 4 | 25,000 | 10/1.000 | 0.800/0.600 | 2.000 | keep |
| BLGX_3.25 | 20260804_104522 | 0.500 | 30.000 | 0.000/3.000 | 4 | 25,000 | 10/1.000 | 0.800/0.600 | 2.000 | keep |
| WeiNeng_6.23 | 20260804_104914 | 0.500 | 30.000 | 0.000/3.000 | 4 | 25,000 | 10/1.000 | 0.800/0.600 | 2.000 | keep |
| YunJing_521 | 20260804_110043 | 0.500 | 30.000 | 0.000/3.000 | 4 | 25,000 | 10/1.000 | 0.800/0.600 | 2.000 | keep |

## Notes For C++ Comparison

- Compare Python vs C++ using these latest-run parameters and the same input KITTI dataset per map.
- `wall_time_seconds`, `effective_cpu_seconds`, and `effective_peak_memory_bytes` come from `resource_usage.yaml`.
- Point and ray statistics come from `point_count_summary.txt`.
