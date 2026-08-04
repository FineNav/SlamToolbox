# Local Hash Voxel native validation

Validation date: 2026-08-04

Dataset: `/home/timory/Map/WeiNeng_6.23/erasor2_dataset`, frames `0..1469`.
Both runs used the same algorithm parameters and disabled `write-before` and per-frame dynamic PCDs
to focus the comparison on the core algorithm and standard output files.

| Backend | Wall time | User CPU | System CPU | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Python | 13.60 s | 12.92 s | 2.18 s | 60,956 KiB |
| C++/pybind11 | 1.03 s | 1.51 s | 1.96 s | 57,736 KiB |

Observed wall-time speedup: **13.2x**.

All model statistics and output counts were identical, including `16,118` statistics voxels,
`2,486` static voxels, `433` dynamic voxels, `10,477,472` static points, and `6,374` dynamic points.
Output order and intensity were identical. The largest transformed-coordinate difference was
`3.81e-6 m`, below the `1e-5 m` float32 compatibility tolerance.
