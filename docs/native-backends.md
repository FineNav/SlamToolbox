# Native compute backends

`slam_toolbox` keeps user interaction, dataset loading, progress reporting, and PCD output in Python.
Compute-heavy algorithm state and point classification can be implemented in C++ and exposed through
the shared `slam_toolbox._native` pybind11 module.

## Build

The native module uses CMake, scikit-build-core, pybind11, and C++17:

```bash
python -m pip install -e . --no-deps
```

The Local Hash Voxel command supports three backend modes:

- `auto`: use C++ when the extension is installed, otherwise use Python.
- `native`: require C++ and fail clearly if the extension is unavailable.
- `python`: force the reference implementation for validation and benchmarking.

The selected backend and native API version are written to `point_count_summary.txt`.

## Extension layout

The compiled module is split into algorithm submodules. Each algorithm owns its configuration,
state, and bindings while sharing the build and common C++ utilities:

```text
cpp/include/slam_toolbox/common/
cpp/include/slam_toolbox/<algorithm>/
cpp/src/<algorithm>/
cpp/src/bindings/<algorithm>_bindings.cpp
slam_toolbox/native/<algorithm>.py
```

New algorithms should use frame-oriented NumPy interfaces and release the Python GIL during heavy
work. File paths and output formats should remain outside the C++ API so the core can be reused with
other datasets and frontends.

## Compatibility contract

Native implementations must be tested against the Python reference using the same scans, poses, and
parameters. Required checks are:

- voxel and point classification counts are identical;
- keep/remove masks are identical;
- output point order and intensity are identical;
- transformed float32 coordinates agree within `1e-5 m`;
- Python fallback still runs when the extension is unavailable.
