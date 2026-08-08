# SlamToolbox OctoMap

面向 ROS2 激光点云地图的动态点清理和 OctoMap 语义层导出工具。项目包含
Python 工作流、C++17/pybind11 native 后端，以及标准 OctoMap 文件导出。

Raycast Voxel 在同一次逐帧积分中生成：

- 动态清理结果：`before.pcd`、`static.pcd`、`dynamic.pcd`；
- 标准 OctoMap：`map.bt`、`map.ot`；
- 语义 voxel：occupied、free、unknown、traversable、risk；
- 风险等级、机器人尺寸、阈值和统计信息。

当前 native API 版本为 **5**。OctoMap 导出使用独立的完整射线 occupancy
模型和精确 3D voxel traversal；动态清理继续使用 endpoint evidence 模型，
因此开启 OctoMap 导出不会改变 static/dynamic PCD 分类结果。

## 1. 支持环境

推荐环境：

- Ubuntu 22.04 或 24.04；
- Python 3.10–3.12；
- CMake 3.18 或更高版本；
- 支持 C++17 的编译器；
- OctoMap development package；
- ROS2（仅交互式完整工作流需要；直接处理已转换 KITTI 数据不需要 ROS2）。

安装系统依赖：

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  liboctomap-dev \
  python3-dev \
  python3-venv
```

如果需要查看 `.bt/.ot`，可以另外安装 OctoMap Viewer：

```bash
sudo apt install -y octovis
```

## 2. 安装项目

所有 Python 包和 native 扩展都建议安装在项目自己的虚拟环境中：

```bash
git clone https://github.com/FineNav/SlamToolbox
cd SlamToolbox

python3 -m venv venv-native
venv-native/bin/python -m pip install --upgrade pip
venv-native/bin/python -m pip install -e .
```

不要使用系统 Python 直接执行 `pip install`。Ubuntu 的 externally-managed
Python 可能报 PEP 668 错误，虚拟环境可以避免这个问题。

验证导入路径和 native API：

```bash
venv-native/bin/python - <<'PY'
import sys
import slam_toolbox
from slam_toolbox import _native

print("python:", sys.executable)
print("package:", slam_toolbox.__file__)
print("native api:", _native.api_version)
print("raycast backend:", hasattr(_native, "raycast_voxel"))
print("octomap export:", hasattr(_native.raycast_voxel.Engine, "write_octomap"))
PY
```

正确安装时应看到：

```text
native api: 5
raycast backend: True
octomap export: True
```

## 3. 最简单的运行方式：直接处理 KITTI 格式数据

这种方式不启动 ROS2 菜单，适合已经准备好逐帧局部点云和真实传感器位姿的用户。

### 3.1 输入目录契约

```text
<kitti-root>/
└── dataset/
    └── sequences/
        └── 00/
            ├── velodyne/
            │   ├── 000000.bin
            │   ├── 000001.bin
            │   └── ...
            └── poses_odom_base.txt
```

输入要求：

- 每个 `.bin` 是连续的 little-endian `float32`，每点四个值：`x y z intensity`；
- 点必须位于对应帧的局部传感器或 `base_link` 坐标系，不能是已经注册好的全局点；
- `poses_odom_base.txt` 每行是一帧位姿，可使用 12 个数的 3×4 矩阵，
  或 16 个数的 4×4 矩阵；
- 位姿必须把该帧局部点变换到全局 `odom/map` 坐标系；
- 位姿数量必须覆盖点云帧数，不能使用全 identity 位姿。

### 3.2 运行 Raycast、OctoMap 和语义层导出

```bash
venv-native/bin/python -m slam_toolbox.algorithms.raycast_voxel_cleanup \
  --dataset /absolute/path/to/kitti-root \
  --seq 00 \
  --out /absolute/path/to/output-run \
  --backend native \
  --export-octomap \
  --write-before \
  --voxel-size 0.5 \
  --ray-point-stride 1 \
  --robot-radius 0.35 \
  --robot-height 1.2 \
  --robot-ground-clearance 0.05 \
  --robot-max-step-height 0.12 \
  --robot-safety-margin 0.20 \
  --robot-ground-z 0.0
```

参数说明：

- `--voxel-size`：OctoMap 叶节点尺寸，单位为米；
- `--ray-point-stride 1`：每个有效点都发射 free-space 射线；增大该值可降低
  计算量和内存，但会降低 free-space 密度；
- `--robot-ground-z`：全局坐标系中的假设地面高度，必须按实际地图调整；
- `--robot-radius/height`：机器人圆形 footprint 和机身高度；
- `--robot-safety-margin`：机器人半径之外的风险膨胀距离；
- `--risk-levels`：风险等级数量，默认 3；
- `--compressed-octomap`：压缩 `layers.npz`，文件更小但导出更慢；
- `--start/--end/--stride`：限制处理帧范围，便于先做小规模验证。

查看全部参数：

```bash
venv-native/bin/python -m slam_toolbox.algorithms.raycast_voxel_cleanup --help
```

## 4. 输出文件

指定 `--out /path/to/output-run` 后，主要输出为：

```text
output-run/
├── raycast_before.pcd       # 使用 --write-before 时生成
├── raycast_after.pcd        # 保留的静态点
├── raycast_removed.pcd      # 被移除的动态点
├── raycast_summary.txt
└── octomap/
    ├── map.bt               # 标准 maximum-likelihood、pruned OcTree
    ├── map.ot               # 标准 full OcTree，保留 log-odds
    ├── layers.npz           # 项目语义 voxel 层
    └── meta.yaml            # 坐标系、阈值、机器人参数和数量统计
```

`layers.npz` 包含：

- `occupied`：达到 occupied threshold 的 voxel；
- `free`：达到 free threshold 的 voxel；
- `unknown`：已观察但置信度尚未达到 occupied/free threshold 的 voxel；
- `kept`、`removed`：动态清理 endpoint 模型的分类；
- `traversable`：地面高度带内且通过 footprint/垂直净空检查的候选可通行 voxel；
- `risk`、`risk_intensity`：靠近障碍物的候选风险 voxel 和风险等级；
- `voxel_coords`、`voxel_type`：组合语义视图。

`voxel_type` 编号记录在 `meta.yaml`，当前为：

| ID | 类型 |
|---:|---|
| 0 | unknown |
| 1 | free |
| 2 | occupied |
| 3 | traversable |
| 4 | risk |

读取语义层：

```bash
venv-native/bin/python - /absolute/path/to/output-run/octomap <<'PY'
from pathlib import Path
import sys
import numpy as np
import yaml

root = Path(sys.argv[1])
layers = np.load(root / "layers.npz")
meta = yaml.safe_load((root / "meta.yaml").read_text(encoding="utf-8"))

print(meta["counts"])
print("free:", layers["free"].shape)
print("traversable:", layers["traversable"].shape)
print("risk:", layers["risk"].shape)
PY
```

使用 OctoVis 查看标准 OctoMap：

```bash
octovis /absolute/path/to/output-run/octomap/map.bt
```

## 5. ROS2 交互式完整工作流

交互式入口支持 rosbag、帧提取、位姿修正、ERASOR2、Removert、Local Hash
Voxel、Raycast Voxel 和地图构建。该入口会扫描 `~/Map` 下的地图目录。

先加载 ROS2 环境，再启动：

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd /absolute/path/to/SlamToolbox
venv-native/bin/slam_toolbox
```

典型地图目录：

```text
~/Map/<map-name>/
├── bag/
├── map/
└── config.yaml
```

`config.yaml` 的最小内容：

```yaml
config:
  fixed_frame: odom
  base_link_frame: base_link
  pointcloud_topic: /cloud_registered
```

在菜单中选择：

```text
3D Map -> Raycast Voxel
```

交互式 Raycast 默认开启 OctoMap 导出，并将结果写入：

```text
~/Map/<map-name>/runs/raycast_voxel/<timestamp>/
```

第一次运行会显示参数，确认后把参数保存在地图目录中；后续运行会复用并允许修改。
bag 转 KITTI 和轨迹插值依赖 ROS2 Python 包以及有效的 corrected trajectory。

## 6. 安装验证

本发布目录不包含开发测试集。安装完成后，可以用下面的命令检查 native API
和 Raycast 命令行入口：

```bash
venv-native/bin/python -c \
  "from slam_toolbox import _native; print(_native.api_version)"

venv-native/bin/python -m \
  slam_toolbox.algorithms.raycast_voxel_cleanup --help
```

第一条命令应输出 `5`，第二条命令应正常显示 Raycast 和 OctoMap 参数。
发布前的开发工作树已通过 native、完整 free-space、动态分类不变性和资源报告测试。

## 7. 常见问题

### CMake 找不到 OctoMap

如果出现：

```text
Could not find a package configuration file provided by "octomap"
```

确认安装 development package：

```bash
sudo apt install -y liboctomap-dev
dpkg -L liboctomap-dev | grep octomap-config.cmake
```

然后重新执行：

```bash
venv-native/bin/python -m pip install -e . --no-deps
```

### API 版本不是 5

通常表示加载了其他 checkout 或旧的 native 扩展。检查：

```bash
venv-native/bin/python - <<'PY'
import sys
import slam_toolbox
from slam_toolbox import _native
print(sys.executable)
print(slam_toolbox.__file__)
print(_native.__file__)
print(_native.api_version)
PY
```

确保所有路径都指向当前项目目录，然后重新执行 editable install。

### `traversable` 为空或范围不正确

优先检查 `robot_ground_z`、`voxel_size`、局部 Z ROI 和 free voxel 数量。
`robot_ground_z` 是全局地图坐标，不一定等于传感器局部坐标中的地面高度。

## 8. 当前导航语义边界

完整射线 free-space 已实现，但 `traversable` 仍是候选导航层。目前检查的是：

- 固定地面高度带；
- 圆形机器人 footprint；
- 机器人高度范围内的障碍物净空；
- 障碍物附近的风险距离。

目前尚未验证地面支撑、坡度、悬崖/落差、图连通性或车辆运动学约束。
在用于真实机器人自主导航前，应结合地形分析、路径连通性和现场安全策略继续验证。

选择并添加合适的开源许可证；在此之前，源码不应被默认视为获得开源授权。
