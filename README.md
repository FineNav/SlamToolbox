# SlamToolbox

面向 ROS2 激光点云地图的交互式处理工具，提供 rosbag 录制、点云帧提取、
位姿修正、动态障碍物清除、3D 地图构建和 2D 栅格地图生成。用户通过统一的
终端菜单完成操作，不需要直接调用内部算法脚本。

## 1. 支持环境

已验证和支持的环境：

- Ubuntu 22.04 x86_64、Python 3.10、ROS2 Humble；
- Ubuntu 24.04 x86_64、Python 3.12、ROS2 Jazzy；
- CMake 3.18 或更高版本；
- 支持 C++17 的编译器；
- OctoMap development package；
- ROS2 的 `rclpy` 和 rosbag2 Python 包。

ERASOR2、Removert 和 Interactive SLAM 通过 Docker 运行；使用这些功能前还需
按 [Docker Engine 官方文档](https://docs.docker.com/engine/install/ubuntu/)
安装 Docker，并确保当前用户能够执行 `docker info`。其他菜单功能不要求 Docker。

其他 Linux 发行版、macOS、Windows 和 ARM64 尚未验证，不能直接套用下面的
Ubuntu 安装命令。完整安装包含 Open3D，建议至少预留 5 GiB 磁盘空间。

## 2. 全新安装

请先安装与 Ubuntu 版本匹配的 ROS2：

- Ubuntu 22.04：[ROS2 Humble 官方安装文档](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)；
- Ubuntu 24.04：[ROS2 Jazzy 官方安装文档](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)。

ROS2 安装完成后，下面的代码块同时安装项目的系统依赖、源码和 Python 环境。

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  liboctomap-dev \
  python3-dev \
  python3-venv

# 必须输出 octomap-config.cmake；没有输出时不要继续安装 Python 包。
dpkg -L liboctomap-dev | grep -E '/octomap-(config|targets)\.cmake$'

git clone https://github.com/FineNav/SlamToolbox.git
cd SlamToolbox

python3 -m venv venv-native
venv-native/bin/python -m pip install --upgrade pip
venv-native/bin/python -m pip install -e .
```

`liboctomap-dev` 是 native 扩展的必需依赖，不是只在导出 OctoMap 时才需要。
必须先安装它，再执行 `pip install -e .`；否则 CMake 会因为找不到
`octomap-config.cmake` 而终止配置。安装 Open3D 等 Python 依赖时下载量较大，
网速较慢并不表示 native 构建卡死。

不要使用系统 Python 直接执行 `pip install`。Ubuntu 的 externally-managed
Python 可能报 PEP 668 错误，虚拟环境可以避免这个问题。

验证当前 Python、项目路径和 native 扩展：

```bash
venv-native/bin/python - <<'PY'
import sys
import slam_toolbox
from slam_toolbox import _native

print("python:", sys.executable)
print("package:", slam_toolbox.__file__)
print("native api:", _native.api_version)
PY
```

正确安装时应看到：

```text
native api: 5
```

## 3. 启动和使用

每次打开新终端后，先加载 ROS2 环境，再从项目目录启动 SlamToolbox。

Ubuntu 22.04 / ROS2 Humble：

```bash
source /opt/ros/humble/setup.bash
cd ~/SlamToolbox
venv-native/bin/python -c "import rclpy, rosbag2_py" （自检，不是每次必须）
venv-native/bin/slam_toolbox
```
或
```
source /opt/ros/humble/setup.bash
venv-native/bin/slam_toolbox
```

Ubuntu 24.04 / ROS2 Jazzy：

```bash
source /opt/ros/jazzy/setup.bash
cd ~/SlamToolbox
venv-native/bin/python -c "import rclpy, rosbag2_py"（自检，不是每次必须）
venv-native/bin/slam_toolbox
```
或
```
source /opt/ros/jazzy/setup.bash
venv-native/bin/slam_toolbox
```

如果项目没有克隆在 `~/SlamToolbox`，把 `cd` 后面的路径换成实际项目路径。
如果还使用自己的 ROS2 工作空间，应在加载系统 ROS2 后继续 source 该工作空间的
`install/setup.bash`。

启动后按终端菜单操作：

1. 选择已有地图，或者选择“新建地图”并输入名称；
2. 第一次进入地图时确认 `fixed_frame`、`base_link_frame` 和点云话题；
3. 进入 `3D Map`，根据数据状态依次完成 rosbag 录制、Frame Extractor、
   位姿修正、动态障碍物清除和 Map Builder；不需要的步骤可以跳过；
4. 需要 2D 导航地图时，进入 `2D Map` 并选择 PGM Generator；
5. 每个功能完成后，终端都会打印本次输出的绝对路径。

SlamToolbox 只扫描 `~/Map` 下的地图目录。选择“新建地图”时会自动创建基本目录；
已有 rosbag 也可以放到 `~/Map/<地图名称>/bag/` 后再启动工具。

## 4. 输入和输出位置

一个地图的主要目录如下：

```text
~/Map/<map-name>/
├── config.yaml                         # 地图坐标系和点云话题
├── bag/                                # 录制或导入的 rosbag
├── frame/                              # 提取的 PCD 帧和对应 .odom 位姿
├── interactive_slam/
│   ├── original/                       # 位姿图原始数据
│   └── corrected/                      # 交互优化结果
├── pose_correction/                    # 位姿修正报告和备份
├── runs/
│   ├── <dynamic-method>/<timestamp>/   # 每次动态清除的独立结果
│   └── map_builder/<timestamp>/        # 每次 3D 建图结果
└── map/
    ├── map.pcd                         # 最新 3D 地图
    ├── map.pgm                         # 最新 2D 栅格图
    └── map.yaml                        # ROS2 2D 地图配置
```

动态障碍物清除的每次运行都写入新的时间戳目录，不会覆盖以前的结果：

```text
~/Map/<map-name>/runs/<dynamic-method>/<timestamp>/
├── before.pcd            # 动态清除前的完整点云
├── static.pcd            # 动态清除后的静态点云
├── dynamic.pcd           # 被移除的动态点
├── resource_usage.yaml   # 运行时间和资源统计
└── ...                   # 参数、日志和可视化文件
```

Map Builder 的历史结果位于：

```text
~/Map/<map-name>/runs/map_builder/<timestamp>/map.pcd
```

同时会把最新结果复制到 `~/Map/<map-name>/map/map.pcd`。PGM Generator
读取这个最新 3D 地图，并把 `map.pgm` 和 `map.yaml` 写入同一个 `map/` 目录。
