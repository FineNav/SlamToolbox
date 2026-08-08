"""Local voxel dynamic-removal workflows."""

import os
import sys
from datetime import datetime
from pathlib import Path

import questionary
import yaml
from questionary import Choice, Style
from rich.table import Table

from .dynamic_removal_common import (
    _PKG_DIR,
    _run_measured_command,
    _timestamped_output_dir,
    console,
)
from .dynamic_removal_dataset import _ensure_local_kitti_dataset
from .dynamic_removal_results import (
    _keep_only_standard_run_pcds,
    _prompt_generate_dynamic_gif,
    _publish_run_result,
)

_PARAMETER_GO_BACK = object()
_PARAMETER_CONFIG_NAME = "dynamic_removal_params.yaml"
_PARAMETER_QUESTION_STYLE = Style(
    [
        ("highlighted", "fg:ansidefault bg:ansidefault noreverse"),
        ("pointer", "fg:#00aa00 bold noreverse"),
        ("selected", "fg:ansidefault bg:ansidefault noreverse"),
    ]
)


def _parameter_config_path(map_path):
    return Path(map_path) / _PARAMETER_CONFIG_NAME


def _coerce_saved_parameter(spec, value):
    if value is None:
        return None if spec.get("optional") else None
    kind = spec["kind"]
    try:
        if kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "y", "1", "on"}:
                    return True
                if lowered in {"false", "no", "n", "0", "off"}:
                    return False
            return bool(value)
        if kind == "int":
            return int(value)
        if kind == "float":
            return float(value)
        if kind == "choice":
            return value if value in spec["choices"] else None
        return str(value)
    except (TypeError, ValueError):
        return None


def _is_valid_parameter_value(spec, value):
    if value is None:
        return bool(spec.get("optional"))
    minimum = spec.get("min")
    maximum = spec.get("max")
    if minimum is not None and value < minimum:
        return False
    if spec.get("positive") and value <= 0:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def _load_algorithm_parameter_config(map_path, method_key, specs, defaults):
    config_path = _parameter_config_path(map_path)
    if not config_path.exists():
        return defaults.copy()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        console.print(f"[yellow]参数配置读取失败，将使用默认参数: {exc}[/yellow]")
        return defaults.copy()

    method_config = config.get(method_key, {})
    saved_values = method_config.get("parameters", {})
    if not isinstance(saved_values, dict):
        return defaults.copy()

    specs_by_key = {spec["key"]: spec for spec in specs}
    values = defaults.copy()
    loaded = []
    for key, raw_value in saved_values.items():
        spec = specs_by_key.get(key)
        if spec is None:
            continue
        value = _coerce_saved_parameter(spec, raw_value)
        if value is None and not spec.get("optional"):
            continue
        if not _is_valid_parameter_value(spec, value):
            continue
        values[key] = value
        loaded.append(key)

    errors = _parameter_relationship_errors(values)
    if errors:
        console.print(
            f"[yellow]{config_path} 中 {method_key} 参数关系无效，将使用默认参数。[/yellow]"
        )
        return defaults.copy()

    if loaded:
        console.print(f"[dim]已加载上次 {method_key} 参数配置: {config_path}[/dim]")
    return values


def _save_algorithm_parameter_config(map_path, method_key, values):
    config_path = _parameter_config_path(map_path)
    try:
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            config = {}
        if not isinstance(config, dict):
            config = {}
        config[method_key] = {
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "parameters": values,
        }
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    except Exception as exc:
        console.print(f"[yellow]参数配置保存失败: {exc}[/yellow]")
        return
    console.print(f"[dim]已保存本次参数配置: {config_path}[/dim]")


def _run_python_script(script_name, args, output_dir, method_name):
    script_path = _PKG_DIR / "algorithms" / script_name
    cmd = [sys.executable, str(script_path)] + args
    env = os.environ.copy()
    project_root = str(_PKG_DIR.parent)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = project_root if not pythonpath else f"{project_root}{os.pathsep}{pythonpath}"
    return _run_measured_command(cmd, output_dir, method_name, env=env)



def _format_parameter_value(value):
    if value is None:
        return "关闭/自动"
    if isinstance(value, bool):
        return "开启" if value else "关闭"
    return str(value)


def _show_algorithm_parameters(title, specs, values, dataset):
    table = Table(title=f"{title} 当前参数", show_lines=False)
    table.add_column("参数", style="cyan", no_wrap=True)
    table.add_column("当前值", style="green", no_wrap=True)
    table.add_column("作用")
    table.add_row("--dataset", str(dataset), "数据集路径（由当前地图自动确定）")
    table.add_row("--out", "自动生成", "本次运行的时间戳输出目录")
    for spec in specs:
        table.add_row(
            f"--{spec['option']}",
            _format_parameter_value(values[spec["key"]]),
            spec["description"],
        )
    console.print(table)


def _prompt_parameter_value(spec, current, position, total):
    kind = spec["kind"]
    prompt = f"[{position}/{total}] {spec['label']} (--{spec['option']})"
    back_label = "← 返回上一个参数" if position > 1 else "← 返回参数选择"
    if kind == "bool":
        answer = questionary.select(
            f"{prompt}:",
            choices=[
                Choice("开启", value=True),
                Choice("关闭", value=False),
                Choice(back_label, value="__back__"),
            ],
            default=bool(current),
            style=_PARAMETER_QUESTION_STYLE,
        ).ask()
        return _PARAMETER_GO_BACK if answer is None or answer == "__back__" else answer
    if kind == "choice":
        answer = questionary.select(
            f"{prompt}:",
            choices=[*spec["choices"], Choice(back_label, value="__back__")],
            default=current,
            style=_PARAMETER_QUESTION_STYLE,
        ).ask()
        return _PARAMETER_GO_BACK if answer is None or answer == "__back__" else answer

    while True:
        default = "" if current is None else str(current)
        back_hint = "输入 back 返回上一参数" if position > 1 else "输入 back 返回参数选择"
        answer = questionary.text(
            f"{prompt}，{back_hint}:",
            default=default,
            style=_PARAMETER_QUESTION_STYLE,
        ).ask()
        if answer is None:
            return _PARAMETER_GO_BACK
        answer = answer.strip()
        if answer.lower() in {"back", ":back"}:
            return _PARAMETER_GO_BACK
        if spec.get("optional") and not answer:
            return None
        try:
            if kind == "int":
                value = int(answer)
            elif kind == "float":
                value = float(answer)
            else:
                if not answer:
                    raise ValueError
                return answer
        except ValueError:
            console.print("[yellow]请输入有效的数值。[/yellow]" if kind in ("int", "float") else "[yellow]输入不能为空。[/yellow]")
            continue

        minimum = spec.get("min")
        maximum = spec.get("max")
        if minimum is not None and value < minimum:
            console.print(f"[yellow]参数值必须大于或等于 {minimum}。[/yellow]")
            continue
        if spec.get("positive") and value <= 0:
            console.print("[yellow]参数值必须大于 0。[/yellow]")
            continue
        if maximum is not None and value > maximum:
            console.print(f"[yellow]参数值必须小于或等于 {maximum}。[/yellow]")
            continue
        return value


def _parameter_relationship_errors(values):
    errors = []
    if values.get("end") is not None and values["start"] > values["end"]:
        errors.append("--start 不能大于 --end")
    if values["local_z_min"] > values["local_z_max"]:
        errors.append("--local-z-min 不能大于 --local-z-max")
    return errors


def _configure_algorithm_parameters(title, specs, defaults, dataset):
    values = defaults.copy()
    while True:
        _show_algorithm_parameters(title, specs, values, dataset)
        action = questionary.select(
            "是否修改参数？",
            choices=[
                Choice("否，直接使用以上参数运行", value="run"),
                Choice("是，选择需要修改的参数", value="edit"),
                Choice("← 返回", value="back"),
            ],
            default="run",
            style=_PARAMETER_QUESTION_STYLE,
        ).ask()
        if action is None or action == "back":
            return None
        if action == "run":
            return values

        while True:
            choices = [
                Choice(
                    title=(
                        f"--{spec['option']:<28} 当前值: "
                        f"{_format_parameter_value(values[spec['key']])}"
                    ),
                    value=spec["key"],
                )
                for spec in specs
            ]
            choices.append(Choice("← 返回参数总览", value="__back__"))
            selected = questionary.checkbox(
                "请先选完所有需要修改的参数，再按 Enter：",
                choices=choices,
                style=_PARAMETER_QUESTION_STYLE,
            ).ask()
            if selected is None or "__back__" in selected:
                break
            if not selected:
                console.print("[yellow]尚未选择参数。请选择参数，或选择“返回参数总览”。[/yellow]")
                continue

            specs_by_key = {spec["key"]: spec for spec in specs}
            returned_to_selection = False
            position = 0
            while position < len(selected):
                key = selected[position]
                value = _prompt_parameter_value(
                    specs_by_key[key],
                    values[key],
                    position + 1,
                    len(selected),
                )
                if value is _PARAMETER_GO_BACK:
                    if position > 0:
                        position -= 1
                    else:
                        returned_to_selection = True
                        break
                    continue
                values[key] = value
                position += 1
            if returned_to_selection:
                continue

            errors = _parameter_relationship_errors(values)
            if errors:
                for error in errors:
                    console.print(f"[yellow]{error}。[/yellow]")
                console.print("[yellow]请重新选择相关参数进行修改。[/yellow]")
                continue

            _show_algorithm_parameters(f"{title}（修改后）", specs, values, dataset)
            final_action = questionary.select(
                "参数修改完成，请选择下一步：",
                choices=[
                    Choice("使用以上参数运行", value="run"),
                    Choice("← 返回参数选择", value="edit"),
                    Choice("← 返回参数总览", value="overview"),
                    Choice("取消并返回上一级", value="cancel"),
                ],
                default="run",
                style=_PARAMETER_QUESTION_STYLE,
            ).ask()
            if final_action == "run":
                return values
            if final_action == "overview":
                break
            if final_action is None or final_action == "cancel":
                return None
            # "edit" returns to the multi-select list with completed edits preserved.
            continue


def _build_algorithm_cli_args(dataset, output_dir, specs, values):
    args = ["--dataset", str(dataset), "--out", str(output_dir)]
    for spec in specs:
        value = values[spec["key"]]
        if value is None:
            continue
        option = f"--{spec['option']}"
        if spec["kind"] == "bool":
            if value:
                args.append(option)
        else:
            args.extend([option, str(value)])
    return args


_LOCAL_HASH_PARAMETER_SPECS = [
    {"key": "seq", "option": "seq", "label": "序列编号", "kind": "str", "description": "要处理的数据序列"},
    {"key": "start", "option": "start", "label": "起始帧", "kind": "int", "min": 0, "description": "处理的第一帧（包含）"},
    {"key": "end", "option": "end", "label": "结束帧（留空表示最后一帧）", "kind": "int", "min": 0, "optional": True, "description": "处理的最后一帧（包含）"},
    {"key": "pose", "option": "pose", "label": "位姿文件（留空自动选择）", "kind": "str", "optional": True, "description": "覆盖自动选择的位姿文件"},
    {"key": "voxel_size", "option": "voxel-size", "label": "Voxel 尺寸（米）", "kind": "float", "positive": True, "description": "Hash voxel 边长"},
    {"key": "max_range", "option": "max-range", "label": "最大水平距离（米）", "kind": "float", "positive": True, "description": "小车局部 XY 观测半径"},
    {"key": "local_z_min", "option": "local-z-min", "label": "局部 Z 下限（米）", "kind": "float", "description": "动态分类区域下限"},
    {"key": "local_z_max", "option": "local-z-max", "label": "局部 Z 上限（米）", "kind": "float", "description": "动态分类区域上限"},
    {"key": "body_radius", "option": "body-radius", "label": "车体过滤半径（米）", "kind": "float", "min": 0, "description": "排除小车周围点；0 关闭"},
    {"key": "ground_z", "option": "ground-protect-local-z-max", "label": "地面保护 Z 上限（留空关闭）", "kind": "float", "optional": True, "description": "强制保留该局部 Z 以下的点"},
    {"key": "lidar_hz", "option": "lidar-hz", "label": "雷达频率（Hz）", "kind": "float", "positive": True, "description": "缺少时间戳时用于计算可见时间"},
    {"key": "ray_stride", "option": "ray-stride", "label": "射线终点步长", "kind": "int", "min": 1, "description": "每 N 个终点 voxel 追踪一条射线"},
    {"key": "max_ray_endpoints", "option": "max-ray-endpoints", "label": "每帧最大射线终点数", "kind": "int", "min": 0, "description": "0 表示不限制"},
    {"key": "min_visible_frames", "option": "min-visible-frames", "label": "最少可见帧数", "kind": "int", "min": 0, "description": "参与分类所需的可见帧数"},
    {"key": "min_visible_time", "option": "min-visible-time", "label": "最短可见时间（秒）", "kind": "float", "min": 0, "description": "参与分类所需的可见时间"},
    {"key": "static_min_hit_ratio", "option": "static-min-hit-ratio", "label": "静态最小命中比例", "kind": "float", "min": 0, "max": 1, "description": "达到该比例时判为静态"},
    {"key": "dynamic_max_hit_ratio", "option": "dynamic-max-hit-ratio", "label": "动态最大命中比例", "kind": "float", "min": 0, "max": 1, "description": "不超过该比例时可能判为动态"},
    {"key": "dynamic_max_hit_time", "option": "dynamic-max-hit-time", "label": "动态最大命中时间（秒）", "kind": "float", "min": 0, "description": "动态 voxel 允许的最长命中跨度"},
    {"key": "unknown_policy", "option": "unknown-policy", "label": "未知 voxel 策略", "kind": "choice", "choices": ["keep", "drop"], "description": "保留或删除无法明确分类的 voxel"},
    {"key": "backend", "option": "backend", "label": "计算后端", "kind": "choice", "choices": ["auto", "native", "python"], "description": "auto 优先使用 C++，不可用时回退到 Python"},
    {"key": "progress_interval", "option": "progress-interval", "label": "进度输出间隔", "kind": "int", "min": 0, "description": "每 N 帧输出进度；0 关闭"},
    {"key": "write_before", "option": "write-before", "label": "输出清理前点云", "kind": "bool", "description": "写出 before.pcd"},
    {"key": "save_dynamic_frames", "option": "save-dynamic-frames", "label": "保存逐帧动态点云", "kind": "bool", "description": "用于动态 GIF"},
]


_RAYCAST_PARAMETER_SPECS = [
    {"key": "seq", "option": "seq", "label": "序列编号", "kind": "str", "description": "要处理的数据序列"},
    {"key": "pose", "option": "pose", "label": "位姿文件（留空自动选择）", "kind": "str", "optional": True, "description": "覆盖默认 poses_odom_base.txt"},
    {"key": "start", "option": "start", "label": "起始帧", "kind": "int", "min": 0, "description": "处理的第一帧（包含）"},
    {"key": "end", "option": "end", "label": "结束帧（留空表示最后一帧）", "kind": "int", "min": 0, "optional": True, "description": "处理的最后一帧（包含）"},
    {"key": "stride", "option": "stride", "label": "处理帧步长", "kind": "int", "min": 1, "description": "每 N 帧处理一帧"},
    {"key": "voxel_size", "option": "voxel-size", "label": "Voxel 尺寸（米）", "kind": "float", "positive": True, "description": "八叉树叶节点边长"},
    {"key": "max_range", "option": "max-range", "label": "最大水平距离（米）", "kind": "float", "min": 0, "description": "小车局部 XY 观测半径；0 关闭"},
    {"key": "body_radius", "option": "body-radius", "label": "车体过滤半径（米）", "kind": "float", "min": 0, "description": "排除小车周围点；0 关闭"},
    {"key": "local_z_min", "option": "local-z-min", "label": "局部 Z 下限（米）", "kind": "float", "description": "参与清理的局部 Z 下限"},
    {"key": "local_z_max", "option": "local-z-max", "label": "局部 Z 上限（米）", "kind": "float", "description": "参与清理的局部 Z 上限"},
    {"key": "ground_z", "option": "ground-protect-local-z-max", "label": "地面保护 Z 上限（留空关闭）", "kind": "float", "optional": True, "description": "强制保留该局部 Z 以下的点"},
    {"key": "ray_point_stride", "option": "ray-point-stride", "label": "射线点步长", "kind": "int", "min": 1, "description": "每 N 个点选择一个射线终点"},
    {"key": "ray_step_factor", "option": "ray-step-factor", "label": "射线采样因子", "kind": "float", "positive": True, "description": "仅动态端点模型使用；完整 OctoMap 使用精确 voxel traversal"},
    {"key": "endpoint_margin", "option": "endpoint-margin", "label": "终点保护距离（米）", "kind": "float", "min": 0, "description": "射线终点前不更新为空闲的距离"},
    {"key": "hit_log_odds", "option": "hit-log-odds", "label": "命中分数增量", "kind": "float", "positive": True, "description": "每帧命中增加的占用分数"},
    {"key": "miss_log_odds", "option": "miss-log-odds", "label": "穿过分数减量", "kind": "float", "positive": True, "description": "每帧射线穿过减少的占用分数"},
    {"key": "occupied_threshold", "option": "occupied-threshold", "label": "占用分数阈值", "kind": "float", "description": "log-odds 达到该分数才判为占用"},
    {"key": "free_threshold", "option": "free-threshold", "label": "空闲分数阈值", "kind": "float", "description": "log-odds 降到该分数以下才判为空闲并删除"},
    {"key": "unknown_policy", "option": "unknown-policy", "label": "未知 voxel 策略", "kind": "choice", "choices": ["keep", "drop"], "description": "保留或删除介于 occupied/free 阈值之间的 voxel"},
    {"key": "export_octomap", "option": "export-octomap", "label": "导出 OctoMap 分层包", "kind": "bool", "description": "同步输出 occupied/free/traversable/risk voxel 层"},
    {"key": "robot_footprint", "option": "robot-footprint", "label": "机器人 footprint", "kind": "choice", "choices": ["circle"], "description": "当前支持圆形 footprint"},
    {"key": "robot_radius", "option": "robot-radius", "label": "机器人半径（米）", "kind": "float", "positive": True, "description": "圆形 footprint 半径"},
    {"key": "robot_height", "option": "robot-height", "label": "机器人高度（米）", "kind": "float", "positive": True, "description": "可通行层头顶净空检查高度"},
    {"key": "robot_ground_clearance", "option": "robot-ground-clearance", "label": "离地间隙（米）", "kind": "float", "min": 0, "description": "低于该高度的点不作为机身碰撞障碍"},
    {"key": "robot_max_step_height", "option": "robot-max-step-height", "label": "最大台阶高度（米）", "kind": "float", "min": 0, "description": "地面高度带 = ground_z ± 该值"},
    {"key": "robot_safety_margin", "option": "robot-safety-margin", "label": "安全边距（米）", "kind": "float", "min": 0, "description": "机器人半径之外的风险膨胀距离"},
    {"key": "robot_ground_z", "option": "robot-ground-z", "label": "地图地面 Z（米）", "kind": "float", "description": "默认假设地图 frame 中地面高度"},
    {"key": "risk_radius", "option": "risk-radius", "label": "风险半径覆盖（留空自动）", "kind": "float", "min": 0, "optional": True, "description": "留空则使用机器人半径 + 安全边距"},
    {"key": "risk_levels", "option": "risk-levels", "label": "风险等级数", "kind": "int", "min": 0, "description": "按距离衰减生成的风险等级数；0 关闭"},
    {"key": "compressed_octomap", "option": "compressed-octomap", "label": "压缩 OctoMap 分层包", "kind": "bool", "description": "减小 layers.npz 体积但导出更慢"},
    {"key": "backend", "option": "backend", "label": "计算后端", "kind": "choice", "choices": ["auto", "native", "python"], "description": "auto 优先使用 C++，不可用时回退 Python"},
    {"key": "write_before", "option": "write-before", "label": "输出清理前点云", "kind": "bool", "description": "写出 before.pcd"},
    {"key": "save_dynamic_frames", "option": "save-dynamic-frames", "label": "保存逐帧动态点云", "kind": "bool", "description": "用于动态 GIF"},
    {"key": "progress_interval", "option": "progress-interval", "label": "进度输出间隔", "kind": "int", "min": 0, "description": "每 N 帧输出进度；0 关闭"},
]


def start_local_hash_voxel(map_path):
    """Local hash voxel 动态障碍物清除。"""
    try:
        kitti_root, frame_count = _ensure_local_kitti_dataset(map_path)
    except Exception as e:
        console.print(f"[red]Local Hash Voxel 无法运行: {e}[/red]")
        return

    print(f"\n检测到 {frame_count} 帧，准备运行 Local Hash Voxel 动态障碍物清除。\n")

    local_defaults = {
        "seq": "00",
        "start": 0,
        "end": None,
        "pose": None,
        "voxel_size": 0.5,
        "max_range": 30.0,
        "local_z_min": 0.0,
        "local_z_max": 3.0,
        "body_radius": 0.5,
        "ground_z": 0.0,
        "lidar_hz": 10.0,
        "ray_stride": 4,
        "max_ray_endpoints": 25000,
        "min_visible_frames": 10,
        "min_visible_time": 2.0,
        "static_min_hit_ratio": 0.5,
        "dynamic_max_hit_ratio": 0.15,
        "dynamic_max_hit_time": 2.0,
        "unknown_policy": "keep",
        "backend": "auto",
        "progress_interval": 25,
        "write_before": True,
        "save_dynamic_frames": True,
    }
    local_defaults = _load_algorithm_parameter_config(
        map_path,
        "local_hash_voxel",
        _LOCAL_HASH_PARAMETER_SPECS,
        local_defaults,
    )
    values = _configure_algorithm_parameters(
        "Local Hash Voxel",
        _LOCAL_HASH_PARAMETER_SPECS,
        local_defaults,
        kitti_root,
    )
    if values is None:
        console.print("[dim]已取消 Local Hash Voxel 运行。[/dim]")
        return
    _save_algorithm_parameter_config(map_path, "local_hash_voxel", values)

    output_dir = _timestamped_output_dir(map_path, "local_hash_voxel")
    args = _build_algorithm_cli_args(kitti_root, output_dir, _LOCAL_HASH_PARAMETER_SPECS, values)

    ret = _run_python_script(
        "local_hash_voxel_filter.py",
        args,
        output_dir,
        "local_hash_voxel",
    )
    if ret != 0:
        console.print(f"[yellow]Local Hash Voxel 返回非零退出码: {ret}[/yellow]")
        return

    console.print("\n[bold green]Local Hash Voxel 处理完成！[/bold green]")
    print(f"  完整输出目录: {output_dir}")
    run_before = _publish_run_result(os.path.join(output_dir, "local_hash_voxel_before.pcd"), output_dir, "before.pcd", map_path, "map_local_hash_voxel_before.pcd", "Local Hash 移除前")
    run_static = _publish_run_result(os.path.join(output_dir, "local_hash_voxel_after.pcd"), output_dir, "static.pcd", map_path, "map_local_hash_voxel_static.pcd", "Local Hash 移除后")
    run_dynamic = _publish_run_result(os.path.join(output_dir, "local_hash_voxel_dynamic.pcd"), output_dir, "dynamic.pcd", map_path, "map_local_hash_voxel_dynamic.pcd", "Local Hash 被移除动态")
    _keep_only_standard_run_pcds(output_dir)
    _prompt_generate_dynamic_gif(
        map_path,
        output_dir,
        "local_hash_voxel",
        "map_local_hash_voxel_static.pcd",
        dynamic_reference=run_dynamic,
        before_reference=run_before,
        static_reference=run_static,
        z_min=values["local_z_min"],
        z_max=values["local_z_max"],
    )


def start_raycast_voxel(map_path):
    """Raycast voxel cleanup 动态障碍物清除。"""
    try:
        kitti_root, frame_count = _ensure_local_kitti_dataset(map_path)
    except Exception as e:
        console.print(f"[red]Raycast Voxel 无法运行: {e}[/red]")
        return

    print(f"\n检测到 {frame_count} 帧，准备运行 Raycast Voxel Cleanup。\n")

    raycast_defaults = {
        "seq": "00",
        "pose": None,
        "start": 0,
        "end": None,
        "stride": 1,
        "voxel_size": 0.5,
        "max_range": 30.0,
        "body_radius": 0.5,
        "local_z_min": 0.0,
        "local_z_max": 3.0,
        "ground_z": 0.0,
        "ray_point_stride": 4,
        "ray_step_factor": 0.75,
        "endpoint_margin": 0.10,
        "hit_log_odds": 0.40,
        "miss_log_odds": 0.80,
        "occupied_threshold": 1.5,
        "free_threshold": -1.0,
        "unknown_policy": "keep",
        "export_octomap": True,
        "robot_footprint": "circle",
        "robot_radius": 0.35,
        "robot_height": 1.2,
        "robot_ground_clearance": 0.05,
        "robot_max_step_height": 0.12,
        "robot_safety_margin": 0.20,
        "robot_ground_z": 0.0,
        "risk_radius": None,
        "risk_levels": 3,
        "compressed_octomap": False,
        "backend": "auto",
        "write_before": True,
        "save_dynamic_frames": True,
        "progress_interval": 25,
    }
    raycast_defaults = _load_algorithm_parameter_config(
        map_path,
        "raycast_voxel",
        _RAYCAST_PARAMETER_SPECS,
        raycast_defaults,
    )
    values = _configure_algorithm_parameters(
        "Raycast Voxel",
        _RAYCAST_PARAMETER_SPECS,
        raycast_defaults,
        kitti_root,
    )
    if values is None:
        console.print("[dim]已取消 Raycast Voxel 运行。[/dim]")
        return
    _save_algorithm_parameter_config(map_path, "raycast_voxel", values)

    output_dir = _timestamped_output_dir(map_path, "raycast_voxel")
    args = _build_algorithm_cli_args(kitti_root, output_dir, _RAYCAST_PARAMETER_SPECS, values)

    ret = _run_python_script(
        "raycast_voxel_cleanup.py",
        args,
        output_dir,
        "raycast_voxel",
    )
    if ret != 0:
        console.print(f"[yellow]Raycast Voxel 返回非零退出码: {ret}[/yellow]")
        return

    console.print("\n[bold green]Raycast Voxel 处理完成！[/bold green]")
    print(f"  完整输出目录: {output_dir}")
    run_before = _publish_run_result(os.path.join(output_dir, "raycast_before.pcd"), output_dir, "before.pcd", map_path, "map_raycast_voxel_before.pcd", "Raycast 移除前")
    run_static = _publish_run_result(os.path.join(output_dir, "raycast_after.pcd"), output_dir, "static.pcd", map_path, "map_raycast_voxel_static.pcd", "Raycast 移除后")
    run_dynamic = _publish_run_result(os.path.join(output_dir, "raycast_removed.pcd"), output_dir, "dynamic.pcd", map_path, "map_raycast_voxel_dynamic.pcd", "Raycast 被移除动态")
    _keep_only_standard_run_pcds(output_dir)
    _prompt_generate_dynamic_gif(
        map_path,
        output_dir,
        "raycast_voxel",
        "map_raycast_voxel_static.pcd",
        dynamic_reference=run_dynamic,
        before_reference=run_before,
        static_reference=run_static,
        z_min=values["local_z_min"],
        z_max=values["local_z_max"],
    )
