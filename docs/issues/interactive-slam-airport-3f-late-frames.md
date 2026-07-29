# Airport_3F Interactive SLAM 后段帧看似消失

## 现象

在 `Airport_3F` 启动 Interactive SLAM 时，观察到后半段帧似乎没有显示。

## 检查结果

地图目录：

```text
/home/timory/Map/Airport_3F
```

- `frame/` 包含 `000000.pcd` 到 `000976.pcd`，共 977 帧。
- `.pcd` 和 `.odom` 数量一致。
- 所有 PCD 都非空，点数约为 25,097 到 114,510。
- 未发现无效的 4×4 odom 矩阵、NaN 或 Inf 点云值。
- `odometry2graph` 源码会扫描目录中的全部 `.pcd` 文件，没有后段帧数截断逻辑。

因此目前没有证据表明 Frame Extractor 在某个帧号处中断或生成了损坏文件。

## 关键帧行为

`odometry2graph` 默认只显示关键帧，默认阈值为：

```text
keyframe_delta_x = 3.0 m
keyframe_delta_angle = 1.0 rad
```

Airport_3F 的 977 个输入帧按默认阈值约筛选为 426 个关键帧，最后一个关键帧为 973。末尾 974–976 帧与前一个关键帧的位姿变化不足阈值，因此不单独显示，这是预期行为。

## 主要原因

Airport_3F 的原始 odom 存在明显非平面漂移：

```text
frame 0:   z ≈   0.04 m
frame 100: z ≈  -2.12 m
frame 250: z ≈ -10.90 m
frame 400: z ≈ -31.72 m
frame 500: z ≈ -46.13 m
最低位置:  z ≈ -47.08 m
```

roll/pitch 也出现约 8° 的漂移。如果 Airport_3F 是单层平面场景，后段点云会被放置到错误高度，或者与返程轨迹重叠，从视图上看起来像是没有加载。

## 建议操作

1. 先执行“强制平面约束”，将 z、roll、pitch 置零。
2. 再启动 Interactive SLAM。
3. 在 GUI 中执行 `View → Reset camera`。
4. 如需显示更多输入帧，将 `keyframe_delta_x` 从 3.0 调低到约 1.0。
5. Airport_3F 点云较大，可将 `downsample_resolution` 调高到 0.3–0.5 以降低渲染内存压力。

## 结论

当前更可能是原始 odom 漂移和关键帧筛选造成的可视化误判，而不是 PCD 转换在后段失败。若平面约束后仍能复现，再进一步检查 GUI 渲染资源和关键帧加载日志。
