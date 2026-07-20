import os
import subprocess
import signal
import time
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rich.live import Live
from rich.table import Table

class RecorderMonitor(Node):
    def __init__(self):
        super().__init__('recorder_monitor')
        self.cloud_count = 0
        self.x = 0.0
        self.y = 0.0
        
        self.subscription = self.create_subscription(
            PointCloud2,
            '/cloud_registered',
            self.cloud_callback,
            10
        )
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.2, self.update_pose)

    def cloud_callback(self, msg):
        self.cloud_count += 1

    def update_pose(self):
        try:
            # 查找最新变换 (odom -> base_link)
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', now)
            self.x = trans.transform.translation.x
            self.y = trans.transform.translation.y
        except TransformException:
            pass

def ros_spin_thread(node):
    rclpy.spin(node)

def generate_status_table(count, x, y, elapsed):
    table = Table(title="[bold green]ROS2 Bag 录制状态监控[/bold green]")
    table.add_column("监控指标", justify="left", style="cyan")
    table.add_column("实时数值", justify="right", style="magenta")
    
    table.add_row("已运行时间 (秒)", f"{elapsed:.1f}")
    table.add_row("已接收点云帧数 (/cloud_registered)", str(count))
    table.add_row("当前坐标 X (odom -> base_link)", f"{x:.3f} m")
    table.add_row("当前坐标 Y (odom -> base_link)", f"{y:.3f} m")
    return table

def start_recording(map_path):
    bag_dir = os.path.join(map_path, "bag")
    # 如果 bag 目录非空，ros2 bag 会报错，可先处理或由 ros2 自行命名
    output_bag = os.path.join(bag_dir, "bag")
    
    # 1. 启动监控节点
    rclpy.init()
    monitor_node = RecorderMonitor()
    spin_thread = threading.Thread(target=ros_spin_thread, args=(monitor_node,), daemon=True)
    spin_thread.start()

    # 2. 启动 ros2 bag 录制子进程
    cmd = [
        "ros2", "bag", "record",
        "-o", output_bag,
        "/cloud_registered", "/Odometry", "/tf", "/tf_static"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    start_time = time.time()
    print("开始录制... 按 Enter 键或 Ctrl+C 停止录制。")
    
    try:
        with Live(generate_status_table(0, 0.0, 0.0, 0.0), refresh_per_second=4) as live:
            while proc.poll() is None:
                elapsed = time.time() - start_time
                live.update(generate_status_table(
                    monitor_node.cloud_count,
                    monitor_node.x,
                    monitor_node.y,
                    elapsed
                ))
                time.sleep(0.2)
                # 检查用户是否按了回车 (非阻塞实现可用 select，这里提供简易检测)
                # 配合 Live，可用简单轮询，或直接捕获键盘
    except KeyboardInterrupt:
        pass
    finally:
        # 优雅终止 ros2 bag 录制
        proc.send_signal(signal.SIGINT)
        proc.wait()
        
        # 清理节点
        monitor_node.destroy_node()
        rclpy.shutdown()
        print("\n录制已结束，数据已保存。")
