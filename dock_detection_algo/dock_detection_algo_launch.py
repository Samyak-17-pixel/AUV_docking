#!/usr/bin/env python3
"""Minimal ROS 2 launch: live dock-light viewer.

  ros2 launch dock_detection_algo_launch.py
  # or: ./run_live.sh   (preferred one-liner)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    root = Path(__file__).resolve().parent
    script = str(root / "live_dock_lights.py")

    topic_arg = DeclareLaunchArgument(
        "topic",
        default_value="/Mako_01/camera_03/image/compressed",
        description="CompressedImage camera topic",
    )
    topic = LaunchConfiguration("topic")

    # ExecuteProcess keeps this folder self-contained (no ament package needed).
    node = ExecuteProcess(
        cmd=["python3", script, "--topic", topic],
        cwd=str(root),
        output="screen",
    )
    return LaunchDescription([topic_arg, node])
