"""Launch file for multi-camera subscriber on PC"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value="configs/pc/camera_subscriber_config.yaml",
                description="Path to subscriber configuration file",
            ),
            DeclareLaunchArgument(
                "display_enabled",
                default_value="true",
                description="Whether to show display window",
            ),
            DeclareLaunchArgument(
                "log_fps",
                default_value="true",
                description="Whether to log FPS information",
            ),
            Node(
                package="robotics_obj_detection",  # Adjust package name as needed
                executable="camera_subscriber",
                name="camera_subscriber_master",
                parameters=[
                    {
                        "config": LaunchConfiguration("config"),
                        "display_enabled": LaunchConfiguration("display_enabled"),
                        "log_fps": LaunchConfiguration("log_fps"),
                    }
                ],
                output="screen",
            ),
        ]
    )
