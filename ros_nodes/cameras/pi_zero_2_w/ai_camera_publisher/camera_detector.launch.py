"""Launch file for camera detector node"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value="output/models/best.pt",
                description="Path to model file (.pt, .onnx, or int8 .onnx)",
            ),
            DeclareLaunchArgument(
                "config",
                default_value="configs/robotics.yaml",
                description="Path to config file",
            ),
            DeclareLaunchArgument(
                "publish_annotated",
                default_value="true",
                description="Whether to publish annotated images",
            ),
            Node(
                package="robotics_obj_detection",
                executable="camera_detector_node",
                name="camera_detector",
                parameters=[
                    {
                        "model": LaunchConfiguration("model"),
                        "config": LaunchConfiguration("config"),
                    }
                ],
                arguments=[
                    "--model",
                    LaunchConfiguration("model"),
                    "--config",
                    LaunchConfiguration("config"),
                ],
                output="screen",
            ),
        ]
    )
