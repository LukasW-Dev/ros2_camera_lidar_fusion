import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    config_path = os.path.expanduser('~/ros2_ws/src/ros2_camera_lidar_fusion/config/box_config.yaml')

    return LaunchDescription([
        Node(
            package='ros2_camera_lidar_fusion',
            executable='lidar_camera_projection',
            name='lidar_camera_projection_node',
            output='screen',
            parameters=[config_path]
        )
    ])
