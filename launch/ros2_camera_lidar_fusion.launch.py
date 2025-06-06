import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # config_path = os.path.expanduser('~/ros2_ws/src/ros2_camera_lidar_fusion/config/box_config.yaml')
    config_path = os.path.expanduser('~/ros2_ws/src/ros2_camera_lidar_fusion/config/general_configuration.yaml')


    return LaunchDescription([
        Node(
            package='ros2_camera_lidar_fusion',
            executable='lidar_camera_projection',
            name='ros2_camera_lidar_fusion',
            output='screen',
            parameters=[config_path]
        )
    ])
