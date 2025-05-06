from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='ros2_camera_lidar_fusion',
            executable='lidar_camera_projection',
            name='lidar_camera_projection_node',
            output='screen',
            parameters=['/home/robolab/ros2_ws/src/ros2_camera_lidar_fusion/config/general_configuration.yaml']
        )
    ])
