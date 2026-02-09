from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    planner_node = Node(
        package='inspection_robot',
        executable='trajectory_planner',
        name='trajectory_planner',
        output='screen',
        parameters=[{
            'standoff_distance': 0.15,
            'scan_line_spacing': 0.03,
            'high_density_spacing': 0.015,
            'overlap_percentage': 20.0,
        }]
    )
    
    return LaunchDescription([
        planner_node,
    ])