import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # Command-line arguments
    db_arg = DeclareLaunchArgument(
        "db", default_value="False", description="Database flag"
    )
    
    pkg_share = get_package_share_directory("fanuc_moveit_config")
    
    # MoveIt Configuration
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_fanuc")
        .robot_description(file_path="config/fanuc.urdf.xacro")
        .robot_description_semantic(file_path="config/fanuc.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )
    
    # Sensor parameters for octomap
    sensors_params = {
        "sensors": ["pointcloud"],
        "pointcloud": {
            "sensor_plugin": "occupancy_map_monitor/PointCloudOctomapUpdater",
            "point_cloud_topic": "/camera/depth/points",
            "max_range": 1.5,
            "point_subsample": 1,
            "padding_offset": 0.02,
            "padding_scale": 1.0,
            "filtered_cloud_topic": "filtered_cloud",
            "max_update_rate": 5.0,
        },
        "octomap_frame": "world",
        "octomap_resolution": 0.02,
    }
    
    # Start move_group node
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            sensors_params,
            {
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
            }
        ],
    )
    
    # RViz
    rviz_base = os.path.join(pkg_share, "launch")
    rviz_full_config = os.path.join(rviz_base, "moveit.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_full_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
    )
    
    # Static TF (world to base_link)
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
    )
    
    # Warehouse mongodb server (optional)
    db_config = LaunchConfiguration("db")
    mongodb_server_node = Node(
        package="warehouse_ros_mongo",
        executable="mongo_wrapper_ros.py",
        parameters=[
            {"warehouse_port": 33829},
            {"warehouse_host": "localhost"},
            {"warehouse_plugin": "warehouse_ros_mongo::MongoDatabaseConnection"},
        ],
        output="screen",
        condition=IfCondition(db_config),
    )
    
    return LaunchDescription([
        db_arg,
        static_tf,
        run_move_group_node,
        rviz_node,
        mongodb_server_node,
    ])