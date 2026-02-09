import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, PathJoinSubstitution, FindExecutable
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ================= PATHS =================
    pkg_fanuc = get_package_share_directory('fanuc_description')
    pkg_world = get_package_share_directory('inspection_robot')
    pkg_moveit = get_package_share_directory("fanuc_moveit_config")

    world_path = os.path.join(pkg_world, 'worlds', 'inspection.world')

    # ================= ENVIRONMENT =================
    gazebo_resource_path = pkg_fanuc + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', '')

    set_resource = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=gazebo_resource_path
    )

    set_plugin_path = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib'
    )

    # ================= ROBOT DESCRIPTION WITH CAMERA =================
    robot_description = ParameterValue(
        Command([
            FindExecutable(name="xacro"),
            " ",
            PathJoinSubstitution([
                FindPackageShare("fanuc_description"),
                "urdf",
                "fanuc.urdf.xacro",
            ]),
            " use_camera:=true",      # IMPORTANT: Enable camera
            " use_gazebo:=true",      # IMPORTANT: Enable Gazebo sensor
        ]),
        value_type=str
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'publish_frequency': 50.0,  # Publish TF at 50Hz
        }],
    )

    # ================= START GAZEBO =================
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', world_path, '-v', '4'],
        output='screen'
    )

    # ================= SPAWN ROBOT =================
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot_description',
            '-name', 'fanuc',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.8'
        ],
        output='screen'
    )

    # ================= CONTROLLERS =================
    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
        output="screen",
    )

    trajectory_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["fanuc_controller", "-c", "/controller_manager"],
        output="screen",
    )

    # Delay controller spawning
    delayed_spawn = RegisterEventHandler(
        OnProcessStart(
            target_action=gazebo,
            on_start=[spawn_robot],
        )
    )
    
    delayed_controllers = RegisterEventHandler(
        OnProcessStart(
            target_action=spawn_robot,
            on_start=[
                joint_state_spawner,
                trajectory_spawner,
            ],
        )
    )

    # ================= CAMERA BRIDGE =================
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera_head/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/camera_head/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_head/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_head/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        remappings=[
            ('/camera_head/points', '/camera/depth/points'),
            ('/camera_head/image', '/camera/color/image_raw'),
            ('/camera_head/depth_image', '/camera/depth/image_raw'),
            ('/camera_head/camera_info', '/camera/depth/camera_info'),
        ],
        output='screen'
    )

    # ================= RETURN =================
    return LaunchDescription([
        set_resource,
        set_plugin_path,
        robot_state_pub,
        gazebo,
        delayed_spawn,
        delayed_controllers,
        camera_bridge,
    ])