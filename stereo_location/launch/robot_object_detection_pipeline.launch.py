import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler, Shutdown, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    FindExecutable,
    PathJoinSubstitution,
)



def launch_setup(context: LaunchContext, *args, **kwargs):
    depthai_prefix = get_package_share_directory("depthai_ros_driver")

    name = LaunchConfiguration("name")
    camera_model = LaunchConfiguration("camera_model")
    parent_frame = LaunchConfiguration("parent_frame")
    cam_pos_x = LaunchConfiguration("cam_pos_x")
    cam_pos_y = LaunchConfiguration("cam_pos_y")
    cam_pos_z = LaunchConfiguration("cam_pos_z")
    cam_roll = LaunchConfiguration("cam_roll")
    cam_pitch = LaunchConfiguration("cam_pitch")
    cam_yaw = LaunchConfiguration("cam_yaw")
    params_file = LaunchConfiguration("params_file")
    use_rviz = LaunchConfiguration("use_rviz")
    rectify_rgb = LaunchConfiguration("rectify_rgb")
    rs_compat = LaunchConfiguration("rs_compat")
    namespace = LaunchConfiguration("namespace")

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(depthai_prefix, "launch", "camera.launch.py")
        ),
        launch_arguments={
            "name": name,
            "camera_model": camera_model,
            "parent_frame": parent_frame,
            "cam_pos_x": cam_pos_x,
            "cam_pos_y": cam_pos_y,
            "cam_pos_z": cam_pos_z,
            "cam_roll": cam_roll,
            "cam_pitch": cam_pitch,
            "cam_yaw": cam_yaw,
            "params_file": params_file,
            "use_rviz": use_rviz,
            "rectify_rgb": rectify_rgb,
            "pointcloud.enable": "false",
            "rs_compat": rs_compat,
            "namespace": namespace,
        }.items(),
    )

    object_detection = Node(
        package="stereo_location",
        executable="object_location",
        output="screen",
        namespace=namespace,
        arguments=[],
        parameters=[{}],
    )

    robot_detection = Node(
        package="stereo_location",
        executable="robot_location",
        output="screen",
        namespace=namespace,
        arguments=[],
        parameters=[{}],
    )


    nodes_to_start = [
        camera_launch,
        object_detection,
        robot_detection,
    ]

    return nodes_to_start


def generate_launch_description():

    stereo_location_prefix = get_package_share_directory("stereo_location")

    declared_arguments = []

    declared_arguments.append(DeclareLaunchArgument("name", default_value="oak"))
    declared_arguments.append(DeclareLaunchArgument("camera_model", default_value="OAK-D"))
    declared_arguments.append(DeclareLaunchArgument("parent_frame", default_value="oak-d-base-frame"))
    declared_arguments.append(DeclareLaunchArgument("cam_pos_x", default_value="0.0"))
    declared_arguments.append(DeclareLaunchArgument("cam_pos_y", default_value="0.0"))
    declared_arguments.append(DeclareLaunchArgument("cam_pos_z", default_value="0.0"))
    declared_arguments.append(DeclareLaunchArgument("cam_roll", default_value="0.0"))
    declared_arguments.append(DeclareLaunchArgument("cam_pitch", default_value="0.0"))
    declared_arguments.append(DeclareLaunchArgument("cam_yaw", default_value="0.0"))
    declared_arguments.append(
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(stereo_location_prefix, "config", "cam_params.yaml"),
        )
    )
    declared_arguments.append(DeclareLaunchArgument("use_rviz", default_value="False"))
    declared_arguments.append(DeclareLaunchArgument("rectify_rgb", default_value="True"))
    declared_arguments.append(DeclareLaunchArgument("rs_compat", default_value="False"))
    declared_arguments.append(DeclareLaunchArgument("namespace", default_value="/cam"))

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )