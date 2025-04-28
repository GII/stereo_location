from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare the launch argument for the YAML file name
    yaml_file_arg = DeclareLaunchArgument(
        'yaml_file',
        default_value='cam_params.yaml',
        description='Name of the YAML configuration file to use'
    )

    # Get the package share directory for stereo_location and depthai_ros_driver
    stereo_location_share = FindPackageShare('stereo_location')
    depthai_share = FindPackageShare('depthai_ros_driver')

    # Define the configuration file path using the launch argument
    config_file = PathJoinSubstitution([
        stereo_location_share,
        'config',
        LaunchConfiguration('yaml_file')
    ])

    # Include the camera.launch.py from depthai_ros_driver
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                depthai_share,
                'launch',
                'camera.launch.py'
            ])
        ]),
        launch_arguments={
            'params_file': config_file
        }.items()
    )

    # Return the launch description
    return LaunchDescription([
        yaml_file_arg,
        camera_launch
    ])
