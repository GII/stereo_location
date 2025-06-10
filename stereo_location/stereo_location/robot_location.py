import rclpy
import numpy as np
import cv2
import tf_transformations as tft
from collections import deque
from rclpy.time import Time


from tf2_ros.transform_broadcaster import TransformBroadcaster
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from stereo_location.oak_subscriber import OakSubscriber
from stereo_location.utils import SpatialCalculator
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_matrix, euler_from_quaternion
from sensor_msgs.msg import Image
from stereo_location.utils import PositionKalmanFilterTimestamped

class RobotLocationNode(OakSubscriber):
    def __init__(self):
        super().__init__('robot_location')

        self._br = TransformBroadcaster(self)

        # Define the dictionary and parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        # Create the ArUco detector
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        self.tf = TransformStamped()
        self.tf.header.frame_id = "torso_lift_link"
        self.tf.child_frame_id = "oak-d-base-frame"

        self.aruco_tag = 42  # ID of the ArUco marker to detect

        # Publisher for robot location annotated image
        self.image_pub = self.create_publisher(Image, f"{self.namespace}/robot_tracker/image", 10)

        self.kf = PositionKalmanFilterTimestamped(
            dim_x=6,  # State vector size: [x, y, z, vx, vy, vz]
            dim_z=3,  # Measurement vector size: [x, y, z]
            process_noise=1e-4,
            measurement_noise=0.01,
            state_covariance=1000.0,
        )

    def camera_info_callback(self, msg):
        super().camera_info_callback(msg)
        self.camera_matrix = np.array(self.camera_info.k).reshape(3, 3)
        marker_size = 0.15  # Size of the marker in meters
        self.object_points = np.array([
            [-marker_size / 2,  marker_size / 2, 0],  # Point 0: Top-left corner
            [ marker_size / 2,  marker_size / 2, 0],  # Point 1: Top-right corner
            [ marker_size / 2, -marker_size / 2, 0],  # Point 2: Bottom-right corner
            [-marker_size / 2, -marker_size / 2, 0]   # Point 3: Bottom-left corner
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)



    def obtain_transformation(self, rvec, tvec):
        # Convert rotation vector to a rotation matrix
        R_aruco_optical, _ = cv2.Rodrigues(rvec)
        R_aruco_optical[2, :] = [0, 0, -1] # Fix z axis orientation 

        # Transformation from optical frame to camera frame TODO: Get this from tf
        R_camera_optical = np.array([
            [0, 0,  1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        # Rotation to align with the robot's frame
        R_aruco_tiago = np.array([
            [-1,  0, 0],
            [ 0, -1, 0],
            [ 0,  0, 1]
        ])

        # Tiago frame relative to the aruco frame
        T_tiago_aruco = np.eye(4)
        T_tiago_aruco[:3, :3] = R_aruco_tiago
        T_tiago_aruco[:3, 3] = np.array([0, 0, 0])

        # Optical frame relative to the aruco frame
        T_aruco_optical = np.eye(4)
        T_aruco_optical[:3, :3] = R_aruco_optical
        T_aruco_optical[:3, 3] = tvec.flatten()

        # Camera frame relative to the optical frame
        T_camera_optical = np.eye(4)
        T_camera_optical[:3, :3] = R_camera_optical
        T_camera_optical[:3, 3] = np.array([0, 0, 0])

        # Combine transformations
        T_aruco_camera = T_camera_optical @ T_aruco_optical

        # Invert transformation to obtain camera relative to aruco
        T_camera_aruco = np.eye(4)
        T_camera_aruco[:3, :3] = T_aruco_camera[:3, :3].T
        T_camera_aruco[:3, 3] = -(T_aruco_camera[:3, :3].T @ T_aruco_camera[:3, 3]).flatten()

        T_camera_tiago = T_tiago_aruco @ T_camera_aruco
        t_camera_tiago = T_camera_tiago[:3, 3]

        # Extract quaternion from the final rotation matrix
        original_quaternion = quaternion_from_matrix(
            np.vstack((np.hstack((R_aruco_optical, [[0], [0], [0]])), [0, 0, 0, 1]))
        )
        quaternion = quaternion_from_matrix(T_camera_tiago)

        return {
            "tvec": tvec.flatten(),
            "t_camera_tiago": t_camera_tiago,
            "original_quaternion": original_quaternion,
            "quaternion": quaternion
        }

    def publish_transformation(self, t_camera_tiago, quaternion):
        # Update the TransformStamped message
        self.tf.transform.translation.x = t_camera_tiago[0]
        self.tf.transform.translation.y = t_camera_tiago[1]
        self.tf.transform.translation.z = t_camera_tiago[2]
        self.tf.transform.rotation.x = quaternion[0]
        self.tf.transform.rotation.y = quaternion[1]
        self.tf.transform.rotation.z = quaternion[2]
        self.tf.transform.rotation.w = quaternion[3]

        # Publish the transform
        self.tf.header.stamp = self.get_clock().now().to_msg()
        self._br.sendTransform(self.tf)

    def process_frames(self, rgb_frame, depth_frame, timestamp):
        # Check the number of channels in the rgb_frame
        if len(rgb_frame.shape) == 2:  # Grayscale rgb_frame
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2BGR)
        elif rgb_frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        elif rgb_frame.shape[2] == 4:  # RGBA rgb_frame
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGBA2BGR)

        # Convert the rgb_frame to grayscale
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, rejected = self.detector.detectMarkers(gray)

        if ids is not None:
            corners = corners_list[0]
            self.get_logger().info(f"-----------------------------------------------------------------------")
            self.get_logger().info(f"Detected marker ID: {ids[0]}")
            
            if int(ids[0]) == self.aruco_tag:
                # Solve PnP to get the pose of the marker
                success, rvec, tvec = cv2.solvePnP(
                    self.object_points,  # 3D points in the real world
                    corners[0],      # 2D points in the rgb_frame
                    self.camera_matrix,  # Camera intrinsic matrix
                    self.dist_coeffs,      # Distortion coefficients
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if success:
                    tf_data = self.obtain_transformation(rvec, tvec)

                    # Update the Kalman filter with the measured position
                    filtered_transform = self.kf.update_kalman_filter(tf_data['t_camera_tiago'], timestamp)

                    # Log the transformations for debugging
                    self.get_logger().debug(f"Position (optical --> aruco) {tf_data['tvec']}")
                    self.get_logger().debug(f"Rotation (optical --> aruco) {euler_from_quaternion(tf_data['original_quaternion'])}")
                    self.get_logger().debug(f"Position (tiago --> camera) {tf_data['t_camera_tiago']}")
                    self.get_logger().debug(f"Rotation (tiago --> camera) {euler_from_quaternion(tf_data['quaternion'])}")
                    self.publish_transformation(filtered_transform, tf_data['quaternion'])

def main(args=None):
    rclpy.init(args=args)
    node = RobotLocationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()