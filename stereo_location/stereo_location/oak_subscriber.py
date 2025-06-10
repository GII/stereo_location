#!/usr/bin/env python3

import rclpy
import traceback
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import message_filters
from threading import Lock
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default

from stereo_location.utils import SpatialCalculator



class OakSubscriber(Node):
    def __init__(self, node_name='oak_subscriber', **kwargs):
        super().__init__(node_name, **kwargs)
        
        # Initialize CV bridge
        self.bridge = CvBridge()

        self.namespace = self.get_namespace()
        if self.namespace == "/":
            self.namespace = ""

        self.camera_info_topic = f"{self.namespace}/oak/stereo/camera_info"
        self.rgb_topic = f"{self.namespace}/oak/rgb/image_rect"
        self.depth_topic = f"{self.namespace}/oak/stereo/image_raw"

        # Image center
        self.origin_x = None  # Will be set when first image arrives
        self.origin_y = None

        
        # Get camera info first
        self.camera_info = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )
        self.get_logger().info(f"Waiting for camera info on topic: {self.camera_info_topic}")
        
        # Initialize image synchronization (will be set up after camera info is received)
        self.rgb_sub = None
        self.depth_sub = None
        self.ts = None
        
        # Lock for thread safety
        self.lock = Lock()
        

    def camera_info_callback(self, msg):
        # Store camera info and set up spatial calculator with actual parameters
        self.camera_info = msg
        
        # Set up image subscribers after camera info is received
        if self.rgb_sub is None:
            qos = qos_profile_system_default
            qos.history = 0
            self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=qos)
            self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos)
            
            # Synchronize messages
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub], 10, 0.1)
            self.ts.registerCallback(self.image_callback)
            
            
            self.get_logger().info(f"Subscribed to RGB topic: {self.rgb_topic}")
            self.get_logger().info(f"Subscribed to Depth topic: {self.depth_topic}")

            # Initialize spatial calculator with camera info
            self.spatial_calc = SpatialCalculator(camera_info=self.camera_info, logger=self.get_logger())
            
        # Properly destroy the subscription to avoid getting more callbacks
        if self.camera_info_sub:
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None

    def image_callback(self, rgb_msg, depth_msg):
        try:
            # Convert ROS messages to OpenCV format
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            
            # Important: Ensure proper depth data format
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            # Obtain timestamp from the RGB message
            timestamp = rgb_msg.header.stamp 
            
            # Check depth encoding and convert if necessary
            if depth_msg.encoding == "16UC1" or depth_frame.dtype == np.uint16:
                # 16-bit depth is typically in millimeters already
                self.get_logger().debug('Depth data is 16-bit (millimeters).')
            else:
                self.get_logger().error(f'Unknown depth encoding: {depth_msg.encoding}, dtype: {depth_frame.dtype}')

            # Initialize origin position if not set
            if self.origin_x is None:
                self.origin_x = rgb_frame.shape[1] // 2
                self.origin_y = rgb_frame.shape[0] // 2
                
            # Process images with lock to avoid UI conflicts
            with self.lock:
                self.process_frames(rgb_frame, depth_frame, timestamp)
                
        except Exception as e:
            tb_str = traceback.format_exc()
            self.get_logger().error(f'Error processing images: {e}\n{tb_str}')

    def process_frames(self, rgb_frame, depth_frame):
        raise NotImplementedError("This method should be implemented in a subclass")
