#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import message_filters
from threading import Lock

class SpatialCalculator:
    def __init__(self, camera_info=None, correction_factor=0.924836601, logger=None):
        self.delta_roi = 5
        self.correction_factor = correction_factor
        self.logger = logger
        
        # Default camera matrix if none provided
        if camera_info is None:
            # Default values
            self.fx = 762.4027099609375
            self.fy = 761.9955444335938
            self.cx = 650.5501708984375
            self.cy = 353.3233642578125
        else:
            # Extract from camera info
            self.fx = camera_info.k[0]
            self.fy = camera_info.k[4]
            self.cx = camera_info.k[2]
            self.cy = camera_info.k[5]
        
        # Initialize frame counter for logging
        self.frame_counter = 0

    def setDeltaRoi(self, delta):
        self.delta_roi = delta

    def calc_spatials(self, depth_data, point, averaging_method=np.median, log_stats=False):
        """Calculate spatial coordinates from depth data at specified point"""
        x, y = point
        
        # Create ROI around the center point
        roi_x_min = max(0, x - self.delta_roi)
        roi_x_max = min(depth_data.shape[1], x + self.delta_roi)
        roi_y_min = max(0, y - self.delta_roi)
        roi_y_max = min(depth_data.shape[0], y + self.delta_roi)
        
        # Extract ROI from depth data
        roi = depth_data[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        # Filter out zero values (no data) for more accurate depth calculation
        valid_roi = roi[roi > 0] if roi.size > 0 else np.array([])
        
        # Log depth statistics if requested
        if log_stats:
            self.frame_counter += 1
            if self.frame_counter % 30 == 0:
                # Get statistics from valid depth values in the ROI only
                if valid_roi.size > 0:
                    depth_min = np.min(valid_roi)
                    depth_max = np.max(valid_roi)
                    depth_mean = np.mean(valid_roi)
                    
                    # Log using the provided logger if available
                    if self.logger:
                        self.logger.info(
                            f"ROI Depth stats - min: {depth_min:.2f}, max: {depth_max:.2f}, mean: {depth_mean:.2f}, dtype: {roi.dtype}, size: {valid_roi.size}")
                    else:
                        # Fallback to standard logging if no logger provided
                        print(f"ROI Depth stats - min: {depth_min:.2f}, max: {depth_max:.2f}, mean: {depth_mean:.2f}, dtype: {roi.dtype}, size: {valid_roi.size}")
        
        
        
        # Calculate average depth within ROI
        if valid_roi.size > 0:
            depth_mm = averaging_method(valid_roi) * self.correction_factor
        else:
            depth_mm = np.nan
            
        # Convert depth from mm to meters
        depth_m = depth_mm / 1000.0

        # Calculate 3D coordinates using pinhole camera model
        # X = (x - cx) * Z / fx, Y = (y - cy) * Z / fy
        X = (x - self.cx) * depth_m / self.fx
        Y = (y - self.cy) * depth_m / self.fy
        
        result = {
            'z': depth_m,   # depth in m
            'x': X,         # x in meters
            'y': Y          # y in meters
        }
        
        return result, (x, y)


class TextHelper:
    def __init__(self, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(255, 255, 255)):
        self.font_face = font_face
        self.font_scale = font_scale
        self.color = color
        self.thickness = 1

    def putText(self, frame, text, position):
        cv2.putText(frame, text, position, self.font_face, self.font_scale, self.color, self.thickness)
    
    def rectangle(self, frame, pt1, pt2, color=(70, 255, 70)):
        cv2.rectangle(frame, pt1, pt2, color, 2)
        
    def point(self, frame, position, size, color=(70, 255, 70)):
        cv2.circle(frame, position, size, color, 2)


class StereoLocationNode(Node):
    def __init__(self):
        super().__init__('stereo_location')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Get camera info first
        self.camera_info = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/oak/stereo/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Initialize spatial calculator with default values first and pass the logger
        self.spatial_calc = SpatialCalculator(logger=self.get_logger())
        self.text_helper = TextHelper()
        
        # Initialize image synchronization (will be set up after camera info is received)
        self.rgb_sub = None
        self.depth_sub = None
        self.ts = None
        
        # Initialize UI parameters
        self.x = 0
        self.y = 0
        self.origin_x = None  # Will be set when first image arrives
        self.origin_y = None
        self.step = 5
        self.delta = 5
        self.rgb_weight = 0.4
        self.depth_weight = 0.6
        
        # Lock for thread safety
        self.lock = Lock()
        
        # Window name
        self.window_name = "RGB-Depth Overlay"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('RGB Weight %', self.window_name, int(self.rgb_weight*100), 100, self.update_blend_weights)
        
        self.get_logger().info('Stereo Location node has started')

    def camera_info_callback(self, msg):
        # Store camera info and set up spatial calculator with actual parameters
        self.camera_info = msg
        self.spatial_calc = SpatialCalculator(camera_info=msg, logger=self.get_logger())
        
        # Set up image subscribers after camera info is received
        if self.rgb_sub is None:
            self.rgb_sub = message_filters.Subscriber(self, Image, '/oak/rgb/image_rect')
            self.depth_sub = message_filters.Subscriber(self, Image, '/oak/stereo/image_raw')
            
            # Synchronize messages
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub], 10, 0.1)
            self.ts.registerCallback(self.image_callback)
            
            self.get_logger().info('Camera info received, processing images')
        
        # Properly destroy the subscription to avoid getting more callbacks
        if self.camera_info_sub:
            self.destroy_subscription(self.camera_info_sub)
            self.camera_info_sub = None

    def update_blend_weights(self, percent_rgb):
        """Update the rgb and depth weights used to blend depth/rgb image"""
        with self.lock:
            self.rgb_weight = float(percent_rgb)/100.0
            self.depth_weight = 1.0 - self.rgb_weight

    def image_callback(self, rgb_msg, depth_msg):
        try:
            # Convert ROS messages to OpenCV format
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            
            # Important: Ensure proper depth data format
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            
            # Check depth encoding and convert if necessary
            if depth_msg.encoding == "8UC1" or depth_frame.dtype == np.uint8:
                self.get_logger().debug('Depth data is 8-bit. Converting to millimeters.')
                # If data is 8-bit, scale to appropriate range (this is an approximation)
                # You may need to adjust this scaling factor based on your camera
                depth_frame = depth_frame.astype(np.float32) * 10.0  # Assuming 1 unit = 10mm
            elif depth_msg.encoding == "16UC1" or depth_frame.dtype == np.uint16:
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
                self.process_frames(rgb_frame, depth_frame)
                
        except Exception as e:
            self.get_logger().error(f'Error processing images: {e}')

    def process_frames(self, rgb_frame, depth_frame):
        # Normalize and colorize depth image for visualization
        norm_depth = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        norm_depth = cv2.equalizeHist(norm_depth)
        colored_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_HOT)
        
        # Calculate spatial coordinates
        point_x = self.x + self.origin_x
        point_y = self.y + self.origin_y
        
        # Log stats only for the interest point, not for the origin
        spatials, _ = self.spatial_calc.calc_spatials(depth_frame, (point_x, point_y), log_stats=True)
        orig_spatials, _ = self.spatial_calc.calc_spatials(depth_frame, (self.origin_x, self.origin_y), log_stats=False)
        
        # Calculate relative coordinates
        spat_x = (spatials['x'] - orig_spatials['x'])
        spat_y = (spatials['y'] - orig_spatials['y'])
        spat_z = (spatials['z'] - orig_spatials['z'])
        
        # Draw visualization elements
        self.text_helper.rectangle(rgb_frame, 
                              (point_x - self.delta, point_y - self.delta), 
                              (point_x + self.delta, point_y + self.delta))
        
        # Display coordinates
        self.text_helper.putText(rgb_frame, "X: " + ("{:.3f}m".format(spat_x) if not math.isnan(spat_x) else "--"), 
                            (point_x + 15, point_y + 20))
        self.text_helper.putText(rgb_frame, "Y: " + ("{:.3f}m".format(spat_y) if not math.isnan(spat_y) else "--"), 
                            (point_x + 15, point_y + 45))
        self.text_helper.putText(rgb_frame, "Z: " + ("{:.3f}m".format(spat_z) if not math.isnan(spat_z) else "--"), 
                            (point_x + 15, point_y + 70))
        
        # Mark origin
        self.text_helper.point(rgb_frame, (self.origin_x, self.origin_y), 10)
        
        # Blend images
        blended = cv2.addWeighted(rgb_frame, self.rgb_weight, colored_depth, self.depth_weight, 0)
        
        # Display result
        cv2.imshow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, blended)
        self.process_key(cv2.waitKey(1))

    def process_key(self, key):
        if key == ord('q'):
            self.get_logger().info('Shutting down')
            rclpy.shutdown()
        elif key == ord('w'):
            self.y -= self.step
        elif key == ord('a'):
            self.x -= self.step
        elif key == ord('s'):
            self.y += self.step
        elif key == ord('d'):
            self.x += self.step
        elif key == ord('r'):  # Increase Delta
            if self.delta < 50:
                self.delta += 1
                self.spatial_calc.setDeltaRoi(self.delta)
        elif key == ord('f'):  # Decrease Delta
            if 3 < self.delta:
                self.delta -= 1
                self.spatial_calc.setDeltaRoi(self.delta)
        elif key == ord('z'):  # Set origin
            self.origin_x = self.x + self.origin_x
            self.origin_y = self.y + self.origin_y
            self.x = 0
            self.y = 0


def main(args=None):
    rclpy.init(args=args)
    node = StereoLocationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
