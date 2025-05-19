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

from stereo_location.utils import SpatialCalculator, TextHelper
from stereo_location.oak_subscriber import OakSubscriber


class StereoLocationNode(OakSubscriber):
    def __init__(self):
        super().__init__('stereo_location')
        
        self.text_helper = TextHelper()
    
        
        # Initialize UI parameters
        self.x = 0
        self.y = 0
        self.step = 5
        self.delta = 5
        self.rgb_weight = 0.4
        self.depth_weight = 0.6
        
        
        # Window name
        self.window_name = "RGB-Depth Overlay"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('RGB Weight %', self.window_name, int(self.rgb_weight*100), 100, self.update_blend_weights)
        
        self.get_logger().info('Stereo Location node has started')

    def update_blend_weights(self, percent_rgb):
        """Update the rgb and depth weights used to blend depth/rgb image"""
        with self.lock:
            self.rgb_weight = float(percent_rgb)/100.0
            self.depth_weight = 1.0 - self.rgb_weight

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
