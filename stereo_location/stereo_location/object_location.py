import rclpy
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from copy import deepcopy
from math import radians

import cv2
from stereo_location.oak_subscriber import OakSubscriber
from stereo_location_interfaces.msg import ObjDet, ObjDetArray
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler

class ObjectLocationNode(OakSubscriber):
    def __init__(self):
        super().__init__('object_location')

        # Get the package share directory for stereo_location using rclpy
        stereo_location_share = get_package_share_directory('stereo_location')

        # Define the model file paths
        model_containers_path = Path(stereo_location_share) / 'config' / 'model_containers.pt'
        model_vegetables_path = Path(stereo_location_share) / 'config' / 'model_vegetables.pt'

        # Load the YOLO models
        self.model_containers = YOLO(str(model_containers_path))
        self.model_vegetables = YOLO(str(model_vegetables_path))
        # Set the model to inference mode
        self.model_containers.eval()
        self.model_vegetables.eval()
        self.names_containers = self.model_containers.names
        self.names_vegetables = self.model_vegetables.names

        # Broadcaster for tf
        self._br = TransformBroadcaster(self)

        # Message for publishing tf
        self.tf = TransformStamped()
        self.tf.header.frame_id = "oak_rgb_camera_optical_frame"
        self.tf.child_frame_id = "oak-d-base-frame"

        # Publisher for object detection messages and annotated images
        self.obj_det_pub = self.create_publisher(ObjDetArray, f"{self.namespace}/object_tracker/detections", 10)
        self.image_pub = self.create_publisher(Image, f"{self.namespace}/object_tracker/image", 10)

        
    def process_detections(self, results, names, depth_frame, rgb_frame, detections_list, segmentation=False):
        detections_counter = {}

        boxes = sorted(results[0].boxes, key=lambda x: x.conf[0], reverse=True)
        
        for detection in boxes:
            obj_msg = ObjDet()
            obj_msg.header.stamp = self.get_clock().now().to_msg()
            x1, y1, x2, y2 = roi = tuple(map(int, detection.xyxy[0]))  # Bounding box coordinates
            confidence = detection.conf[0]  # Confidence score
            class_id = int(detection.cls[0])  # Class ID
            class_name = names[class_id]
            if class_name not in detections_counter:
                detections_counter[class_name] = 0
            detections_counter[class_name] += 1
            obj_msg.header.frame_id = f"detection_{class_name}_{detections_counter[class_name]}"
            obj_msg.class_name = class_name
            obj_msg.confidence = float(confidence)
            if segmentation:
                segmentation_result = self.segment_object(rgb_frame, depth_frame, roi)
                spatials, _ = self.spatial_calc.calc_spatials(depth_frame, segmentation_result['centroid'])
                major_length = self.spatial_calc.length_pixels_to_meters(segmentation_result['major_axis'], spatials['z'], segmentation_result['orientation'])
                minor_length = self.spatial_calc.length_pixels_to_meters(segmentation_result['minor_axis'], spatials['z'], segmentation_result['orientation'])
                self.get_logger().debug(
                    f"Segmented {class_name} with fitted ellipse at "
                    f"centroid ({segmentation_result['centroid'][0]}, {segmentation_result['centroid'][1]}, "
                    f"Major axis: {major_length:.2f} m, "
                    f"Minor axis: {minor_length:.2f} m, "
                    f"Orientation: {segmentation_result['orientation']:.2f} degrees, "
                    f"{spatials['x']:.2f}, {spatials['y']:.2f}, {spatials['z']:.2f})"
                )
                obj_msg.bounding_box.x_min = float(segmentation_result['bbox'][0])
                obj_msg.bounding_box.y_min = float(segmentation_result['bbox'][1])
                obj_msg.bounding_box.x_max = float(segmentation_result['bbox'][2])
                obj_msg.bounding_box.y_max = float(segmentation_result['bbox'][3])
                obj_msg.position.x = float(spatials['x'])
                obj_msg.position.y = float(spatials['y'])
                obj_msg.position.z = float(spatials['z'])
                obj_msg.dimensions.orientation = float(segmentation_result['orientation'])
                obj_msg.dimensions.major_dim = float(major_length)
                obj_msg.dimensions.minor_dim = float(minor_length)
                # TODO: Add dimensions of the major and minor axes
                mask = segmentation_result['mask']
            else:
                spatials, _ = self.spatial_calc.calc_spatials(depth_frame, (np.mean([x1, x2]), np.mean([y1, y2])))
                obj_msg.bounding_box.x_min = float(x1)
                obj_msg.bounding_box.y_min = float(y1)
                obj_msg.bounding_box.x_max = float(x2)
                obj_msg.bounding_box.y_max = float(y2)
                obj_msg.position.x = float(spatials['x'])
                obj_msg.position.y = float(spatials['y'])
                obj_msg.position.z = float(spatials['z'])
                obj_msg.dimensions.orientation = float(segmentation_result['orientation'])
                obj_msg.dimensions.major_dim = float(major_length)
                obj_msg.dimensions.minor_dim = float(minor_length)
            

            detections_list.append({
                'obj_msg': obj_msg,
                'mask': mask if segmentation else None,
            })

    def segment_object(self, rgb_frame, depth_frame, roi):
        x1, y1, x2, y2 = roi
        # Extract the region of interest from the RGB frame
        roi_rgb = rgb_frame[y1:y2, x1:x2]

        # Convert ROI to HSV and extract the saturation channel
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # White regions have low saturation and high value (brightness)
        # Create a mask to keep only colored (non-white) regions
        sat_thresh = 30   # below this is considered "white" or grayish
        val_thresh = 100  # above this is considered "bright" (white)

        # Mask for low saturation (potentially white/gray)
        low_sat_mask = saturation < sat_thresh
        # Mask for high value (bright)
        value = hsv[:, :, 2]
        high_val_mask = value > val_thresh

        # Combine: white = low sat & high value
        white_mask = np.logical_and(low_sat_mask, high_val_mask).astype(np.uint8) * 255

        # Invert to keep only colored regions (not white)
        mask = cv2.bitwise_not(white_mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            vertical_length = y2 - y1
            horizontal_length = x2 - x1
            major_axis = max(vertical_length, horizontal_length)
            minor_axis = min(vertical_length, horizontal_length)
            orientation = 0.0 if horizontal_length >= vertical_length else 90.0
            return {
                'bbox': (x1, y1, x2, y2),
                'centroid': ((x1 + x2) // 2, (y1 + y2) // 2),
                'orientation': orientation,
                'mask': mask,
                'major_axis': major_axis,
                'minor_axis': minor_axis
                
            }

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box (relative to ROI, so add offsets)
        x, y, w, h = cv2.boundingRect(largest_contour)
        refined_bbox = (x1 + x, y1 + y, x1 + x + w, y1 + y + h)

        # Compute centroid
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00']) + x1
            cy = int(M['m01'] / M['m00']) + y1
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Compute orientation using PCA or fitEllipse if possible
        orientation = 0.0
        major_axis = 0.0
        minor_axis = 0.0
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (center_x, center_y), (axis1, axis2), angle = ellipse
            # Ensure major_axis >= minor_axis
            if axis1 >= axis2:
                major_axis = axis1
                minor_axis = axis2
                orientation = angle
            else:
                major_axis = axis2
                minor_axis = axis1
                orientation = (angle + 90.0) % 180.0  # Rotate by 90 degrees
            axis_ratio = minor_axis / major_axis
            if axis_ratio > 0.75:
                orientation = 0.0
        
        return {
            'bbox': refined_bbox,
            'centroid': (cx, cy),
            'orientation': orientation,
            'mask': mask,
            'major_axis': major_axis,
            'minor_axis': minor_axis
        }
    
    def annotate_image(self, rgb_frame, detections, draw_mask=False):
        """
        Annotate the image with bounding boxes, class/confidence, centroid, orientation axes,
        and optionally overlay the segmentation mask.
        """
        annotated = rgb_frame.copy()
        for detection in detections:
            obj_msg = detection['obj_msg']
            mask = detection.get('mask', None)
            # Bounding box
            x1 = int(obj_msg.bounding_box.x_min)
            y1 = int(obj_msg.bounding_box.y_min)
            x2 = int(obj_msg.bounding_box.x_max)
            y2 = int(obj_msg.bounding_box.y_max)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Class name and confidence
            label = f"{obj_msg.class_name}: {obj_msg.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Centroid
            cx = int(obj_msg.position.x_img) if hasattr(obj_msg.position, 'x_img') else int((x1 + x2) / 2)
            cy = int(obj_msg.position.y_img) if hasattr(obj_msg.position, 'y_img') else int((y1 + y2) / 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            # Orientation axes
            orientation = getattr(obj_msg.dimensions, 'orientation', 0.0)
            angle_rad = np.deg2rad(orientation)
            axis_length = 40
            perp_length = 15
            # Main axis
            x2_axis = int(cx + axis_length * np.cos(angle_rad))
            y2_axis = int(cy + axis_length * np.sin(angle_rad))
            cv2.line(annotated, (cx, cy), (x2_axis, y2_axis), (255, 0, 0), 2)
            # Perpendicular axis
            perp_angle = angle_rad + np.pi / 2
            x2_perp = int(cx + perp_length * np.cos(perp_angle))
            y2_perp = int(cy + perp_length * np.sin(perp_angle))
            cv2.line(annotated, (cx, cy), (x2_perp, y2_perp), (0, 255, 255), 2)
            # Overlay mask if requested and available
            if draw_mask and mask is not None:
                mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                mask_color = np.zeros_like(annotated[y1:y2, x1:x2])
                mask_color[mask_resized > 0] = (0, 128, 255)
                overlay = cv2.addWeighted(annotated[y1:y2, x1:x2], 0.3, mask_color, 0.7, 0)
                annotated[y1:y2, x1:x2] = overlay
        return annotated

    def process_frames(self, rgb_frame, depth_frame):
        result_containers = self.model_containers.predict(rgb_frame, stream=False, conf=0.5)
        result_vegetables = self.model_vegetables.predict(rgb_frame, stream=False, conf=0.5)
        detections = []

        # Process detections for containers and vegetables
        self.process_detections(result_containers, self.names_containers, depth_frame, rgb_frame, detections)
        self.process_detections(result_vegetables, self.names_vegetables, depth_frame, rgb_frame, detections, segmentation=True)

        # Obtain annotated image
        annotated_image = self.annotate_image(rgb_frame, detections, draw_mask=True)
        
        # Publish annotated image
        annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        annotated_image_msg.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(annotated_image_msg)

        # Publish object detections
        
        self.tf.header.stamp = self.get_clock().now().to_msg()
        detection_msgs = [detection['obj_msg'] for detection in detections]
        for detection in detection_msgs:
            self.tf.child_frame_id = detection.header.frame_id
            self.tf.transform.translation.x = detection.position.x
            self.tf.transform.translation.y = detection.position.y
            self.tf.transform.translation.z = detection.position.z
            orient = quaternion_from_euler(radians(detection.dimensions.orientation), 0.0, 0.0, axes="rzyx")
            self.tf.transform.rotation.x = orient[0]
            self.tf.transform.rotation.y = orient[1]
            self.tf.transform.rotation.z = orient[2]
            self.tf.transform.rotation.w = orient[3]
            self._br.sendTransform(self.tf)
            
            self.get_logger().info(
                f"Detected {detection.class_name} with confidence {detection.confidence:.2f} at "
                f"bbox ({detection.bounding_box.x_min}, {detection.bounding_box.y_min}, "
                f"{detection.bounding_box.x_max}, {detection.bounding_box.y_max}) with position "
                f"({detection.position.x:.2f}, {detection.position.y:.2f}, {detection.position.z:.2f})"
            )
        obj_det_array = ObjDetArray()
        obj_det_array.header.stamp = self.get_clock().now().to_msg()
        obj_det_array.header.frame_id = "oak_rgb_camera_optical_frame"
        obj_det_array.objects = detection_msgs
        self.obj_det_pub.publish(obj_det_array)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectLocationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()