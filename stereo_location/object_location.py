import rclpy
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

from stereo_location.oak_subscriber import OakSubscriber
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


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

        # Messge for publishing tf
        self.tf = TransformStamped()
        self.tf.header.frame_id = "oak_rgb_camera_optical_frame"
        self.tf.child_frame_id = "oak-d-base-frame"
        

    def process_detections(self, results, names, depth_frame, detections_list):
        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
            confidence = detection.conf[0]  # Confidence score
            class_id = int(detection.cls[0])  # Class ID
            class_name = names[class_id]
            spatials, _ = self.spatial_calc.calc_spatials(depth_frame, (np.mean([x1, x2]), np.mean([y1, y2])))
            detections_list.append({
                'class_name': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'spatials': spatials
            })

    def process_frames(self, rgb_frame, depth_frame):
        result_containers = self.model_containers.predict(rgb_frame, stream=False, conf=0.5)
        result_vegetables = self.model_vegetables.predict(rgb_frame, stream=False, conf=0.5)
        detections = []

        # Process detections for containers
        self.process_detections(result_containers, self.names_containers, depth_frame, detections)

        # Process detections for vegetables
        self.process_detections(result_vegetables, self.names_vegetables, depth_frame, detections)

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        detections_counter = {}
        self.tf.header.stamp = self.get_clock().now().to_msg()
        for detection in detections:
            class_name = detection['class_name']
            if class_name not in detections_counter:
                detections_counter[class_name] = 0
            detections_counter[class_name] += 1

            self.tf.child_frame_id = f"detection_{detection['class_name']}_{detections_counter[class_name]}"
            self.tf.transform.translation.x = detection['spatials']['x']
            self.tf.transform.translation.y = detection['spatials']['y']
            self.tf.transform.translation.z = detection['spatials']['z']
            self._br.sendTransform(self.tf)
            
            self.get_logger().info(
                f"Detected {detection['class_name']} with confidence {detection['confidence']:.2f} at "
                f"bbox {detection['bbox']} with spatials {detection['spatials']}"
            )

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