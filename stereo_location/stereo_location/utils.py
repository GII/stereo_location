import cv2
import math
import numpy as np

from filterpy.kalman import KalmanFilter
from rclpy.time import Time

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
        x, y = map(int, point)
        
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
    
    def length_pixels_to_meters(self, length_pixels, depth_m, angle_deg):
        """Convert pixel length to meters using the axis angle and camera's focal lengths."""
        angle_rad = math.radians(angle_deg)
        fx = self.fx
        fy = self.fy
        # Effective focal length for the given angle
        f_eff = math.sqrt((fx*math.cos(angle_rad))**2 + (fy * math.sin(angle_rad))**2)
        length_meters = (length_pixels * depth_m) / f_eff
        return length_meters


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


class PositionKalmanFilterTimestamped:

    def __init__(
        self,
        dim_x,
        dim_z,
        velocity_indices=None,   # indices in state vector that are velocities
        output_indices=None,      # indices in state vector to return as output
        process_noise=1e-4,
        measurement_noise=0.01,
        state_covariance=1000.,
    ):
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.kf.x = np.zeros(dim_x)
        self.kf.P *= state_covariance
        self.kf.H = np.hstack([np.eye(dim_z), np.zeros((dim_z, dim_x - dim_z))])
        self.kf.R = np.eye(dim_z) * measurement_noise
        self.kf.Q = np.eye(dim_x) * process_noise
        self.last_time = 0.
        self.initialized = False

        # Default: assume first half are positions, second half are velocities
        if velocity_indices is None:
            self.velocity_indices = list(range(dim_x // 2, dim_x))
        else:
            self.velocity_indices = velocity_indices

        # Default: output first dim_z states (positions)
        if output_indices is None:
            self.output_indices = list(range(dim_z))
        else:
            self.output_indices = output_indices

        # For F update: map each velocity index to its corresponding position index
        self.position_indices = [
            i - (self.velocity_indices[0] - self.output_indices[0])
            for i in self.velocity_indices
        ]

    def update_kalman_filter(self, measurement, timestamp):
        # Flatten the measurement to ensure it is a 1D array
        measurement_flat = measurement.flatten()
        current_time = Time.from_msg(timestamp).nanoseconds / 1e9  # Convert to seconds
        if not self.initialized:
            # Initialize the Kalman filter state with the first measurement
            self.kf.x[:len(measurement_flat)] = measurement_flat
            self.last_time = current_time
            self.initialized = True
            return self.kf.x[self.output_indices]
        dt = current_time - self.last_time
        self.last_time = current_time

        # Update F with current dt for position-velocity pairs
        F = np.eye(self.kf.dim_x)
        for pos_idx, vel_idx in zip(self.position_indices, self.velocity_indices):
            F[pos_idx, vel_idx] = dt
        self.kf.F = F

        self.kf.predict()
        self.kf.update(measurement_flat)

        # Return only the requested output indices
        return self.kf.x[self.output_indices]