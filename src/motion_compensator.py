#!/usr/bin/env python3
"""
Motion Compensator for Pan-Tilt Camera System
Compensates for camera movement to prevent feedback loops in tracking
"""

import numpy as np
from typing import Tuple, Optional, List

class MotionCompensator:
    """
    Compensates for the camera's own movement to prevent feedback loops
    when tracking stationary targets
    """
    
    def __init__(self, stabilization_factor=0.95, debug=False):
        """
        Initialize motion compensator
        
        Args:
            stabilization_factor: How much to trust that target is stationary (0-1)
                                 Higher values provide more stability for stationary targets
                                 Lower values make the system more responsive to actual movement
            debug: Enable debug output
        """
        # Store configuration parameters
        self.debug = bool(debug)  # Ensure it's a boolean value
        self.stabilization_factor = stabilization_factor
        self.stabilization_factor = stabilization_factor
        
        # Store previous positions and movements
        self.previous_target_pos = None
        self.previous_servo_movement = (0.0, 0.0)  # (pan_diff, tilt_diff) in degrees
        
        # Expected position shift from camera movement
        self.expected_position_shift = (0, 0)  # (x_shift, y_shift) in pixels
        
        # Conversion factors (will be calibrated)
        self.pan_to_pixel_factor = 20.0   # Estimated pixels per degree of pan
        self.tilt_to_pixel_factor = 20.0  # Estimated pixels per degree of tilt
        
        # Add a deadband to ignore tiny movements (in pixels)
        self.base_deadband = 5.0          # Base deadband size
        self.position_deadband = 5.0      # Adaptive deadband, increases with confidence
        
        # Confidence in stationary targets
        self.stationary_confidence = 0.0  # 0.0-1.0 confidence that target is stationary
        self.consistency_buffer = []      # Buffer of recent position deltas
        self.max_buffer_size = 10         # Increased buffer size for better stability
        
        # Activity detection
        self.movement_threshold = 3.0     # Degrees per second to consider "moving" (reduced)
        self.stationary_frames = 0        # Counter for consecutive stationary frames
        
        # Tracking duration for adaptive stabilization
        self.tracking_duration = 0        # Number of frames spent tracking the same target
        self.max_tracking_duration = 90   # Maximum tracking duration (3 seconds at 30 fps)
        
        # Position history for trend analysis - reduced for better performance
        self.position_history = []
        self.max_history_size = 15        # 0.5 seconds at 30 fps - reduced for performance
        
    def calibrate(self, frame_width, frame_height, fov_horizontal=103, fov_vertical=None):
        """
        Calibrate conversion factors based on camera field of view
        
        The pan_to_pixel_factor and tilt_to_pixel_factor convert between servo angles and pixel
        coordinates in the image. These factors are essential for calculating how much the 
        image content will shift when the servos move by a certain angle.
        
        For example, if pan_to_pixel_factor = 18.6:
        - Moving the pan servo by 1 degree will shift the image content by ~18.6 pixels horizontally
        
        Args:
            frame_width: Width of camera frame in pixels (1920 for 1080p)
            frame_height: Height of camera frame in pixels (1080 for 1080p) 
            fov_horizontal: Horizontal field of view in degrees (default 103° for this camera)
            fov_vertical: Vertical field of view in degrees (computed from aspect ratio if None)
        """
        # For this camera: 103° horizontal FOV, 130° diagonal FOV
        # Calculate vertical FOV if not provided using the aspect ratio
        if fov_vertical is None:
            # Calculate vertical FOV using the aspect ratio and horizontal FOV
            aspect_ratio = frame_height / frame_width
            fov_vertical = fov_horizontal * aspect_ratio
            
        # Calculate pixels per degree (conversion factor)
        self.pan_to_pixel_factor = frame_width / fov_horizontal
        self.tilt_to_pixel_factor = frame_height / fov_vertical
        
        # Print calibration info if debug is enabled
        debug_enabled = getattr(self, 'debug', False)
        if debug_enabled:
            print(f"Motion Compensator Calibration:")
            print(f"  Camera FOV: {fov_horizontal}° horizontal, {fov_vertical:.1f}° vertical")
            print(f"  Frame size: {frame_width}x{frame_height} pixels")
            print(f"  Pan conversion: {self.pan_to_pixel_factor:.2f} pixels/degree")
            print(f"  Tilt conversion: {self.tilt_to_pixel_factor:.2f} pixels/degree")
        
    def update_servo_movement(self, pan_diff: float, tilt_diff: float):
        """
        Update the compensator with the latest servo movement
        
        Args:
            pan_diff: Change in pan angle in degrees
            tilt_diff: Change in tilt angle in degrees
        """
        self.previous_servo_movement = (pan_diff, tilt_diff)
        
        # Calculate expected pixel shift based on servo movement
        # Note: Direction is inverted because when camera moves right, 
        # objects in the frame appear to move left
        x_shift = -int(pan_diff * self.pan_to_pixel_factor)
        y_shift = -int(tilt_diff * self.tilt_to_pixel_factor)
        
        self.expected_position_shift = (x_shift, y_shift)
        
        # Update activity detection with reduced threshold
        movement_magnitude = np.sqrt(pan_diff**2 + tilt_diff**2)
        if movement_magnitude < self.movement_threshold:
            self.stationary_frames += 1
        else:
            # Reset more gradually when movement is detected
            self.stationary_frames = max(0, self.stationary_frames - 2)
        
        # Update stationary confidence with more aggressive buildup
        if self.stationary_frames > 5:  # After 5 frames of minimal movement
            self.stationary_confidence = min(1.0, self.stationary_confidence + 0.15)
        else:
            self.stationary_confidence = max(0.0, self.stationary_confidence - 0.1)
        
        # Update tracking duration (increased while tracking)
        self.tracking_duration = min(self.max_tracking_duration, self.tracking_duration + 1)
        
    def compensate(self, target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Compensate for camera motion to get true target position
        
        Args:
            target_pos: Current detected target position (x, y)
            
        Returns:
            Tuple[int, int]: Compensated target position
        """
        if self.previous_target_pos is None:
            # First frame - nothing to compensate
            self.previous_target_pos = target_pos
            
            # Initialize position history
            self.position_history.append(target_pos)
            return target_pos
            
        # Calculate observed position delta
        observed_delta_x = target_pos[0] - self.previous_target_pos[0]
        observed_delta_y = target_pos[1] - self.previous_target_pos[1]
        
        # Calculate adaptive deadband that increases with stationary confidence
        # This helps prevent oscillation when target is likely stationary
        adaptive_deadband = self.base_deadband * (1.0 + 3.0 * self.stationary_confidence)
        self.position_deadband = adaptive_deadband
        
        # Apply deadband to filter out noise
        if abs(observed_delta_x) < adaptive_deadband:
            observed_delta_x = 0
        if abs(observed_delta_y) < adaptive_deadband:
            observed_delta_y = 0
            
        # Calculate expected position delta due to camera movement
        expected_delta_x = self.expected_position_shift[0]
        expected_delta_y = self.expected_position_shift[1]
        
        # Determine if observed movement matches expected movement from camera motion
        motion_match_factor = self._calculate_motion_match(
            (observed_delta_x, observed_delta_y), 
            (expected_delta_x, expected_delta_y)
        )
        
        # Calculate adaptive stabilization factor based on tracking duration
        adaptive_factor = min(1.0, self.tracking_duration / (self.max_tracking_duration / 2))
        
        # Calculate position stability from position variance
        position_variance = self._calculate_position_variance()
        position_stability = max(0.0, 1.0 - min(1.0, position_variance / 20.0))
        
        # When target is very likely stationary (high confidence + low variance),
        # use a much higher compensation factor to eliminate oscillations completely
        if self.stationary_confidence > 0.8 and position_stability > 0.9:
            # Very stationary target - apply stronger stabilization
            compensation_factor = min(1.0, 0.95 + (0.05 * adaptive_factor))
        else:
            # Normal tracking - balance between observed position and expected position
            # Higher stabilization_factor = more correction
            # Include adaptive factor for longer tracking periods
            compensation_factor = (self.stabilization_factor * motion_match_factor * 
                                  self.stationary_confidence * (1 + 0.5 * adaptive_factor) *
                                  (1 + 0.3 * position_stability))
        
        # Limit compensation factor to valid range
        compensation_factor = min(1.0, max(0.0, compensation_factor))
        
        # Apply compensation
        compensated_x = int(target_pos[0] - (expected_delta_x * compensation_factor))
        compensated_y = int(target_pos[1] - (expected_delta_y * compensation_factor))
        
        # Add position smoothing for stationary targets
        if self.stationary_confidence > 0.5 and len(self.position_history) > 2:
            # Use stronger smoothing as confidence increases
            # Get more recent history for higher accuracy
            history_length = min(len(self.position_history), 
                                int(5 + (self.max_history_size - 5) * self.stationary_confidence))
            recent_positions = self.position_history[-history_length:]
            
            # Apply weighted average with history for additional stability
            # Weight increases with stationary confidence and tracking duration
            base_weight = self.stationary_confidence * 0.5
            duration_factor = min(1.0, self.tracking_duration / 30)  # 30 frames = 1 sec at 30fps
            history_weight = base_weight * (1 + 0.5 * duration_factor)
            
            # For very high confidence, use even stronger smoothing
            if self.stationary_confidence > 0.9:
                history_weight = min(0.9, history_weight + 0.2)
                
            recent_avg_x = sum(pos[0] for pos in recent_positions) / len(recent_positions)
            recent_avg_y = sum(pos[1] for pos in recent_positions) / len(recent_positions)
            
            compensated_x = int((1 - history_weight) * compensated_x + history_weight * recent_avg_x)
            compensated_y = int((1 - history_weight) * compensated_y + history_weight * recent_avg_y)
        
        # Update previous position for next frame
        self.previous_target_pos = target_pos
        
        # Update consistency buffer
        if len(self.consistency_buffer) >= self.max_buffer_size:
            self.consistency_buffer.pop(0)
        self.consistency_buffer.append((observed_delta_x - expected_delta_x, 
                                       observed_delta_y - expected_delta_y))
        
        # Update position history
        if len(self.position_history) >= self.max_history_size:
            self.position_history.pop(0)
        self.position_history.append((compensated_x, compensated_y))
        
        # Print debug information
        debug_enabled = getattr(self, 'debug', False)
        if debug_enabled and (self.tracking_duration % 10 == 0 or self.stationary_confidence > 0.8):
            position_variance = self._calculate_position_variance()
            print(f"Motion Comp: conf={self.stationary_confidence:.2f}, frames={self.stationary_frames}")
            print(f"  Deadband={adaptive_deadband:.1f}, Variance={position_variance:.2f}")
            print(f"  Expected shift: ({expected_delta_x:.1f}, {expected_delta_y:.1f})")
            print(f"  Observed shift: ({observed_delta_x:.1f}, {observed_delta_y:.1f})")
            print(f"  Comp factor: {compensation_factor:.2f}")
        
        return (compensated_x, compensated_y)
    
    def _calculate_position_variance(self):
        """
        Calculate the variance of recent position samples to detect target movement
        Optimized version that uses NumPy for faster computation
        
        Returns:
            float: Variance score of recent positions (higher = more movement)
        """
        # Need at least 3 positions for meaningful variance
        if len(self.position_history) < 3:
            return 0.0
            
        # Use fewer positions for performance (last 10 is sufficient)
        num_positions = min(10, len(self.position_history))
        positions = self.position_history[-num_positions:]
        
        # Convert to numpy array for faster computation
        try:
            # Fast path with numpy
            pos_array = np.array(positions)
            x_values = pos_array[:, 0]
            y_values = pos_array[:, 1]
            
            # Calculate variance
            x_variance = np.var(x_values)
            y_variance = np.var(y_values)
            
            # Combined variance score (sqrt of sum of variances)
            return np.sqrt(x_variance + y_variance)
        except:
            # Fall back to slower Python implementation
            x_values = [pos[0] for pos in positions]
            y_values = [pos[1] for pos in positions]
            
            # Calculate average position
            avg_x = sum(x_values) / len(x_values)
            avg_y = sum(y_values) / len(y_values)
            
            # Calculate variance
            x_variance = sum((x - avg_x)**2 for x in x_values) / len(x_values)
            y_variance = sum((y - avg_y)**2 for y in y_values) / len(y_values)
            
            return np.sqrt(x_variance + y_variance)
        
    def _calculate_motion_match(self, observed_delta, expected_delta):
        """
        Calculate how well the observed motion matches the expected motion from camera movement
        
        Returns:
            float: 0.0-1.0 indicating how likely the observed motion is due to camera movement
                 1.0 = observed motion exactly matches expected camera-induced motion
                 0.0 = observed motion is completely different from expected motion
        """
        # No expected motion
        if expected_delta[0] == 0 and expected_delta[1] == 0:
            return 0.0
            
        # Calculate magnitudes
        observed_mag = np.sqrt(observed_delta[0]**2 + observed_delta[1]**2)
        expected_mag = np.sqrt(expected_delta[0]**2 + expected_delta[1]**2)
        
        # No observed motion
        if observed_mag < 1.0:
            return 0.0
            
        # Calculate dot product to determine direction similarity
        if expected_mag > 0:
            dot_product = (observed_delta[0] * expected_delta[0] + 
                          observed_delta[1] * expected_delta[1]) / (observed_mag * expected_mag)
        else:
            dot_product = 0
            
        # Calculate magnitude similarity
        if observed_mag > expected_mag:
            mag_similarity = expected_mag / observed_mag
        else:
            mag_similarity = observed_mag / expected_mag
            
        # Combine direction and magnitude similarity
        # Higher values mean the observed motion is more likely due to camera movement
        return (dot_product * 0.7 + mag_similarity * 0.3)
    
    def reset(self):
        """Reset the compensator state"""
        self.previous_target_pos = None
        self.previous_servo_movement = (0.0, 0.0)
        self.expected_position_shift = (0, 0)
        self.stationary_confidence = 0.0
        self.consistency_buffer.clear()
        self.stationary_frames = 0
        self.tracking_duration = 0
        self.position_history.clear()