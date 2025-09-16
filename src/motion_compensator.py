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
    
    def __init__(self, stabilization_factor=0.8):
        """
        Initialize motion compensator
        
        Args:
            stabilization_factor: How much to trust that target is stationary (0-1)
                                 Higher values provide more stability for stationary targets
                                 Lower values make the system more responsive to actual movement
        """
        self.stabilization_factor = stabilization_factor
        
        # Store previous positions and movements
        self.previous_target_pos = None
        self.previous_servo_movement = (0.0, 0.0)  # (pan_diff, tilt_diff) in degrees
        
        # Expected position shift from camera movement
        self.expected_position_shift = (0, 0)  # (x_shift, y_shift) in pixels
        
        # Conversion factors (will be calibrated)
        self.pan_to_pixel_factor = 20.0   # Estimated pixels per degree of pan
        self.tilt_to_pixel_factor = 20.0  # Estimated pixels per degree of tilt
        
        # Confidence in stationary targets
        self.stationary_confidence = 0.0  # 0.0-1.0 confidence that target is stationary
        self.consistency_buffer = []      # Buffer of recent position deltas
        self.max_buffer_size = 5
        
        # Activity detection
        self.movement_threshold = 5.0     # Degrees per second to consider "moving"
        self.stationary_frames = 0        # Counter for consecutive stationary frames
        
    def calibrate(self, frame_width, frame_height, fov_horizontal=60, fov_vertical=40):
        """
        Calibrate conversion factors based on camera field of view
        
        Args:
            frame_width: Width of camera frame in pixels
            frame_height: Height of camera frame in pixels
            fov_horizontal: Horizontal field of view in degrees
            fov_vertical: Vertical field of view in degrees
        """
        # Calculate pixels per degree
        self.pan_to_pixel_factor = frame_width / fov_horizontal
        self.tilt_to_pixel_factor = frame_height / fov_vertical
        
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
        
        # Update activity detection
        movement_magnitude = np.sqrt(pan_diff**2 + tilt_diff**2)
        if movement_magnitude < self.movement_threshold:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0
        
        # Update stationary confidence
        if self.stationary_frames > 10:  # After 10 frames of minimal movement
            self.stationary_confidence = min(1.0, self.stationary_confidence + 0.1)
        else:
            self.stationary_confidence = max(0.0, self.stationary_confidence - 0.2)
        
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
            return target_pos
            
        # Calculate observed position delta
        observed_delta_x = target_pos[0] - self.previous_target_pos[0]
        observed_delta_y = target_pos[1] - self.previous_target_pos[1]
        
        # Calculate expected position delta due to camera movement
        expected_delta_x = self.expected_position_shift[0]
        expected_delta_y = self.expected_position_shift[1]
        
        # Determine if observed movement matches expected movement from camera motion
        motion_match_factor = self._calculate_motion_match(
            (observed_delta_x, observed_delta_y), 
            (expected_delta_x, expected_delta_y)
        )
        
        # Balance between observed position and expected position
        # Higher stabilization_factor = more correction
        compensation_factor = self.stabilization_factor * motion_match_factor * self.stationary_confidence
        
        # Apply compensation
        compensated_x = int(target_pos[0] - (expected_delta_x * compensation_factor))
        compensated_y = int(target_pos[1] - (expected_delta_y * compensation_factor))
        
        # Update previous position for next frame
        self.previous_target_pos = target_pos
        
        # Update consistency buffer
        if len(self.consistency_buffer) >= self.max_buffer_size:
            self.consistency_buffer.pop(0)
        self.consistency_buffer.append((observed_delta_x - expected_delta_x, 
                                       observed_delta_y - expected_delta_y))
        
        return (compensated_x, compensated_y)
    
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