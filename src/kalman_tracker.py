#!/usr/bin/env python3
"""
Kalman Filter Tracking Module for Pan-Tilt Camera
Implements a Kalman filter for smooth tracking of objects
"""

import numpy as np
import cv2


class KalmanTracker:
    """
    Kalman filter implementation for tracking objects in 2D space
    Tracks position (x, y) and velocity (dx/dt, dy/dt) for more accurate predictions
    """
    
    def __init__(self, process_noise=0.03, measurement_noise=0.1):
        """
        Initialize Kalman filter tracker
        
        Args:
            process_noise: Process noise covariance (how much we expect model state to change)
            measurement_noise: Measurement noise covariance (how much we trust measurements)
        """
        # Create Kalman filter with 4 dynamic parameters (x, y, dx/dt, dy/dt)
        # and 2 measurement parameters (x, y)
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Set measurement matrix - we only measure position, not velocity
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # x position
            [0, 1, 0, 0]   # y position
        ], np.float32)
        
        # Set transition matrix - includes position and velocity components
        # Add delta time (dt) factor to make the filter adapt to different frame rates
        # Each timestep: x' = x + dx/dt, y' = y + dy/dt
        # Velocity components remain constant (dx'/dt = dx/dt, dy'/dt = dy/dt)
        dt = 0.033  # Assuming ~30fps
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],  # x position update: x' = x + dx*dt
            [0, 1, 0, dt],  # y position update: y' = y + dy*dt
            [0, 0, 1, 0],   # x velocity remains constant: dx'/dt = dx/dt
            [0, 0, 0, 1]    # y velocity remains constant: dy'/dt = dy/dt
        ], np.float32)
        
        # Process noise - how much we expect the model to change
        # Higher values = more responsive but less smooth
        # Lower values = smoother but may lag behind quick changes
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise - how much we trust the measurements
        # Higher values = trust model more than measurements (smoother but may lag)
        # Lower values = trust measurements more than model (responsive but jittery)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Initialize state
        self.initialized = False
        self.last_prediction = None
        self.last_measurement = None
        self.frames_without_detection = 0
    
    def update(self, measurement):
        """
        Update filter with new measurement and return filtered position
        
        Args:
            measurement: Tuple (x, y) with current position measurement
            
        Returns:
            Tuple (x, y) with filtered position estimate
        """
        # Ensure measurement is valid and contains numeric values
        if not isinstance(measurement, tuple) or len(measurement) != 2:
            # Handle invalid measurement by returning the last prediction or None
            print(f"Warning: Invalid measurement format: {measurement}")
            return self.last_prediction if self.last_prediction else (0, 0)
            
        # Convert to numpy array format for Kalman filter
        measurement_array = np.array([[float(measurement[0])], [float(measurement[1])]], np.float32)
        self.last_measurement = measurement
        
        if not self.initialized:
            # First measurement - initialize the state
            self.kalman.statePre = np.array([
                [measurement[0]],  # x position
                [measurement[1]],  # y position
                [0],               # x velocity (initially 0)
                [0]                # y velocity (initially 0)
            ], np.float32)
            
            self.kalman.statePost = self.kalman.statePre.copy()
            self.initialized = True
            self.frames_without_detection = 0
            return measurement
        
        # First predict, then correct with measurement
        prediction = self.kalman.predict()
        self.last_prediction = (prediction[0, 0], prediction[1, 0])
        
        # Update with measurement
        corrected_state = self.kalman.correct(measurement_array)
        
        # Return corrected position
        filtered_pos = (corrected_state[0, 0], corrected_state[1, 0])
        self.frames_without_detection = 0
        
        return filtered_pos
    
    def predict(self):
        """
        Predict next position without measurement update
        Useful when object is temporarily occluded/not detected
        
        Returns:
            Tuple (x, y) with predicted position, (0,0) if not initialized
        """
        if not self.initialized:
            return (0, 0)
        
        # Predict next state
        prediction = self.kalman.predict()
        self.last_prediction = (float(prediction[0, 0]), float(prediction[1, 0]))
        self.frames_without_detection += 1
        
        # Return predicted position
        return self.last_prediction
    
    def get_velocity(self):
        """
        Get current estimated velocity
        
        Returns:
            Tuple (vx, vy) with velocity components, None if not initialized
        """
        if not self.initialized:
            return None
        
        state = self.kalman.statePost
        return (state[2, 0], state[3, 0])
    
    def reset(self):
        """Reset the tracker state"""
        self.initialized = False
        self.last_prediction = None
        self.last_measurement = None
        self.frames_without_detection = 0


def create_kalman_tracker(tracking_mode):
    """
    Factory function to create a Kalman tracker with appropriate settings for the tracking mode
    
    Args:
        tracking_mode: String 'surveillance' or 'turret' indicating tracking mode
    
    Returns:
        KalmanTracker instance with mode-appropriate settings
    """
    if tracking_mode == 'turret':
        # Turret mode - heavily prioritize stability over responsiveness
        # Very low process noise = high trust in model predictions (smooth movement)
        # Higher measurement noise = less trust in individual measurements (filter jitter)
        # This creates very strong smoothing effect:
        return KalmanTracker(process_noise=0.003, measurement_noise=0.2)
    else:
        # Surveillance mode - extremely smooth movement
        # Very low process noise = very high trust in model predictions
        # Very high measurement noise = very low trust in measurements (maximize smoothing)
        return KalmanTracker(process_noise=0.005, measurement_noise=0.25)