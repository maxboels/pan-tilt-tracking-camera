#!/usr/bin/env python3
"""
Advanced Tracking Logger for Pan-Tilt Tracking Camera

This module provides comprehensive logging functionality to track performance metrics
and create a consistent evaluation protocol for the pan-tilt tracking camera system.

It logs:
- Timestamp for each frame
- Camera center point coordinates
- Detected object bounding box coordinates
- Center point of detected objects
- Servo commands sent at each timestep
- Distance error between target object and camera center
- System performance metrics (FPS, latency)
"""

import os
import time
import csv
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any


class TrackingLogger:
    """Advanced tracking logger for Pan-Tilt Tracking Camera system"""
    
    def __init__(self, log_dir="logs", experiment_name=None):
        """
        Initialize the tracking logger
        
        Args:
            log_dir: Directory where log files will be stored
            experiment_name: Optional name for the experiment (will be used for subfolder)
        """
        # Create logs directory if it doesn't exist
        self.base_log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate a unique timestamp for this experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment name (use provided name or default to timestamp)
        if experiment_name:
            self.experiment_name = f"{experiment_name}_{timestamp}"
        else:
            self.experiment_name = f"experiment_{timestamp}"
        
        # Create experiment subfolder
        self.log_dir = os.path.join(self.base_log_dir, self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log filename within the experiment subfolder
        self.log_filename = "tracking_data.log"
        self.log_path = os.path.join(self.log_dir, self.log_filename)
        
        # Create CSV writer for the log file
        self.fieldnames = [
            'timestamp',                 # Unix timestamp
            'frame_number',              # Sequential frame counter
            'camera_center_x',           # Camera center X coordinate
            'camera_center_y',           # Camera center Y coordinate
            'target_detected',           # Boolean: was target detected in this frame
            'target_class',              # Class name of the detected target
            'target_confidence',         # Confidence score of the detection
            'bbox_x1',                   # Bounding box top-left X
            'bbox_y1',                   # Bounding box top-left Y
            'bbox_x2',                   # Bounding box bottom-right X
            'bbox_y2',                   # Bounding box bottom-right Y
            'target_center_x',           # Target center X coordinate
            'target_center_y',           # Target center Y coordinate
            'smoothed_target_x',         # Smoothed target position X
            'smoothed_target_y',         # Smoothed target position Y
            'pixel_error_x',             # X distance error in pixels
            'pixel_error_y',             # Y distance error in pixels
            'distance_error',            # Euclidean distance error in pixels
            'current_pan',               # Current pan angle in degrees
            'current_tilt',              # Current tilt angle in degrees
            'target_pan',                # Target pan angle in degrees
            'target_tilt',               # Target tilt angle in degrees
            'servo_command_pan',         # Pan command sent to servos
            'servo_command_tilt',        # Tilt command sent to servos
            'processing_time_ms',        # Frame processing time in milliseconds
            'tracking_enabled',          # Boolean: is tracking active
        ]
        
        # Initialize the log file with headers
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        
        # Frame counter
        self.frame_number = 0
        
        # Record system configuration
        self.record_system_config()
        
        print(f"Tracking logger initialized. Logging to {self.log_path}")
    
    def record_system_config(self):
        """Record system configuration in a separate JSON file"""
        config_filename = "system_config.json"
        config_path = os.path.join(self.log_dir, config_filename)
        
        # Try to load existing configuration
        system_config = {}
        try:
            # Try to load main config
            if os.path.exists("config/config.json"):
                with open("config/config.json", 'r') as f:
                    system_config["app_config"] = json.load(f)
            
            # Try to load calibration data
            if os.path.exists("config/calibration.json"):
                with open("config/calibration.json", 'r') as f:
                    system_config["calibration"] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load system configuration: {e}")
        
        # Add timestamp and system info
        system_config["timestamp"] = time.time()
        system_config["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_config["log_file"] = self.log_filename
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(system_config, f, indent=2)
    
    def log_frame(self, 
                  frame_data: Dict[str, Any],
                  processing_time_ms: float = 0.0):
        """
        Log a single frame of tracking data
        
        Args:
            frame_data: Dictionary containing all frame tracking data
            processing_time_ms: Processing time for this frame in milliseconds
        """
        # Increment frame counter
        self.frame_number += 1
        
        # Prepare log row with defaults
        log_row = {
            'timestamp': time.time(),
            'frame_number': self.frame_number,
            'camera_center_x': frame_data.get('camera_center_x', 0),
            'camera_center_y': frame_data.get('camera_center_y', 0),
            'target_detected': frame_data.get('target_detected', False),
            'target_class': frame_data.get('target_class', ''),
            'target_confidence': frame_data.get('target_confidence', 0.0),
            'bbox_x1': frame_data.get('bbox_x1', 0),
            'bbox_y1': frame_data.get('bbox_y1', 0),
            'bbox_x2': frame_data.get('bbox_x2', 0),
            'bbox_y2': frame_data.get('bbox_y2', 0),
            'target_center_x': frame_data.get('target_center_x', 0),
            'target_center_y': frame_data.get('target_center_y', 0),
            'smoothed_target_x': frame_data.get('smoothed_target_x', 0),
            'smoothed_target_y': frame_data.get('smoothed_target_y', 0),
            'pixel_error_x': frame_data.get('pixel_error_x', 0),
            'pixel_error_y': frame_data.get('pixel_error_y', 0),
            'distance_error': frame_data.get('distance_error', 0),
            'current_pan': frame_data.get('current_pan', 0.0),
            'current_tilt': frame_data.get('current_tilt', 0.0),
            'target_pan': frame_data.get('target_pan', 0.0),
            'target_tilt': frame_data.get('target_tilt', 0.0),
            'servo_command_pan': frame_data.get('servo_command_pan', 0.0),
            'servo_command_tilt': frame_data.get('servo_command_tilt', 0.0),
            'processing_time_ms': processing_time_ms,
            'tracking_enabled': frame_data.get('tracking_enabled', False)
        }
        
        # Write to log file
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(log_row)
    
    def log_servo_command(self, 
                          pan_angle: float, 
                          tilt_angle: float, 
                          current_pan: float,
                          current_tilt: float,
                          target_position: Optional[Tuple[int, int]] = None,
                          frame_center: Optional[Tuple[int, int]] = None):
        """
        Log a servo command event
        
        Args:
            pan_angle: Pan angle command sent to servo
            tilt_angle: Tilt angle command sent to servo
            current_pan: Current pan angle before command
            current_tilt: Current tilt angle before command
            target_position: Target object position (x, y) if available
            frame_center: Frame center position (x, y)
        """
        # Calculate pixel errors if both target and frame center are available
        pixel_error_x = 0
        pixel_error_y = 0
        distance_error = 0
        
        if target_position and frame_center:
            pixel_error_x = target_position[0] - frame_center[0]
            pixel_error_y = target_position[1] - frame_center[1]
            distance_error = ((pixel_error_x ** 2) + (pixel_error_y ** 2)) ** 0.5
        
        frame_data = {
            'camera_center_x': frame_center[0] if frame_center else 0,
            'camera_center_y': frame_center[1] if frame_center else 0,
            'target_detected': target_position is not None,
            'smoothed_target_x': target_position[0] if target_position else 0,
            'smoothed_target_y': target_position[1] if target_position else 0,
            'pixel_error_x': pixel_error_x,
            'pixel_error_y': pixel_error_y,
            'distance_error': distance_error,
            'current_pan': current_pan,
            'current_tilt': current_tilt,
            'servo_command_pan': pan_angle,
            'servo_command_tilt': tilt_angle,
            'tracking_enabled': True
        }
        
        self.log_frame(frame_data)
    
    def log_detection(self,
                     target_detection: Dict[str, Any],
                     frame_center: Tuple[int, int],
                     servo_data: Dict[str, float],
                     processing_time_ms: float = 0.0):
        """
        Log a detection event with comprehensive tracking data
        
        Args:
            target_detection: Detection data for the current target
            frame_center: Camera frame center coordinates (x, y)
            servo_data: Dictionary with servo position data
            processing_time_ms: Frame processing time in milliseconds
        """
        # Extract detection data
        bbox = target_detection.get('bbox', (0, 0, 0, 0))
        center = target_detection.get('center', (0, 0))
        smoothed_center = target_detection.get('smoothed_center', center)
        
        # Calculate pixel errors
        pixel_error_x = smoothed_center[0] - frame_center[0]
        pixel_error_y = smoothed_center[1] - frame_center[1]
        distance_error = ((pixel_error_x ** 2) + (pixel_error_y ** 2)) ** 0.5
        
        frame_data = {
            'camera_center_x': frame_center[0],
            'camera_center_y': frame_center[1],
            'target_detected': True,
            'target_class': target_detection.get('class_name', ''),
            'target_confidence': target_detection.get('confidence', 0.0),
            'bbox_x1': bbox[0],
            'bbox_y1': bbox[1],
            'bbox_x2': bbox[2],
            'bbox_y2': bbox[3],
            'target_center_x': center[0],
            'target_center_y': center[1],
            'smoothed_target_x': smoothed_center[0],
            'smoothed_target_y': smoothed_center[1],
            'pixel_error_x': pixel_error_x,
            'pixel_error_y': pixel_error_y,
            'distance_error': distance_error,
            'current_pan': servo_data.get('current_pan', 0.0),
            'current_tilt': servo_data.get('current_tilt', 0.0),
            'target_pan': servo_data.get('target_pan', 0.0),
            'target_tilt': servo_data.get('target_tilt', 0.0),
            'servo_command_pan': servo_data.get('command_pan', 0.0),
            'servo_command_tilt': servo_data.get('command_tilt', 0.0),
            'tracking_enabled': target_detection.get('tracking_enabled', False)
        }
        
        self.log_frame(frame_data, processing_time_ms)