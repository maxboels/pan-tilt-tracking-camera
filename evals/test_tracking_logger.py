#!/usr/bin/env python3
"""
Test script for the tracking logger functionality
"""

import os
import sys
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking_logger import TrackingLogger
import time
import random

def test_tracking_logger():
    """Test the tracking logger with simulated data"""
    print("Testing Tracking Logger...")
    
    # Create a test logger with a descriptive experiment name
    logger = TrackingLogger(log_dir="logs", experiment_name="test_simulation")
    
    # Simulate frame center
    frame_center = (960, 540)  # Full HD center
    
    # Simulate detection and tracking over 100 frames
    for i in range(1, 101):
        print(f"Simulating frame {i}/100")
        
        # Simulate some movement of the target
        target_x = frame_center[0] + int(100 * math.sin(i * 0.1))
        target_y = frame_center[1] + int(50 * math.cos(i * 0.2))
        
        # Simulate servo positions (they follow the target with some lag)
        servo_lag = 5
        if i > servo_lag:
            prev_x = frame_center[0] + int(100 * math.sin((i-servo_lag) * 0.1))
            prev_y = frame_center[1] + int(50 * math.cos((i-servo_lag) * 0.2))
        else:
            prev_x = frame_center[0]
            prev_y = frame_center[1]
        
        # Convert pixel positions to servo angles (simple simulation)
        # Assume 1 degree ~ 20 pixels
        pan_error = target_x - frame_center[0]
        tilt_error = target_y - frame_center[1]
        
        current_pan = (prev_x - frame_center[0]) / 20
        current_tilt = (prev_y - frame_center[1]) / 20
        target_pan = pan_error / 20
        target_tilt = tilt_error / 20
        
        # Create a simulated detection
        detection = {
            'class_name': 'person',
            'confidence': 0.8 + random.uniform(-0.1, 0.1),  # Vary confidence slightly
            'bbox': (target_x - 50, target_y - 100, target_x + 50, target_y + 100),
            'center': (target_x, target_y),
            'smoothed_center': (target_x, target_y),
            'tracking_enabled': True
        }
        
        # Create servo data
        servo_data = {
            'current_pan': current_pan,
            'current_tilt': current_tilt,
            'target_pan': target_pan,
            'target_tilt': target_tilt,
            'command_pan': target_pan,
            'command_tilt': target_tilt
        }
        
        # Log the frame data
        logger.log_detection(detection, frame_center, servo_data, processing_time_ms=16.7)  # ~60 FPS
        
        # Simulate servo commands every 5 frames
        if i % 5 == 0:
            logger.log_servo_command(
                pan_angle=target_pan,
                tilt_angle=target_tilt,
                current_pan=current_pan,
                current_tilt=current_tilt,
                target_position=(target_x, target_y),
                frame_center=frame_center
            )
        
        # Small delay
        time.sleep(0.01)
    
    print(f"Tracking logger test complete. Log file: {logger.log_path}")


if __name__ == "__main__":
    import math
    test_tracking_logger()