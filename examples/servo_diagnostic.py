#!/usr/bin/env python3
"""
Servo Direction Diagnostic Tool
Test the mathematical relationships and servo directions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.servo_controller import ArduinoServoController
from src.calibration import CameraServoCalibrator
import time

def test_servo_directions():
    """Test servo direction logic"""
    print("=== Servo Direction Test ===")
    
    # Test with both inverted_pan settings
    for inverted in [False, True]:
        print(f"\nTesting with inverted_pan={inverted}")
        controller = ArduinoServoController(inverted_pan=inverted)
        
        print(f"  pan_direction: {controller.pan_direction}")
        print(f"  tilt_direction: {controller.tilt_direction}")
        
        # Simulate movement requests
        test_angles = [10, -10, 30, -30]
        for angle in test_angles:
            # Calculate what would be sent to servo
            calibrated_angle = (angle * controller.pan_direction) + controller.pan_offset
            print(f"  Request {angle:3.0f}° -> Servo gets {calibrated_angle:3.0f}°")

def test_pixel_to_servo_math():
    """Test pixel to servo coordinate conversion"""
    print("\n=== Pixel to Servo Math Test ===")
    
    calibrator = CameraServoCalibrator()
    calibrator.frame_center = (960, 540)  # 1920x1080 center
    
    # Test scenarios
    test_cases = [
        ("Center", 960, 540),
        ("Right side", 1200, 540),   # Person on right side of frame
        ("Left side", 720, 540),     # Person on left side of frame
        ("Top center", 960, 300),    # Person above center
        ("Bottom center", 960, 780), # Person below center
    ]
    
    for name, pixel_x, pixel_y in test_cases:
        pan_angle, tilt_angle = calibrator.pixel_to_servo(pixel_x, pixel_y)
        pan_error = pixel_x - calibrator.frame_center[0]
        tilt_error = pixel_y - calibrator.frame_center[1]
        
        print(f"{name:12} | Pixel: ({pixel_x:4}, {pixel_y:3}) | Error: ({pan_error:4}, {tilt_error:4}) | Servo: Pan={pan_angle:6.1f}°, Tilt={tilt_angle:6.1f}°")

def test_expected_behavior():
    """Test what the expected behavior should be"""
    print("\n=== Expected Behavior Analysis ===")
    print("When person is on RIGHT side of camera view:")
    print("  - Camera should pan RIGHT (positive angle) to center on them")
    print("  - Pixel X > Center X (positive pan_error)")
    print("  - Should result in positive servo angle")
    print()
    print("When person is on LEFT side of camera view:")
    print("  - Camera should pan LEFT (negative angle) to center on them")
    print("  - Pixel X < Center X (negative pan_error)")
    print("  - Should result in negative servo angle")
    print()
    print("Current calibration logic:")
    print("  pan_error = pixel_x - frame_center[0]")
    print("  pan_angle = pan_error * 0.1")
    print("  This should be CORRECT for normal servo direction")

def live_test_with_servo():
    """Interactive test with actual servo"""
    print("\n=== Live Servo Test ===")
    print("This will test actual servo movements")
    
    controller = ArduinoServoController(inverted_pan=False)  # Test current setting
    
    if not controller.connect():
        print("Could not connect to servo controller")
        return
    
    try:
        print("Connected to servo controller")
        print("Testing pan movements...")
        
        # Center first
        print("Centering servos...")
        controller.center_servos()
        time.sleep(2)
        
        # Test right movement (should be positive angle)
        print("Moving RIGHT (positive angle)...")
        controller.move_servo(0, 20)  # Pan channel, +20 degrees
        time.sleep(2)
        
        # Back to center
        controller.move_servo(0, 0)
        time.sleep(2)
        
        # Test left movement (should be negative angle)
        print("Moving LEFT (negative angle)...")
        controller.move_servo(0, -20)  # Pan channel, -20 degrees
        time.sleep(2)
        
        # Back to center
        controller.move_servo(0, 0)
        
        print("Test complete. Did the servo move in the expected directions?")
        print("RIGHT movement should turn the camera to the right")
        print("LEFT movement should turn the camera to the left")
        
    finally:
        controller.disconnect()

if __name__ == "__main__":
    print("Pan-Tilt Servo Diagnostic Tool")
    print("=" * 40)
    
    test_servo_directions()
    test_pixel_to_servo_math()
    test_expected_behavior()
    
    while True:
        response = input("\nRun live servo test? (y/n): ").lower()
        if response in ['y', 'yes']:
            live_test_with_servo()
            break
        elif response in ['n', 'no']:
            break
        else:
            print("Please enter 'y' or 'n'")
