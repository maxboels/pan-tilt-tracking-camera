#!/usr/bin/env python3
"""
Camera-Servo Calibration Tool
Simple tool to calibrate pixel-to-servo coordinate mapping
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.usb_camera import USBCamera
from src.servo_controller import ArduinoServoController
from src.calibration import CameraServoCalibrator
import cv2
import time

def main():
    print("Pan-Tilt Camera Calibration Tool")
    print("=" * 40)
    
    # Initialize components
    camera = USBCamera()
    servo_controller = ArduinoServoController()
    calibrator = CameraServoCalibrator("../config/calibration.json")
    
    try:
        # Start camera
        if not camera.open() or not camera.start_capture():
            print("Failed to start camera")
            return
        
        # Connect to servos
        if not servo_controller.connect():
            print("Warning: Could not connect to servo controller")
            print("Running in camera-only mode for visual calibration")
        else:
            servo_controller.center_servos()
        
        print("\nCalibration Instructions:")
        print("- Click on the image to test pixel-to-servo mapping")
        print("- Use keyboard to manually control servos:")
        print("  'w/s' - Tilt up/down")
        print("  'a/d' - Pan left/right") 
        print("  'c' - Center servos")
        print("- Press 'q' to quit")
        print("- Press 'r' to run auto-calibration")
        
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibration", mouse_callback, 
                           (calibrator, servo_controller))
        
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            
            # Draw center crosshair
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            cv2.circle(frame, center, 10, (0, 255, 0), 2)
            cv2.line(frame, (center[0]-20, center[1]), 
                    (center[0]+20, center[1]), (0, 255, 0), 2)
            cv2.line(frame, (center[0], center[1]-20), 
                    (center[0], center[1]+20), (0, 255, 0), 2)
            
            # Show servo positions if connected
            if servo_controller.connected:
                pan_pos = servo_controller.current_pan
                tilt_pos = servo_controller.current_tilt
                cv2.putText(frame, f"Pan: {pan_pos:.1f}° Tilt: {tilt_pos:.1f}°",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and servo_controller.connected:
                servo_controller.center_servos()
                print("Servos centered")
            elif key == ord('w') and servo_controller.connected:
                tilt = servo_controller.current_tilt + 5
                servo_controller.move_servo(1, tilt)
                print(f"Tilt up: {tilt}°")
            elif key == ord('s') and servo_controller.connected:
                tilt = servo_controller.current_tilt - 5
                servo_controller.move_servo(1, tilt)
                print(f"Tilt down: {tilt}°")
            elif key == ord('a') and servo_controller.connected:
                pan = servo_controller.current_pan - 5
                servo_controller.move_servo(0, pan)
                print(f"Pan left: {pan}°")
            elif key == ord('d') and servo_controller.connected:
                pan = servo_controller.current_pan + 5
                servo_controller.move_servo(0, pan)
                print(f"Pan right: {pan}°")
            elif key == ord('r'):
                print("Auto-calibration not implemented yet")
                print("Use manual calibration by clicking points and adjusting servos")
    
    finally:
        camera.close()
        servo_controller.disconnect()
        cv2.destroyAllWindows()

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for calibration"""
    calibrator, servo_controller = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at pixel ({x}, {y})")
        
        if servo_controller.connected:
            try:
                # Convert to servo angles
                pan_angle, tilt_angle = calibrator.pixel_to_servo(x, y)
                print(f"Calculated servo angles: Pan={pan_angle:.1f}°, Tilt={tilt_angle:.1f}°")
                
                # Move servos
                servo_controller.move_servos(pan_angle, tilt_angle)
                
            except Exception as e:
                print(f"Calibration error: {e}")
                print("Using default pixel-to-angle conversion")
                
                # Simple fallback conversion
                frame_center_x = 960  # Assuming 1920x1080
                frame_center_y = 540
                
                pan_angle = (x - frame_center_x) * 0.05  # Simple scaling
                tilt_angle = (y - frame_center_y) * 0.03
                
                servo_controller.move_servos(pan_angle, tilt_angle)

if __name__ == "__main__":
    main()
