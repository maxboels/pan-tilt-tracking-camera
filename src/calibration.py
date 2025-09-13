#!/usr/bin/env python3
"""
Camera-Servo Calibration System for Pan-Tilt Tracking Camera
Helps calibrate the mapping between pixel coordinates and servo positions
"""

import cv2
import numpy as np
import json
import time

# Handle imports for both module usage and standalone execution
try:
    from .servo_controller import ArduinoServoController
    from .usb_camera import USBCamera
except ImportError:
    from servo_controller import ArduinoServoController
    from usb_camera import USBCamera

class CameraServoCalibrator:
    def __init__(self, config_file="calibration.json"):
        """
        Initialize calibration system
        
        Args:
            config_file: File to save/load calibration data
        """
        self.config_file = config_file
        self.servo_controller = ArduinoServoController()
        self.camera = USBCamera()
        
        # Calibration data
        self.calibration_points = []  # [(pixel_x, pixel_y, servo_pan, servo_tilt), ...]
        self.transformation_matrix = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Current servo positions
        self.current_pan = 0
        self.current_tilt = 0
        
        # Grid calibration points (servo angles)
        self.calibration_grid = [
            (-45, -30), (-45, 0), (-45, 30),  # Left column
            (0, -30),   (0, 0),   (0, 30),    # Center column  
            (45, -30),  (45, 0),  (45, 30)    # Right column
        ]
        
        self.current_point_index = 0
        self.frame_center = None
        
    def load_calibration(self):
        """Load existing calibration data"""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                
            self.calibration_points = data.get('calibration_points', [])
            
            if 'transformation_matrix' in data:
                self.transformation_matrix = np.array(data['transformation_matrix'])
                
            if 'camera_matrix' in data:
                self.camera_matrix = np.array(data['camera_matrix'])
                
            if 'distortion_coeffs' in data:
                self.distortion_coeffs = np.array(data['distortion_coeffs'])
                
            print(f"Loaded calibration data with {len(self.calibration_points)} points")
            return True
            
        except FileNotFoundError:
            print("No existing calibration file found")
            return False
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def save_calibration(self):
        """Save calibration data"""
        data = {
            'calibration_points': self.calibration_points,
            'timestamp': time.time()
        }
        
        if self.transformation_matrix is not None:
            data['transformation_matrix'] = self.transformation_matrix.tolist()
            
        if self.camera_matrix is not None:
            data['camera_matrix'] = self.camera_matrix.tolist()
            
        if self.distortion_coeffs is not None:
            data['distortion_coeffs'] = self.distortion_coeffs.tolist()
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Calibration data saved to {self.config_file}")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def calculate_transformation(self):
        """Calculate transformation matrix from pixel to servo coordinates"""
        if len(self.calibration_points) < 4:
            print("Need at least 4 calibration points")
            return False
        
        # Separate pixel and servo coordinates
        pixel_coords = np.array([[p[0], p[1]] for p in self.calibration_points], dtype=np.float32)
        servo_coords = np.array([[p[2], p[3]] for p in self.calibration_points], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        if len(self.calibration_points) >= 4:
            self.transformation_matrix = cv2.getPerspectiveTransform(
                pixel_coords[:4], servo_coords[:4]
            )
            print("Perspective transformation matrix calculated")
        
        # Calculate homography for more robust transformation
        if len(self.calibration_points) >= 4:
            homography, mask = cv2.findHomography(
                pixel_coords, servo_coords, 
                cv2.RANSAC, 5.0
            )
            
            if homography is not None:
                self.transformation_matrix = homography
                print("Homography transformation calculated")
                return True
        
        return False
    
    def pixel_to_servo(self, pixel_x, pixel_y):
        """Convert pixel coordinates to servo angles"""
        # EMERGENCY OVERRIDE - Direct fix ignoring all other calibration
        
        # User-adjustable settings for different servo configurations
        USE_EMERGENCY_FIX = True  # Set to False to use original calibration
        INVERT_PAN_DIRECTION = False  # IMPORTANT: Set to FALSE since servo_controller already inverts
        INVERT_TILT_DIRECTION = False  # Set to True if tilt servo is inverted
        
        if not USE_EMERGENCY_FIX and self.transformation_matrix is not None:
            # Use original calibration transformation
            pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            servo_point = cv2.perspectiveTransform(pixel_point, self.transformation_matrix)
            return servo_point[0][0][0], servo_point[0][0][1]
        
        # Direct mapping with configurable inversions
        if self.frame_center is None:
            self.frame_center = (320, 240)  # Default for 640x480 camera
            
        # Calculate error from center
        pan_error = pixel_x - self.frame_center[0]  # Positive if target is to the RIGHT
        tilt_error = pixel_y - self.frame_center[1]  # Positive if target is at BOTTOM
        
        # Target quadrant detection for logging
        quadrant = ""
        if pan_error > 0 and tilt_error < 0: quadrant = "TOP-RIGHT"
        elif pan_error < 0 and tilt_error < 0: quadrant = "TOP-LEFT"
        elif pan_error > 0 and tilt_error > 0: quadrant = "BOTTOM-RIGHT"
        elif pan_error < 0 and tilt_error > 0: quadrant = "BOTTOM-LEFT"
        
        # IMPROVED MAPPING for wide-angle cameras (130° FOV)
        # Calculate frame size for scaling
        frame_width = self.frame_center[0] * 2
        frame_height = self.frame_center[1] * 2
        
        # Dynamic scaling based on pixel position (more aggressive at edges for wide-angle lens)
        # For wide-angle cameras, use non-linear mapping to compensate for distortion
        normalized_x = pan_error / (frame_width / 2)  # -1 to 1
        normalized_y = tilt_error / (frame_height / 2)  # -1 to 1
        
        # Apply non-linear correction for wide-angle lens
        # This increases sensitivity at the edges where distortion is greater
        if abs(normalized_x) > 0.5:
            pan_scale = 0.15 + 0.05 * (abs(normalized_x) - 0.5) / 0.5  # 0.15-0.2 based on distance from center
        else:
            pan_scale = 0.15  # Base scale factor
            
        if abs(normalized_y) > 0.5:
            tilt_scale = 0.12 + 0.03 * (abs(normalized_y) - 0.5) / 0.5  # 0.12-0.15 based on distance from center
        else:
            tilt_scale = 0.12  # Base scale factor
        
        # Apply user direction settings
        pan_multiplier = -1 if INVERT_PAN_DIRECTION else 1
        tilt_multiplier = -1 if INVERT_TILT_DIRECTION else 1
        
        # Calculate servo angles with direction control and dynamic scaling
        # For RIGHT side (positive error), pan right (positive angle if not inverted)
        # For BOTTOM (positive error), tilt down (negative angle if not inverted)
        pan_angle = pan_error * pan_scale * pan_multiplier
        tilt_angle = -tilt_error * tilt_scale * tilt_multiplier
        
        print(f"EMERGENCY FIX ACTIVE: Object in {quadrant}, pixel error: ({pan_error}, {tilt_error})")
        print(f"EMERGENCY FIX: Calculated angles: Pan={pan_angle:.1f}°, Tilt={tilt_angle:.1f}°")
        print(f"EMERGENCY FIX: Expected camera movement: " + 
              f"Pan={'RIGHT' if (pan_error > 0) == (pan_multiplier > 0) else 'LEFT'}, " +
              f"Tilt={'DOWN' if (tilt_error > 0) == (tilt_multiplier > 0) else 'UP'}")
        
        return pan_angle, tilt_angle
    
    def servo_to_pixel(self, pan_angle, tilt_angle):
        """Convert servo angles to expected pixel coordinates"""
        if self.transformation_matrix is None:
            return None, None
        
        # Use inverse transformation
        inv_matrix = np.linalg.inv(self.transformation_matrix)
        servo_point = np.array([[[pan_angle, tilt_angle]]], dtype=np.float32)
        pixel_point = cv2.perspectiveTransform(servo_point, inv_matrix)
        
        return int(pixel_point[0][0][0]), int(pixel_point[0][0][1])
    
    def run_calibration(self):
        """Run interactive calibration process"""
        if not self.servo_controller.connect():
            print("Failed to connect to servo controller")
            return False
        
        if not self.camera.open():
            print("Failed to open camera")
            return False
        
        if not self.camera.start_capture():
            print("Failed to start camera")
            return False
        
        print("\n=== Camera-Servo Calibration ===")
        print("Instructions:")
        print("1. The servo will move to different positions")
        print("2. Click on the center crosshair in the camera view")
        print("3. Press SPACE to confirm the point")
        print("4. Press 'q' to quit, 's' to save calibration")
        print("5. Press 'r' to restart calibration")
        
        self.current_point_index = 0
        
        # Mouse callback for clicking points
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.clicked_point = (x, y)
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)
        
        self.clicked_point = None
        
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Store frame center
            if self.frame_center is None:
                self.frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
            
            # Move to current calibration point
            if self.current_point_index < len(self.calibration_grid):
                target_pan, target_tilt = self.calibration_grid[self.current_point_index]
                
                # Move servo if position changed
                if abs(target_pan - self.current_pan) > 1 or abs(target_tilt - self.current_tilt) > 1:
                    self.servo_controller.move_servo(0, target_pan)
                    self.servo_controller.move_servo(1, target_tilt)
                    self.current_pan = target_pan
                    self.current_tilt = target_tilt
                    time.sleep(1)  # Wait for servos to settle
                
                # Draw instructions
                cv2.putText(frame, f"Point {self.current_point_index + 1}/{len(self.calibration_grid)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Pan: {target_pan}°, Tilt: {target_tilt}°", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Click on center crosshair, then press SPACE", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Calibration Complete! Press 's' to save", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw center crosshair
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            cv2.drawMarker(frame, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            
            # Draw clicked point
            if self.clicked_point:
                cv2.circle(frame, self.clicked_point, 5, (0, 255, 0), -1)
                cv2.putText(frame, f"Clicked: {self.clicked_point}", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw existing calibration points
            for i, (px, py, pan, tilt) in enumerate(self.calibration_points):
                cv2.circle(frame, (int(px), int(py)), 3, (255, 0, 0), -1)
                cv2.putText(frame, f"{i+1}", (int(px) + 5, int(py) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to confirm point
                if (self.clicked_point and 
                    self.current_point_index < len(self.calibration_grid)):
                    
                    pan, tilt = self.calibration_grid[self.current_point_index]
                    point = (self.clicked_point[0], self.clicked_point[1], pan, tilt)
                    self.calibration_points.append(point)
                    
                    print(f"Added calibration point: pixel({self.clicked_point[0]}, {self.clicked_point[1]}) -> servo({pan}, {tilt})")
                    
                    self.clicked_point = None
                    self.current_point_index += 1
                    
                    # Calculate transformation if we have enough points
                    if len(self.calibration_points) >= 4:
                        self.calculate_transformation()
                        
            elif key == ord('s'):
                if len(self.calibration_points) >= 4:
                    self.calculate_transformation()
                    self.save_calibration()
                else:
                    print("Need at least 4 calibration points to save")
                    
            elif key == ord('r'):
                self.calibration_points.clear()
                self.current_point_index = 0
                self.transformation_matrix = None
                print("Calibration reset")
        
        # Center servos and cleanup
        self.servo_controller.center_servos()
        self.servo_controller.disconnect()
        self.camera.close()
        cv2.destroyAllWindows()
        
        return len(self.calibration_points) >= 4
    
    def test_calibration(self):
        """Test the calibration by moving servos and showing expected vs actual positions"""
        if not self.servo_controller.connect():
            print("Failed to connect to servo controller")
            return
        
        if not self.camera.open() or not self.camera.start_capture():
            print("Failed to open camera")
            return
        
        print("\n=== Testing Calibration ===")
        print("Click anywhere in the image to move servos to that position")
        print("Green circle = clicked position, Red circle = expected position based on servo angles")
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert pixel to servo angles
                pan_angle, tilt_angle = self.pixel_to_servo(x, y)
                
                # Clamp to servo limits
                pan_angle = max(-90, min(90, pan_angle))
                tilt_angle = max(-45, min(45, tilt_angle))
                
                # Move servos
                self.servo_controller.move_servo(0, pan_angle)
                self.servo_controller.move_servo(1, tilt_angle)
                
                print(f"Clicked ({x}, {y}) -> Servo angles: Pan={pan_angle:.1f}°, Tilt={tilt_angle:.1f}°")
                
                # Store for display
                param['clicked'] = (x, y)
                param['servo_angles'] = (pan_angle, tilt_angle)
        
        cv2.namedWindow('Calibration Test')
        callback_data = {'clicked': None, 'servo_angles': None}
        cv2.setMouseCallback('Calibration Test', mouse_callback, callback_data)
        
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Draw clicked position
            if callback_data['clicked']:
                x, y = callback_data['clicked']
                cv2.circle(frame, (x, y), 8, (0, 255, 0), 2)
                cv2.putText(frame, f"Clicked: ({x}, {y})", (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw expected position based on current servo angles
            if callback_data['servo_angles']:
                pan, tilt = callback_data['servo_angles']
                expected_x, expected_y = self.servo_to_pixel(pan, tilt)
                
                if expected_x and expected_y:
                    cv2.circle(frame, (expected_x, expected_y), 8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Expected: ({expected_x}, {expected_y})", 
                               (expected_x + 10, expected_y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw center reference
            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            cv2.drawMarker(frame, center, (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            
            cv2.putText(frame, "Click to test servo positioning", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Green=Clicked, Red=Expected, Press 'q' to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Calibration Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.servo_controller.center_servos()
        self.servo_controller.disconnect()
        self.camera.close()
        cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Camera-Servo Calibration System')
    parser.add_argument('--calibrate', '-c', action='store_true', 
                       help='Run calibration process')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Test existing calibration')
    parser.add_argument('--config', default='calibration.json',
                       help='Calibration file path')
    
    args = parser.parse_args()
    
    calibrator = CameraServoCalibrator(args.config)
    
    if args.calibrate:
        calibrator.load_calibration()
        calibrator.run_calibration()
    elif args.test:
        if calibrator.load_calibration():
            calibrator.test_calibration()
        else:
            print("No calibration data found. Run calibration first with --calibrate")
    else:
        print("Use --calibrate to run calibration or --test to test existing calibration")

if __name__ == "__main__":
    main()