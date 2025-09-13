#!/usr/bin/env python3
"""
Tracking Simulation Test Script
Tests tracking by simulating detected people at different positions in the frame
"""

import sys
import os
import time
import json
import numpy as np
import cv2

# Add parent directory to path to import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.servo_controller import ArduinoServoController
from src.calibration import CameraServoCalibrator

class TrackingSimulator:
    """Simulates tracking with fake detections"""
    
    def __init__(self):
        """Initialize the tracking simulator"""
        # Load configuration
        self.config = self.load_config()
        
        # Initialize servo controller with current settings
        servo_config = self.config.get('servo', {})
        self.servo_controller = ArduinoServoController(
            port=servo_config.get('port', '/dev/ttyUSB0'),
            baudrate=servo_config.get('baudrate', 115200),
            inverted_pan=servo_config.get('inverted_pan', True)
        )
        
        # Initialize calibrator
        self.calibrator = CameraServoCalibrator("../config/calibration.json")
        
        # Get frame dimensions from config
        camera_config = self.config.get('camera', {})
        self.frame_width = camera_config.get('resolution', [1920, 1080])[0]
        self.frame_height = camera_config.get('resolution', [1920, 1080])[1]
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        
        # Update calibrator with frame center
        self.calibrator.frame_center = self.frame_center
        
        # Initialize test visualization
        self.visualization = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Test points (x, y, description)
        self.test_points = self.generate_test_points()
        
        # Test results
        self.results = []

    def load_config(self):
        """Load configuration from JSON file"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {
                "camera": {"resolution": [1920, 1080]},
                "servo": {"inverted_pan": True}
            }

    def generate_test_points(self):
        """Generate test points at different locations in the frame"""
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        
        # Generate test points in a grid across the frame
        points = [
            # Center point as reference
            (center_x, center_y, "CENTER"),
            
            # Cardinal directions
            (center_x + 400, center_y, "RIGHT"),
            (center_x - 400, center_y, "LEFT"),
            (center_x, center_y - 300, "TOP"),
            (center_x, center_y + 300, "BOTTOM"),
            
            # Corners
            (center_x + 400, center_y - 300, "TOP-RIGHT"),
            (center_x - 400, center_y - 300, "TOP-LEFT"),
            (center_x + 400, center_y + 300, "BOTTOM-RIGHT"),
            (center_x - 400, center_y + 300, "BOTTOM-LEFT"),
            
            # Far sides
            (center_x + 800, center_y, "FAR-RIGHT"),
            (center_x - 800, center_y, "FAR-LEFT"),
            (center_x, center_y - 400, "FAR-TOP"),
            (center_x, center_y + 400, "FAR-BOTTOM")
        ]
        
        return points

    def draw_simulation_frame(self, current_point=None):
        """Create a visualization frame with the current test point"""
        # Create a blank frame
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Draw a grid
        for x in range(0, self.frame_width, 100):
            cv2.line(frame, (x, 0), (x, self.frame_height), (50, 50, 50), 1)
        for y in range(0, self.frame_height, 100):
            cv2.line(frame, (0, y), (self.frame_width, y), (50, 50, 50), 1)
        
        # Draw frame center
        cv2.circle(frame, self.frame_center, 10, (0, 0, 255), -1)
        cv2.circle(frame, self.frame_center, 50, (0, 0, 255), 2)
        
        # Draw all test points
        for x, y, label in self.test_points:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
        
        # Draw current test point with a person icon
        if current_point:
            x, y, label = current_point
            
            # Draw a simplified person icon (head and body)
            person_height = 120
            head_radius = 20
            
            # Head
            cv2.circle(frame, (x, y - person_height//2 + head_radius), 
                      head_radius, (255, 255, 0), -1)
            
            # Body
            cv2.rectangle(frame, 
                         (x - 15, y - person_height//2 + head_radius*2),
                         (x + 15, y + person_height//2),
                         (255, 255, 0), -1)
            
            # Highlight active test point
            cv2.circle(frame, (x, y), 50, (255, 255, 0), 3)
            cv2.putText(frame, f"TESTING: {label}", (x - 70, y - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Add expected movement
            pan_error = x - self.frame_center[0]
            tilt_error = y - self.frame_center[1]
            
            if pan_error > 0:
                expected_pan = "RIGHT"
            elif pan_error < 0:
                expected_pan = "LEFT"
            else:
                expected_pan = "CENTER"
                
            if tilt_error > 0:
                expected_tilt = "DOWN"
            elif tilt_error < 0:
                expected_tilt = "UP"
            else:
                expected_tilt = "CENTER"
                
            cv2.putText(frame, f"Expected: Pan {expected_pan}, Tilt {expected_tilt}", 
                       (30, self.frame_height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame

    def calculate_servo_angles(self, pixel_x, pixel_y):
        """Calculate servo angles for a given pixel position"""
        try:
            # Try to use the calibrator's conversion
            pan_angle, tilt_angle = self.calibrator.pixel_to_servo(pixel_x, pixel_y)
        except Exception as e:
            print(f"Error using calibrator: {e}")
            # Fallback to basic conversion
            pan_error = pixel_x - self.frame_center[0]
            tilt_error = pixel_y - self.frame_center[1]
            
            # Simple conversion (matching what's in main.py)
            pan_angle = -pan_error * 0.1  # Added negative sign
            tilt_angle = -tilt_error * 0.1  # Negative for Y-axis inversion
            
            print(f"Using fallback calculation: pan_angle={pan_angle}, tilt_angle={tilt_angle}")
        
        return pan_angle, tilt_angle

    def test_point(self, point):
        """Test a single point with the servo controller"""
        x, y, label = point
        print(f"\n=== Testing point: {label} ({x}, {y}) ===")
        
        # Create visualization
        frame = self.draw_simulation_frame(point)
        cv2.imshow("Tracking Simulation", frame)
        cv2.waitKey(1)
        
        # Calculate servo angles
        pan_angle, tilt_angle = self.calculate_servo_angles(x, y)
        
        # Calculate expected direction
        pan_error = x - self.frame_center[0]
        tilt_error = y - self.frame_center[1]
        
        expected_pan_dir = "RIGHT" if pan_error > 0 else "LEFT" if pan_error < 0 else "CENTER"
        expected_tilt_dir = "DOWN" if tilt_error > 0 else "UP" if tilt_error < 0 else "CENTER"
        
        print(f"Pixel position: ({x}, {y})")
        print(f"Frame center: ({self.frame_center[0]}, {self.frame_center[1]})")
        print(f"Pixel error: Pan={pan_error}, Tilt={tilt_error}")
        print(f"Calculated angles: Pan={pan_angle:.2f}°, Tilt={tilt_angle:.2f}°")
        print(f"Expected movement: Pan {expected_pan_dir}, Tilt {expected_tilt_dir}")
        
        # Move servos
        if self.servo_controller.connected:
            self.servo_controller.move_servos(pan_angle, tilt_angle)
            time.sleep(1.5)  # Wait for servos to move
            
            # Get user feedback on actual movement
            cv2.putText(frame, "Did the camera move in the expected direction?",
                       (30, self.frame_height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Tracking Simulation", frame)
            cv2.waitKey(1)
            
            response = input(f"Did the camera move correctly? (Pan {expected_pan_dir}, Tilt {expected_tilt_dir}) [y/n]: ").lower()
            correct = response in ['y', 'yes']
            
            result = {
                'position': (x, y),
                'label': label,
                'pan_error': pan_error,
                'tilt_error': tilt_error,
                'pan_angle': pan_angle,
                'tilt_angle': tilt_angle,
                'expected_pan': expected_pan_dir,
                'expected_tilt': expected_tilt_dir,
                'correct': correct
            }
            
            self.results.append(result)
            return result
        else:
            print("Servo controller not connected - skipping physical test")
            return None

    def run_all_tests(self):
        """Run tests for all points"""
        if not self.servo_controller.connected:
            print("Error: Servo controller not connected")
            return False
        
        print("\n=== Starting Tracking Simulation Test ===")
        print(f"Frame dimensions: {self.frame_width}x{self.frame_height}")
        print(f"Frame center: {self.frame_center}")
        print(f"Servo config: inverted_pan={self.servo_controller.pan_direction == -1}")
        
        # Center servos initially
        print("Centering servos...")
        self.servo_controller.center_servos()
        time.sleep(2)
        
        # Test each point
        for point in self.test_points:
            self.test_point(point)
            
            # Return to center between tests
            print("Returning to center...")
            self.servo_controller.center_servos()
            time.sleep(1.5)
        
        # Analyze results
        self.analyze_results()
        return True

    def analyze_results(self):
        """Analyze test results and suggest fixes"""
        if not self.results:
            print("No test results to analyze")
            return
        
        print("\n=== Test Results Analysis ===")
        
        total = len(self.results)
        correct = sum(1 for r in self.results if r['correct'])
        incorrect = total - correct
        
        print(f"Total tests: {total}")
        print(f"Correct movements: {correct} ({correct/total*100:.1f}%)")
        print(f"Incorrect movements: {incorrect} ({incorrect/total*100:.1f}%)")
        
        # Check for patterns in the failures
        if incorrect > 0:
            print("\nAnalyzing incorrect movements:")
            
            # Check pan direction failures
            pan_right_failures = sum(1 for r in self.results 
                                    if not r['correct'] and r['expected_pan'] == 'RIGHT')
            pan_left_failures = sum(1 for r in self.results 
                                   if not r['correct'] and r['expected_pan'] == 'LEFT')
            
            # Check tilt direction failures
            tilt_up_failures = sum(1 for r in self.results 
                                  if not r['correct'] and r['expected_tilt'] == 'UP')
            tilt_down_failures = sum(1 for r in self.results 
                                    if not r['correct'] and r['expected_tilt'] == 'DOWN')
            
            print(f"Pan RIGHT failures: {pan_right_failures}")
            print(f"Pan LEFT failures: {pan_left_failures}")
            print(f"Tilt UP failures: {tilt_up_failures}")
            print(f"Tilt DOWN failures: {tilt_down_failures}")
            
            # Check if all pan directions fail (suggests inverted pan)
            if pan_right_failures > 0 and pan_left_failures > 0:
                print("\n⚠️ Both RIGHT and LEFT pan movements fail - likely inverted pan servo")
                print("Try changing inverted_pan setting in config.json")
            
            # Check if all tilt directions fail (suggests inverted tilt)
            if tilt_up_failures > 0 and tilt_down_failures > 0:
                print("\n⚠️ Both UP and DOWN tilt movements fail - likely inverted tilt servo")
                print("Try modifying the tilt_direction in servo_controller.py")
            
            # Show specific failures
            print("\nFailed test points:")
            for r in self.results:
                if not r['correct']:
                    print(f"- {r['label']}: Expected Pan {r['expected_pan']}, Tilt {r['expected_tilt']}")
                    print(f"  Pixel: ({r['position'][0]}, {r['position'][1]}), Error: ({r['pan_error']}, {r['tilt_error']})")
                    print(f"  Angles: Pan={r['pan_angle']:.2f}°, Tilt={r['tilt_angle']:.2f}°")
            
            # Suggest code fixes
            print("\nPossible code fixes:")
            print("1. In calibration.py, check the pixel_to_servo function:")
            print("   Current: pan_angle = -pan_error * 0.1  # Already has negative sign")
            print("   Try: pan_angle = pan_error * 0.1  # Remove negative sign")
            print("\n2. In servo_controller.py, check the pan_direction variable:")
            print("   Current: self.pan_direction = -1 if inverted_pan else 1")
            print("   Try flipping this logic or the sign")
        else:
            print("\n✅ All movements were correct! No fixes needed.")

    def run_interactive(self):
        """Run an interactive test with manual click positions"""
        if not self.servo_controller.connect():
            print("Error: Could not connect to servo controller")
            return False
        
        print("\n=== Interactive Tracking Test ===")
        print("Click anywhere on the frame to test servo movement to that position")
        print("Press 'c' to center servos")
        print("Press 'q' to quit")
        
        # Create window and set mouse callback
        cv2.namedWindow("Interactive Test")
        
        # Initialize interactive frame
        frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Mouse callback function
        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"\nClicked at ({x}, {y})")
                self.test_point((x, y, "CLICK"))
        
        cv2.setMouseCallback("Interactive Test", on_mouse_click)
        
        # Center servos initially
        self.servo_controller.center_servos()
        
        while True:
            # Create visualization frame
            frame = self.draw_simulation_frame()
            
            # Add instructions
            cv2.putText(frame, "Click anywhere to test servo movement", 
                       (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'c' to center, 'q' to quit", 
                       (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Interactive Test", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Centering servos...")
                self.servo_controller.center_servos()
        
        # Clean up
        cv2.destroyAllWindows()
        self.servo_controller.disconnect()
        return True

def main():
    """Main function"""
    print("=== Tracking Simulation Test ===")
    print("This script simulates tracking with fake detections at different positions")
    print("and verifies whether the camera moves in the expected directions.")
    
    simulator = TrackingSimulator()
    
    if not simulator.servo_controller.connect():
        print("Error: Could not connect to servo controller")
        return
    
    try:
        print("\nSelect test mode:")
        print("1. Run all predefined test points")
        print("2. Interactive test (click anywhere)")
        
        choice = input("Enter choice (1/2): ")
        
        if choice == '1':
            simulator.run_all_tests()
        elif choice == '2':
            simulator.run_interactive()
        else:
            print("Invalid choice")
    
    finally:
        # Ensure servos are centered and disconnected
        if simulator.servo_controller.connected:
            simulator.servo_controller.center_servos()
            time.sleep(1)
            simulator.servo_controller.disconnect()

if __name__ == "__main__":
    main()
