#!/usr/bin/env python3
"""
Arduino Servo Controller for Pan-Tilt Tracking Camera
Communicates with Arduino running servo bridge firmware
"""

import serial
import time
import threading
from collections import deque

def calculate_pan_command(self, pan_error):
        """Convert pan error to servo angle adjustment"""
        
        # Apply deadzone - don't move if error is small
        if abs(pan_error) < self.deadzone_pixels:
            return 0.0
        
        # Convert pixel error to angle adjustment
        angle_adjustment = pan_error * self.error_to_angle_ratio
        
        # Limit movement speed
        angle_adjustment = max(-self.max_speed, min(self.max_speed, angle_adjustment))

class ArduinoServoController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, pan_channel=0, tilt_channel=1, inverted_pan=True):
        """
        Initialize Arduino servo controller
        
        Args:
            port: Serial port for Arduino connection
            baudrate: Serial communication speed
            pan_channel: PCA9685 channel for pan servo
            tilt_channel: PCA9685 channel for tilt servo
            inverted_pan: Whether to invert pan servo direction
        """
        self.port = port
        self.baudrate = baudrate
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel
        self.serial_connection = None
        
        # Servo position limits (in degrees)
        self.pan_min = -90
        self.pan_max = 90
        self.tilt_min = -45
        self.tilt_max = 45
        
        # Current servo positions
        self.current_pan = 0
        self.current_tilt = 0
        
        # Movement parameters - enhanced for wide-angle camera
        self.max_speed = 12.0  # degrees per update - slightly faster for better responsiveness
        self.error_to_angle_ratio = 0.15  # Convert pixel error to degrees - increased for better range
        
        # Small deadzone to prevent micro-jitter
        self.deadzone_pixels = 5
        
        # Calibration offsets - adjust these to correct servo alignment
        self.pan_offset = 0  # Add this to pan commands
        self.tilt_offset = 0  # Add this to tilt commands
        self.pan_direction = -1 if inverted_pan else 1  # Pan servo direction
        self.tilt_direction = 1  # 1 for normal, -1 to reverse direction
        
        # Camera offset compensation for tilt drift
        self.camera_offset_x = 0  # Horizontal offset of camera from tilt axis (pixels)
        self.camera_offset_compensation = True  # Enable/disable offset compensation
        
        # Command queue for thread-safe operation
        self.command_queue = deque()
        self.queue_lock = threading.Lock()
        
        self.connected = False
        
    def connect(self):
        """Connect to Arduino"""
        # Try multiple common ports
        possible_ports = [self.port, '/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyUSB1']
        
        for port in possible_ports:
            try:
                self.serial_connection = serial.Serial(
                    port, 
                    self.baudrate, 
                    timeout=1.0,
                    write_timeout=1.0
                )
                time.sleep(2)  # Wait for Arduino to reset
                
                # Test connection
                self.serial_connection.write(b"CENTER\n")
                response = self.serial_connection.readline().decode().strip()
                
                if "OK" in response or "Ready" in response:
                    self.connected = True
                    self.port = port  # Update port to the one that worked
                    print(f"Arduino servo controller connected on {port}")
                    return True
                else:
                    self.serial_connection.close()
                    
            except Exception as e:
                if self.serial_connection:
                    try:
                        self.serial_connection.close()
                    except:
                        pass
                continue
        
        print("Failed to connect to Arduino on any port")
        return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.serial_connection and self.serial_connection.is_open:
            # Center servos before disconnecting
            self.send_command("CENTER")
            time.sleep(0.5)
            self.serial_connection.close()
            self.connected = False
            print("Arduino servo controller disconnected")
    
    def send_command(self, command):
        """Send command to Arduino"""
        if not self.connected or not self.serial_connection:
            return False
        
        try:
            self.serial_connection.write(f"{command}\n".encode())
            response = self.serial_connection.readline().decode().strip()
            return "OK" in response
        except Exception as e:
            return False
    
    def move_servo(self, channel, angle):
        """Move servo to specific angle"""
        # Apply calibration adjustments
        if channel == self.pan_channel:
            # Apply direction and offset corrections
            calibrated_angle = (angle * self.pan_direction) + self.pan_offset
            calibrated_angle = max(self.pan_min, min(self.pan_max, calibrated_angle))
        elif channel == self.tilt_channel:
            # Apply direction and offset corrections
            calibrated_angle = (angle * self.tilt_direction) + self.tilt_offset
            calibrated_angle = max(self.tilt_min, min(self.tilt_max, calibrated_angle))
        else:
            calibrated_angle = angle
        
        command = f"SERVO,{channel},{calibrated_angle:.1f}"
        # print(f"Sending command: {command}")
        success = self.send_command(command)
        
        if success:
            if channel == self.pan_channel:
                self.current_pan = angle  # Store the requested angle, not calibrated
            elif channel == self.tilt_channel:
                self.current_tilt = angle
        
        return success
    
    def compensate_camera_offset(self, pan_angle, tilt_angle):
        """
        Compensate for camera offset from tilt axis
        When camera is offset from tilt axis, panning causes vertical drift
        """
        if not self.camera_offset_compensation or self.camera_offset_x == 0:
            return pan_angle, tilt_angle
        
        # Calculate tilt compensation based on pan angle
        # This is an approximation - may need fine-tuning based on your setup
        tilt_compensation = pan_angle * (self.camera_offset_x / 1000.0)  # Rough approximation
        compensated_tilt = tilt_angle - tilt_compensation
        
        return pan_angle, compensated_tilt

    def move_to_click(self, click_x, click_y, frame_width, frame_height):
        """Simple, direct movement to click position with offset compensation"""
        # Calculate frame center
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Calculate errors (distance from center in pixels)
        pan_error = click_x - center_x
        tilt_error = click_y - center_y
        
        # Simple conversion: pixels to degrees
        pan_degrees = pan_error * 0.1  # 10 pixels = 1 degree
        tilt_degrees = -tilt_error * 0.1  # negative for Y axis (up/down flip)
        
        # Apply camera offset compensation
        pan_degrees, tilt_degrees = self.compensate_camera_offset(pan_degrees, tilt_degrees)
        
        # Limit to servo ranges
        pan_degrees = max(-45, min(45, pan_degrees))
        tilt_degrees = max(-30, min(30, tilt_degrees))
        
        # Move servos directly to calculated angles
        success = self.move_servo(0, pan_degrees)  # Pan servo on channel 0
        success &= self.move_servo(1, tilt_degrees)  # Tilt servo on channel 1
        
        return success

    def calculate_pan_command(self, pan_error):
        """Convert pan error to servo angle adjustment"""
        print(f"Pan error input: {pan_error} pixels")
        
        # Apply deadzone - don't move if error is small
        if abs(pan_error) < self.deadzone_pixels:
            print(f"Pan error within deadzone ({self.deadzone_pixels} pixels), no movement")
            return self.current_pan
        
        # Convert pixel error to angle adjustment
        angle_adjustment = pan_error * self.error_to_angle_ratio
        print(f"Pan angle adjustment: {angle_adjustment:.2f}°")
        
        # Limit movement speed
        angle_adjustment = max(-self.max_speed, min(self.max_speed, angle_adjustment))
        print(f"Pan angle adjustment (speed limited): {angle_adjustment:.2f}°")
        
        # Calculate new position
        new_angle = self.current_pan + angle_adjustment
        new_angle = max(self.pan_min, min(self.pan_max, new_angle))
        
        print(f"Pan: current={self.current_pan:.1f}°, new={new_angle:.1f}°, change={new_angle-self.current_pan:.1f}°")
        return new_angle
    
    def calculate_tilt_command(self, tilt_error):
        """Convert tilt error to servo angle adjustment"""
        print(f"Tilt error input: {tilt_error} pixels")
        
        # Apply deadzone - don't move if error is small
        if abs(tilt_error) < self.deadzone_pixels:
            print(f"Tilt error within deadzone ({self.deadzone_pixels} pixels), no movement")
            return self.current_tilt
        
        # Convert pixel error to angle adjustment (negative because camera Y is inverted)
        angle_adjustment = -tilt_error * self.error_to_angle_ratio
        print(f"Tilt angle adjustment: {angle_adjustment:.2f}°")
        
        # Limit movement speed
        angle_adjustment = max(-self.max_speed, min(self.max_speed, angle_adjustment))
        print(f"Tilt angle adjustment (speed limited): {angle_adjustment:.2f}°")
        
        # Calculate new position
        new_angle = self.current_tilt + angle_adjustment
        new_angle = max(self.tilt_min, min(self.tilt_max, new_angle))
        
        print(f"Tilt: current={self.current_tilt:.1f}°, new={new_angle:.1f}°, change={new_angle-self.current_tilt:.1f}°")
        return new_angle
    
    def move_servos(self, pan_angle, tilt_angle):
        """Move both servos to specified angles"""
        pan_success = self.move_servo(self.pan_channel, pan_angle)
        tilt_success = self.move_servo(self.tilt_channel, tilt_angle)
        
        if pan_success and tilt_success:
            print(f"Servos moved - Pan: {pan_angle:.1f}°, Tilt: {tilt_angle:.1f}°")
        
        return pan_success and tilt_success
    
    def center_servos(self):
        """Center both servos"""
        return self.send_command("CENTER")
    
    def disable_servos(self):
        """Disable servo outputs"""
        return self.send_command("OFF")
    
    def get_status(self):
        """Get current servo status"""
        return {
            'connected': self.connected,
            'pan_angle': self.current_pan,
            'tilt_angle': self.current_tilt,
            'port': self.port
        }
    
    def calibrate_servo(self, channel, test_angles=None):
        """Calibrate servo by testing different positions"""
        if test_angles is None:
            test_angles = [-45, -30, -15, 0, 15, 30, 45]
        
        print(f"Calibrating servo on channel {channel}")
        print("Watch the servo movement and note the actual positions")
        
        for angle in test_angles:
            print(f"Moving to {angle}° - Press Enter to continue...")
            self.move_servo(channel, angle)
            input()  # Wait for user input
        
        print("Calibration test complete")
    
    def set_calibration(self, pan_offset=None, tilt_offset=None, pan_direction=None, tilt_direction=None, camera_offset_x=None):
        """Set calibration parameters"""
        if pan_offset is not None:
            self.pan_offset = pan_offset
            print(f"Pan offset set to {pan_offset}°")
        
        if tilt_offset is not None:
            self.tilt_offset = tilt_offset
            print(f"Tilt offset set to {tilt_offset}°")
        
        if pan_direction is not None:
            self.pan_direction = pan_direction
            print(f"Pan direction set to {'normal' if pan_direction == 1 else 'reversed'}")
        
        if tilt_direction is not None:
            self.tilt_direction = tilt_direction
            print(f"Tilt direction set to {'normal' if tilt_direction == 1 else 'reversed'}")
        
        if camera_offset_x is not None:
            self.camera_offset_x = camera_offset_x
            print(f"Camera offset X set to {camera_offset_x} pixels")

    def print_calibration_status(self):
        """Print current calibration settings"""
        print("\n=== Current Calibration Settings ===")
        print(f"Pan direction: {'reversed' if self.pan_direction == -1 else 'normal'} ({self.pan_direction})")
        print(f"Tilt direction: {'reversed' if self.tilt_direction == -1 else 'normal'} ({self.tilt_direction})")
        print(f"Pan offset: {self.pan_offset}°")
        print(f"Tilt offset: {self.tilt_offset}°")
        print(f"Camera offset X: {self.camera_offset_x} pixels")
        print(f"Offset compensation: {'enabled' if self.camera_offset_compensation else 'disabled'}")
        print(f"Error to angle ratio: {self.error_to_angle_ratio} degrees/pixel")
        print("=" * 40)

    def run_calibration_wizard(self):
        """Interactive calibration wizard to help set up servo parameters"""
        print("\n=== Servo Calibration Wizard ===")
        print("This will help you calibrate your servos for accurate positioning.")
        print("Make sure your servos are properly connected and powered.")
        
        # Start with center position
        print("\nStep 1: Centering servos...")
        self.center_servos()
        input("Press Enter when you've observed the center position...")
        
        # Test pan servo
        print("\nStep 2: Testing PAN servo (channel 0)")
        pan_tests = [
            (0, "CENTER"),
            (45, "RIGHT 45°"),
            (-45, "LEFT 45°"),
            (90, "RIGHT 90°"),
            (-90, "LEFT 90°"),
            (0, "BACK TO CENTER")
        ]
        
        for angle, description in pan_tests:
            print(f"Moving pan to {description}...")
            self.move_servo(0, angle)
            response = input(f"Does the camera point {description}? (y/n/q to quit): ").lower()
            if response == 'q':
                return
            elif response == 'n':
                print("Note: Pan servo may need calibration adjustment")
        
        # Test tilt servo
        print("\nStep 3: Testing TILT servo (channel 1)")
        tilt_tests = [
            (0, "CENTER"),
            (30, "UP 30°"),
            (-30, "DOWN 30°"),
            (45, "UP 45°"),
            (-45, "DOWN 45°"),
            (0, "BACK TO CENTER")
        ]
        
        for angle, description in tilt_tests:
            print(f"Moving tilt to {description}...")
            self.move_servo(1, angle)
            response = input(f"Does the camera tilt {description}? (y/n/q to quit): ").lower()
            if response == 'q':
                return
            elif response == 'n':
                print("Note: Tilt servo may need calibration adjustment")
        
        print("\n=== Calibration Complete ===")
        print("If movements were incorrect, you may need to adjust:")
        print("1. Servo direction (set to -1 to reverse)")
        print("2. Servo offset (add/subtract degrees)")
        print("3. Arduino pulse width ranges (SERVO_MIN/SERVO_MAX)")
        
        # Offer to set basic calibration
        print("\nWould you like to set basic calibration adjustments?")
        if input("Set pan direction to reverse? (y/n): ").lower() == 'y':
            self.set_calibration(pan_direction=-1)
        
        if input("Set tilt direction to reverse? (y/n): ").lower() == 'y':
            self.set_calibration(tilt_direction=-1)
        
        pan_offset = input("Enter pan offset in degrees (0 for none): ")
        if pan_offset and pan_offset != '0':
            try:
                self.set_calibration(pan_offset=float(pan_offset))
            except ValueError:
                print("Invalid offset value")
        
        tilt_offset = input("Enter tilt offset in degrees (0 for none): ")
        if tilt_offset and tilt_offset != '0':
            try:
                self.set_calibration(tilt_offset=float(tilt_offset))
            except ValueError:
                print("Invalid offset value")
        
        print("Calibration wizard complete!")
    
    def debug_tracking_movement(self, click_x, click_y, frame_width, frame_height):
        """Debug tracking movement calculations"""
        print(f"\n=== Debug Tracking Movement ===")
        print(f"Click position: ({click_x}, {click_y})")
        print(f"Frame size: {frame_width}x{frame_height}")
        
        # Calculate frame center
        center_x = frame_width // 2
        center_y = frame_height // 2
        print(f"Frame center: ({center_x}, {center_y})")
        
        # Calculate errors (how far from center)
        pan_error = click_x - center_x  # Positive = right of center
        tilt_error = click_y - center_y  # Positive = below center
        
        print(f"Raw errors - Pan: {pan_error}, Tilt: {tilt_error}")
        print(f"Error interpretation:")
        print(f"  Pan error {pan_error}: {'RIGHT' if pan_error > 0 else 'LEFT'} of center")
        print(f"  Tilt error {tilt_error}: {'BELOW' if tilt_error > 0 else 'ABOVE'} center")
        
        # Calculate servo movements
        pan_angle = self.calculate_pan_command(pan_error)
        tilt_angle = self.calculate_tilt_command(tilt_error)
        
        print(f"\nCalculated servo angles:")
        print(f"  Pan angle: {pan_angle:.1f}° ({'RIGHT' if pan_angle > self.current_pan else 'LEFT'})")
        print(f"  Tilt angle: {tilt_angle:.1f}° ({'UP' if tilt_angle > self.current_tilt else 'DOWN'})")
        
        return pan_angle, tilt_angle
    
    def test_coordinate_system(self):
        """Test the coordinate system understanding"""
        print("\n=== Coordinate System Test ===")
        print("This will test if we understand the coordinate system correctly")
        
        # Simulate frame dimensions (adjust to match your actual camera)
        frame_width = 1920  # Adjust to your camera resolution
        frame_height = 1080
        
        test_points = [
            (frame_width//2, frame_height//2, "CENTER"),
            (frame_width//4, frame_height//2, "LEFT of center"),
            (3*frame_width//4, frame_height//2, "RIGHT of center"),
            (frame_width//2, frame_height//4, "ABOVE center"),
            (frame_width//2, 3*frame_height//4, "BELOW center"),
            (frame_width//4, frame_height//4, "UPPER LEFT"),
            (3*frame_width//4, frame_height//4, "UPPER RIGHT"),
            (frame_width//4, 3*frame_height//4, "LOWER LEFT"),
            (3*frame_width//4, 3*frame_height//4, "LOWER RIGHT")
        ]
        
        for x, y, description in test_points:
            print(f"\nTesting click at {description}: ({x}, {y})")
            pan_angle, tilt_angle = self.debug_tracking_movement(x, y, frame_width, frame_height)
            
            if input("Execute this movement? (y/n): ").lower() == 'y':
                self.move_servos(pan_angle, tilt_angle)
                time.sleep(2)
                print("Movement executed. Does it point in the correct direction?")
                input("Press Enter to continue...")
    
    def test_pulse_widths(self):
        """Test different pulse widths to find servo limits"""
        print("WARNING: This will send raw pulse width commands.")
        print("Monitor your servos carefully and stop if they strain against limits.")
        
        channel = int(input("Enter servo channel to test (0 or 1): "))
        
        # Test common pulse width values
        test_values = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
        
        for pulse in test_values:
            print(f"Testing pulse width {pulse}...")
            # Send raw pulse command to Arduino (you'd need to add this command to Arduino code)
            command = f"PULSE,{channel},{pulse}"
            self.send_command(command)
            input(f"Pulse {pulse} - Press Enter to continue or 'q' to quit: ")
            if input() == 'q':
                break
    
    def diagnostic_test(self):
        """Run a diagnostic test to identify calibration issues"""
        print("\n=== Servo Diagnostic Test ===")
        print("This test will help identify why servos aren't pointing correctly")
        
        if not self.connected:
            print("Error: Not connected to Arduino")
            return
        
        # Test each servo individually with known positions
        print("\nTesting PAN servo (Channel 0):")
        pan_test_angles = [0, 45, -45, 90, -90]
        
        for angle in pan_test_angles:
            print(f"\n--- Testing PAN angle: {angle}° ---")
            self.move_servo(0, angle)
            time.sleep(2)  # Give servo time to move
            
            # Ask user to observe and report
            actual_position = input(f"What direction is the camera pointing? (left/center/right or angle): ")
            print(f"Expected: {angle}°, Observed: {actual_position}")
        
        print("\nTesting TILT servo (Channel 1):")
        tilt_test_angles = [0, 30, -30, 45, -45]
        
        for angle in tilt_test_angles:
            print(f"\n--- Testing TILT angle: {angle}° ---")
            self.move_servo(1, angle)
            time.sleep(2)  # Give servo time to move
            
            # Ask user to observe and report
            actual_position = input(f"What direction is the camera tilted? (up/center/down or angle): ")
            print(f"Expected: {angle}°, Observed: {actual_position}")
        
        # Test pulse width mapping
        print("\n--- Pulse Width Mapping Test ---")
        print("The Arduino maps -90° to 150 pulse width, +90° to 600 pulse width")
        print("Center (0°) should be around 375 pulse width")
        
        # Return to center
        print("\nReturning to center...")
        self.center_servos()
        
        print("\nDiagnostic complete! Based on the results:")
        print("1. If directions are opposite, use set_calibration() to reverse direction")
        print("2. If angles are off by a constant amount, use offset calibration")
        print("3. If range is wrong, check Arduino SERVO_MIN/SERVO_MAX values")
    
    def debug_fixed_position_issue(self):
        """Debug why camera always goes to the same position"""
        print("\n=== Fixed Position Issue Debug ===")
        print("This will help identify why the camera always goes to (961, 541)")
        
        # Test what happens with the problematic coordinate
        print("Testing the problematic coordinate (961, 541):")
        
        # Use typical frame dimensions - adjust these to match your camera
        frame_width = 1920
        frame_height = 1080
        
        # Calculate what should happen with this coordinate
        center_x = frame_width // 2  # 960
        center_y = frame_height // 2  # 540
        
        print(f"Frame center: ({center_x}, {center_y})")
        print(f"Click at: (961, 541)")
        
        pan_error = 961 - center_x  # Should be 1
        tilt_error = 541 - center_y  # Should be 1
        
        print(f"Expected pan error: {pan_error}")
        print(f"Expected tilt error: {tilt_error}")
        
        # Test with different click positions
        test_clicks = [
            (100, 100, "Upper left corner"),
            (960, 540, "Center"),
            (1800, 980, "Lower right corner"),
            (100, 980, "Lower left corner"),
            (1800, 100, "Upper right corner")
        ]
        
        for x, y, description in test_clicks:
            print(f"\n--- Testing click at {description}: ({x}, {y}) ---")
            pan_error = x - center_x
            tilt_error = y - center_y
            
            print(f"Pan error: {pan_error} ({'RIGHT' if pan_error > 0 else 'LEFT'})")
            print(f"Tilt error: {tilt_error} ({'DOWN' if tilt_error > 0 else 'UP'})")
            
            # Calculate angles
            pan_angle = self.calculate_pan_command(pan_error)
            tilt_angle = self.calculate_tilt_command(tilt_error)
            
            print(f"Resulting angles - Pan: {pan_angle:.1f}°, Tilt: {tilt_angle:.1f}°")
        
        print("\n=== Potential Issues to Check ===")
        print("1. Are you always getting the same click coordinates (961, 541)?")
        print("2. Is the frame size calculation correct?")
        print("3. Are the servo directions reversed?")
        print("4. Is there a bug in the tracking code that always sends the same coordinates?")
        
        # Test current servo positions
        print(f"\nCurrent servo positions:")
        print(f"Pan: {self.current_pan}°")
        print(f"Tilt: {self.current_tilt}°")
        
        # Reset servos to known positions
        print("\nResetting servos to center...")
        self.center_servos()
        self.current_pan = 0
        self.current_tilt = 0
    
    def track_to_position(self, click_x, click_y, frame_width, frame_height, debug=True):
        """Track to a specific position with debugging"""
        if debug:
            print(f"\n=== Tracking to Position ===")
            print(f"Click: ({click_x}, {click_y})")
            print(f"Frame: {frame_width}x{frame_height}")
            
        # Calculate center
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Calculate errors
        pan_error = click_x - center_x
        tilt_error = click_y - center_y
        
        if debug:
            print(f"Center: ({center_x}, {center_y})")
            print(f"Errors: Pan={pan_error}, Tilt={tilt_error}")
        
        # Calculate servo movements
        pan_angle = self.calculate_pan_command(pan_error)
        tilt_angle = self.calculate_tilt_command(tilt_error)
        
        # Move servos
        success = self.move_servos(pan_angle, tilt_angle)
        
        return success

# Test function
if __name__ == "__main__":
    controller = ArduinoServoController()
    
    if controller.connect():
        print("Testing servo movements...")
        
        # Center servos
        controller.center_servos()
        time.sleep(1)
        
        # Test pan movement
        controller.move_servo(0, 30)  # Pan right
        time.sleep(1)
        controller.move_servo(0, -30)  # Pan left
        time.sleep(1)
        
        # Test tilt movement
        controller.move_servo(1, 20)  # Tilt up
        time.sleep(1)
        controller.move_servo(1, -20)  # Tilt down
        time.sleep(1)
        
        # Return to center
        controller.center_servos()
        
        controller.disconnect()
    else:
        print("Failed to connect to Arduino")