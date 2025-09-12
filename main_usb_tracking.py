#!/usr/bin/env python3
"""
Pan-Tilt Tracking Camera - Main Application
Using USB Camera (U20CAM-1080P-1) for object tracking and pan-tilt control
"""

from usb_camera import USBCamera
from servo_controller import ArduinoServoController
from calibration import CameraServoCalibrator
import numpy as np
import cv2
import mediapipe as mp
import time
from collections import deque
import threading
import argparse
import json
import os

def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return {}

class PanTiltTracker:
    def __init__(self, config=None):
        """Initialize the pan-tilt tracking system"""
        # Load configuration
        if config is None:
            config = load_config()
        
        # Camera configuration
        camera_config = config.get('camera', {})
        camera_index = camera_config.get('index', 0)
        resolution = tuple(camera_config.get('resolution', [1920, 1080]))
        target_fps = camera_config.get('fps', 30)
        
        self.camera = USBCamera(camera_index=camera_index, resolution=resolution, fps=target_fps)
        
        # Servo controller configuration
        servo_config = config.get('servo', {})
        servo_port = servo_config.get('port', '/dev/ttyUSB0')
        servo_baudrate = servo_config.get('baudrate', 115200)
        
        self.servo_controller = ArduinoServoController(
            port=servo_port, 
            baudrate=servo_baudrate
        )
        
        # Initialize calibrator for pixel-to-servo conversion
        self.calibrator = CameraServoCalibrator("calibration.json")
        self.calibrator.load_calibration()
        
        # Set frame center for calibrator
        self.calibrator.frame_center = (resolution[0] // 2, resolution[1] // 2)
        
        # Detection configuration
        detection_config = config.get('detection', {})
        model_complexity = detection_config.get('model_complexity', 1)
        min_detection_confidence = detection_config.get('min_detection_confidence', 0.5)
        min_tracking_confidence = detection_config.get('min_tracking_confidence', 0.5)
        
        # MediaPipe setup for person detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Pan-tilt configuration
        pantilt_config = config.get('pantilt', {})
        self.dead_zone = pantilt_config.get('dead_zone', 50)
        
        # Display configuration
        display_config = config.get('display', {})
        display_size = display_config.get('window_size', [1280, 720])
        self.frame_center = (resolution[0] // 2, resolution[1] // 2)
        
        # Store config for reference
        self.config = config
        
        # Tracking variables
        self.target_position = None
        self.tracking_history = deque(maxlen=10)
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        
        # Control thread
        self.control_thread = None
        self.running = False
        
    def start(self):
        """Start the tracking system"""
        print("Starting Pan-Tilt Tracking Camera...")
        
        if not self.camera.open():
            print("Failed to open camera")
            return False
        
        if not self.camera.start_capture():
            print("Failed to start camera capture")
            return False
        
        # Connect to servo controller
        if not self.servo_controller.connect():
            print("Failed to connect to servo controller")
            return False
        
        # Center servos at startup
        self.servo_controller.center_servos()
        
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        return True
    
    def stop(self):
        """Stop the tracking system"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()
        self.servo_controller.disconnect()
        self.camera.close()
        cv2.destroyAllWindows()
    
    def detect_person(self, frame):
        """Detect person using MediaPipe Pose"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Get nose landmark for tracking (most stable point)
            nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            
            # Convert normalized coordinates to pixel coordinates
            h, w = frame.shape[:2]
            x = int(nose.x * w)
            y = int(nose.y * h)
            
            return (x, y), results.pose_landmarks
        
        return None, None
    
    def calculate_pan_tilt_error(self, target_pos):
        """Calculate pan and tilt error from center"""
        if target_pos is None:
            return 0, 0
        
        # Calculate error from center
        pan_error = target_pos[0] - self.frame_center[0]
        tilt_error = target_pos[1] - self.frame_center[1]
        
        # Apply dead zone
        if abs(pan_error) < self.dead_zone:
            pan_error = 0
        if abs(tilt_error) < self.dead_zone:
            tilt_error = 0
        
        return pan_error, tilt_error
    
    def control_loop(self):
        """Main control loop for pan-tilt adjustment"""
        while self.running:
            if self.target_position:
                # Use calibrated pixel-to-servo conversion
                target_pan, target_tilt = self.calibrator.pixel_to_servo(
                    self.target_position[0], self.target_position[1]
                )
                
                # Clamp to servo limits
                target_pan = max(-90, min(90, target_pan))
                target_tilt = max(-45, min(45, target_tilt))
                
                # Calculate movement (smooth incremental movement)
                current_pan = self.servo_controller.current_pan
                current_tilt = self.servo_controller.current_tilt
                
                # Smooth movement - limit speed
                max_step = 2.0  # degrees per step
                pan_diff = target_pan - current_pan
                tilt_diff = target_tilt - current_tilt
                
                if abs(pan_diff) > max_step:
                    pan_diff = max_step if pan_diff > 0 else -max_step
                if abs(tilt_diff) > max_step:
                    tilt_diff = max_step if tilt_diff > 0 else -max_step
                
                new_pan = current_pan + pan_diff
                new_tilt = current_tilt + tilt_diff
                
                # Only move if there's significant error
                if abs(pan_diff) > 0.5 or abs(tilt_diff) > 0.5:
                    print(f"Target: ({self.target_position[0]}, {self.target_position[1]}) -> "
                          f"Servo: Pan={new_pan:.1f}°, Tilt={new_tilt:.1f}°")
                    
                    self.servo_controller.move_servos(new_pan, new_tilt)
            
            time.sleep(0.1)  # Control loop rate
    
    def process_frame(self, frame):
        """Process a single frame for tracking"""
        start_time = time.time()
        
        # Detect person
        person_pos, landmarks = self.detect_person(frame)
        
        if person_pos:
            self.target_position = person_pos
            self.tracking_history.append(person_pos)
            
            # Draw tracking point
            cv2.circle(frame, person_pos, 10, (0, 255, 0), -1)
            cv2.putText(frame, f"Target: {person_pos}", 
                       (person_pos[0] + 15, person_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw pose landmarks
            if landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, landmarks, self.mp_pose.POSE_CONNECTIONS)
        else:
            # No person detected
            if len(self.tracking_history) == 0:
                self.target_position = None
        
        # Draw center crosshair and dead zone
        cv2.circle(frame, self.frame_center, 5, (255, 0, 0), -1)
        cv2.circle(frame, self.frame_center, self.dead_zone, (255, 0, 0), 2)
        
        # Calculate and display pan/tilt error
        if self.target_position:
            pan_error, tilt_error = self.calculate_pan_tilt_error(self.target_position)
            target_pan, target_tilt = self.calibrator.pixel_to_servo(
                self.target_position[0], self.target_position[1]
            )
            
            cv2.putText(frame, f"Error: Pan={pan_error:4.0f} Tilt={tilt_error:4.0f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Target: Pan={target_pan:.1f}° Tilt={target_tilt:.1f}°",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show current servo positions
            current_pan = self.servo_controller.current_pan
            current_tilt = self.servo_controller.current_tilt
            cv2.putText(frame, f"Current: Pan={current_pan:.1f}° Tilt={current_tilt:.1f}°",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Calculate FPS
        self.fps_counter.append(time.time() - start_time)
        if len(self.fps_counter) > 0:
            avg_fps = 1.0 / (sum(self.fps_counter) / len(self.fps_counter))
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main run loop"""
        print("Starting tracking... Press 'q' to quit, 's' to save frame")
        
        while self.running:
            frame = self.camera.get_frame()
            if frame is not None:
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Pan-Tilt Tracking Camera', processed_frame)
                self.frame_count += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if frame is not None:
                    filename = f"tracking_frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Saved frame as {filename}")
            elif key == ord('r'):
                # Reset tracking
                self.tracking_history.clear()
                self.target_position = None
                print("Tracking reset")

def main():
    # Load configuration
    config = load_config()
    
    parser = argparse.ArgumentParser(description='Pan-Tilt Tracking Camera')
    parser.add_argument('--camera', '-c', type=int, default=config.get('camera', {}).get('index', 0),
                       help='Camera index (default: from config or 0)')
    parser.add_argument('--fps', '-f', type=int, default=config.get('camera', {}).get('fps', 30),
                       help='Target FPS (default: from config or 30)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path (default: config.json)')
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.config != 'config.json':
        config = load_config(args.config)
    
    # Update config with command line overrides
    if 'camera' not in config:
        config['camera'] = {}
    config['camera']['index'] = args.camera
    config['camera']['fps'] = args.fps
    
    tracker = PanTiltTracker(config=config)
    
    try:
        if tracker.start():
            tracker.run()
        else:
            print("Failed to start tracker")
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        tracker.stop()

if __name__ == "__main__":
    main()