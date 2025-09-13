#!/usr/bin/env python3
"""
Pan-Tilt Tracking Camera - Main Application
YOLO-based person tracking with USB camera and servo control
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.usb_camera import USBCamera
from src.servo_controller import ArduinoServoController
from src.calibration import CameraServoCalibrator
from src.yolo_tracker import YOLOTracker
from src.tracking_logger import TrackingLogger
import numpy as np
import cv2
import time
from collections import deque
import threading
import argparse
import json


def load_config(config_path="config/config.json"):
    """Load configuration from JSON file"""
    default_config = {
        "camera": {
            "index": 0,
            "resolution": [1920, 1080],
            "fps": 30
        },
        "servo": {
            "port": "/dev/ttyACM0",
            "baudrate": 115200,
            "inverted_pan": True # Our pan servo is inverted (default True)
        },
        "tracking": {
            "model_path": "yolov8n.pt",
            "confidence_threshold": 0.5,
            "dead_zone": 50
        },
        "display": {
            "window_size": [1280, 720]
        }
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
                return config
    except Exception as e:
        print(f"Error loading config: {e}")
    
    return default_config


class PanTiltYOLOTracker:
    """Main pan-tilt tracking system using YOLO for person detection"""
    
    def __init__(self, config=None):
        """Initialize the tracking system"""
        if config is None:
            config = load_config()
        
        print("Initializing Pan-Tilt YOLO Tracking System...")
        
        # Camera setup
        camera_config = config.get('camera', {})
        self.camera = USBCamera(
            camera_index=camera_config.get('index', 0),
            resolution=tuple(camera_config.get('resolution', [1920, 1080])),
            fps=camera_config.get('fps', 30)
        )
        
        # Servo controller setup
        servo_config = config.get('servo', {})
        self.servo_controller = ArduinoServoController(
            port=servo_config.get('port', '/dev/ttyACM0'),
            baudrate=servo_config.get('baudrate', 115200),
            inverted_pan=servo_config.get('inverted_pan', True)
        )
        
        # YOLO tracker setup
        tracking_config = config.get('tracking', {})
        self.yolo_tracker = YOLOTracker(
            model_path=tracking_config.get('model_path', 'yolov8n.pt'),
            confidence_threshold=tracking_config.get('confidence_threshold', 0.5)
        )
        
        # Calibration setup
        self.calibrator = CameraServoCalibrator("config/calibration.json")
        self.calibrator.load_calibration()
        
        # Tracking parameters
        self.dead_zone = tracking_config.get('dead_zone', 50)
        self.frame_center = (
            camera_config.get('resolution', [1920, 1080])[0] // 2,
            camera_config.get('resolution', [1920, 1080])[1] // 2
        )
        self.calibrator.frame_center = self.frame_center
        
        # Control variables
        self.running = False
        self.control_thread = None
        self.target_position = None
        self.tracking_enabled = True
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        
        # Initialize tracking logger
        experiment_name = config.get('experiment_name', None)
        self.tracking_logger = TrackingLogger(log_dir="logs", experiment_name=experiment_name)
        
        print("Pan-Tilt YOLO Tracker initialized")
    
    def start(self):
        """Start the tracking system"""
        print("Starting Pan-Tilt YOLO Tracking System...")
        
        # Open camera
        if not self.camera.open():
            print("Failed to open camera")
            return False
        
        if not self.camera.start_capture():
            print("Failed to start camera capture")
            return False
        
        # Connect servo controller
        if not self.servo_controller.connect():
            print("Warning: Failed to connect to servo controller (running in simulation mode)")
        else:
            # Center servos at startup
            self.servo_controller.center_servos()
            time.sleep(1)
        
        # Start control thread
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        
        print("System started successfully!")
        return True
    
    def stop(self):
        """Stop the tracking system"""
        print("Stopping tracking system...")
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2)
        
        self.servo_controller.disconnect()
        self.camera.close()
        cv2.destroyAllWindows()
        print("System stopped")
    
    def control_loop(self):
        """Main servo control loop"""
        while self.running:
            if self.target_position and self.tracking_enabled:
                try:
                    # DEBUG: Log target position (bounding box)
                    print(f"DEBUG: Target position (bbox center): ({self.target_position[0]}, {self.target_position[1]})")
                    print(f"DEBUG: Frame center: ({self.frame_center[0]}, {self.frame_center[1]})")
                    
                    # Calculate pixel error from center
                    pixel_error_x = self.target_position[0] - self.frame_center[0]
                    pixel_error_y = self.target_position[1] - self.frame_center[1]
                    print(f"DEBUG: Pixel error: X={pixel_error_x}, Y={pixel_error_y}")
                    
                    # Log quadrant information
                    quadrant = ""
                    if pixel_error_x > 0 and pixel_error_y < 0:
                        quadrant = "TOP-RIGHT"
                        expected_movement = "Camera should turn RIGHT and UP"
                    elif pixel_error_x < 0 and pixel_error_y < 0:
                        quadrant = "TOP-LEFT"
                        expected_movement = "Camera should turn LEFT and UP"
                    elif pixel_error_x > 0 and pixel_error_y > 0:
                        quadrant = "BOTTOM-RIGHT"
                        expected_movement = "Camera should turn RIGHT and DOWN"
                    elif pixel_error_x < 0 and pixel_error_y > 0:
                        quadrant = "BOTTOM-LEFT"
                        expected_movement = "Camera should turn LEFT and DOWN"
                    print(f"DEBUG: Target is in {quadrant} quadrant - {expected_movement}")
                    
                    # Convert pixel coordinates to servo angles
                    target_pan, target_tilt = self.calibrator.pixel_to_servo(
                        self.target_position[0], self.target_position[1]
                    )
                    print(f"DEBUG: Calculated servo angles: Pan={target_pan:.1f}, Tilt={target_tilt:.1f}")
                    
                    # Apply servo limits
                    target_pan = np.clip(target_pan, -90, 90)
                    target_tilt = np.clip(target_tilt, -45, 45)
                    
                    # Get current servo positions
                    current_pan = self.servo_controller.current_pan
                    current_tilt = self.servo_controller.current_tilt
                    print(f"DEBUG: Current servo positions: Pan={current_pan:.1f}, Tilt={current_tilt:.1f}")
                    
                    # Calculate smooth movement
                    max_step = 2.0  # degrees per control cycle
                    pan_diff = target_pan - current_pan
                    tilt_diff = target_tilt - current_tilt
                    print(f"DEBUG: Servo movement: Pan diff={pan_diff:.1f}, Tilt diff={tilt_diff:.1f}")
                    
                    # Limit movement speed
                    if abs(pan_diff) > max_step:
                        pan_diff = max_step if pan_diff > 0 else -max_step
                    if abs(tilt_diff) > max_step:
                        tilt_diff = max_step if tilt_diff > 0 else -max_step
                    
                    new_pan = current_pan + pan_diff
                    new_tilt = current_tilt + tilt_diff
                    print(f"DEBUG: New servo positions: Pan={new_pan:.1f}, Tilt={new_tilt:.1f}")
                    
                    # Expected camera movement direction
                    pan_direction = "RIGHT" if new_pan > current_pan else "LEFT"
                    tilt_direction = "UP" if new_tilt > current_tilt else "DOWN"
                    print(f"DEBUG: Expected camera movement: Pan={pan_direction}, Tilt={tilt_direction}")
                    
                    # Move servos if movement is significant
                    if abs(pan_diff) > 0.5 or abs(tilt_diff) > 0.5:
                        self.servo_controller.move_servos(new_pan, new_tilt)
                        
                        # Log servo command with detailed information
                        self.tracking_logger.log_servo_command(
                            pan_angle=new_pan,
                            tilt_angle=new_tilt,
                            current_pan=current_pan,
                            current_tilt=current_tilt,
                            target_position=self.target_position,
                            frame_center=self.frame_center
                        )
                
                except Exception as e:
                    print(f"Control loop error: {e}")
            
            time.sleep(0.05)  # 20Hz control rate
    
    def calculate_pan_tilt_error(self, target_pos):
        """Calculate pan and tilt error from center"""
        pan_error = target_pos[0] - self.frame_center[0]
        tilt_error = target_pos[1] - self.frame_center[1]
        return pan_error, tilt_error
    
    def process_frame(self, frame):
        """Process frame for tracking and visualization"""
        start_time = time.time()
        
        # Run YOLO detection
        detections = self.yolo_tracker.detect_objects(frame)
        
        # Update tracking
        current_target = self.yolo_tracker.update_tracking(detections)
        
        # Get smoothed target position
        if current_target:
            smoothed_pos = self.yolo_tracker.get_smoothed_target_position()
            if smoothed_pos:
                # Check dead zone
                pan_error, tilt_error = self.calculate_pan_tilt_error(smoothed_pos)
                if abs(pan_error) > self.dead_zone or abs(tilt_error) > self.dead_zone:
                    self.target_position = smoothed_pos
                else:
                    # Person is in dead zone - stop tracking to avoid jitter
                    self.target_position = None
        else:
            self.target_position = None
        
        # Draw visualization
        vis_frame = self.yolo_tracker.draw_detections(frame, detections)
        
        # Draw UI elements
        self.draw_ui(vis_frame)
        
        # Calculate FPS
        frame_time = time.time() - start_time
        self.fps_counter.append(frame_time)
        processing_time_ms = frame_time * 1000  # Convert to milliseconds
        
        # Log frame data for performance evaluation
        if current_target:
            # Convert current_target to dict for logging
            target_dict = {
                'class_name': current_target.class_name,
                'confidence': current_target.confidence,
                'bbox': current_target.bbox,
                'center': current_target.center,
                'smoothed_center': smoothed_pos if smoothed_pos else current_target.center,
                'tracking_enabled': self.tracking_enabled
            }
            
            # Get servo data
            servo_data = {
                'current_pan': self.servo_controller.current_pan,
                'current_tilt': self.servo_controller.current_tilt,
                'target_pan': 0.0,
                'target_tilt': 0.0,
                'command_pan': 0.0,
                'command_tilt': 0.0
            }
            
            # If we have a valid target position, calculate target pan/tilt angles
            if self.target_position:
                try:
                    target_pan, target_tilt = self.calibrator.pixel_to_servo(
                        self.target_position[0], self.target_position[1]
                    )
                    servo_data['target_pan'] = target_pan
                    servo_data['target_tilt'] = target_tilt
                except:
                    pass
            
            # Log the detection
            self.tracking_logger.log_detection(
                target_dict,
                self.frame_center,
                servo_data,
                processing_time_ms
            )
        else:
            # Log frame with no detection
            frame_data = {
                'camera_center_x': self.frame_center[0],
                'camera_center_y': self.frame_center[1],
                'target_detected': False,
                'current_pan': self.servo_controller.current_pan,
                'current_tilt': self.servo_controller.current_tilt,
                'tracking_enabled': self.tracking_enabled
            }
            self.tracking_logger.log_frame(frame_data, processing_time_ms)
        
        return vis_frame
    
    def draw_ui(self, frame):
        """Draw user interface elements"""
        # Draw center crosshair
        cv2.circle(frame, self.frame_center, 5, (255, 0, 0), -1)
        cv2.circle(frame, self.frame_center, self.dead_zone, (255, 0, 0), 2)
        
        # Status information
        status_y = 30
        line_height = 30
        
        # FPS
        if len(self.fps_counter) > 0:
            avg_fps = 1.0 / (sum(self.fps_counter) / len(self.fps_counter))
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            status_y += line_height
        
        # Tracking status
        tracking_status = "ON" if self.tracking_enabled else "OFF"
        color = (0, 255, 0) if self.tracking_enabled else (0, 0, 255)
        cv2.putText(frame, f"Tracking: {tracking_status}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        status_y += line_height
        
        # Target information
        if self.target_position:
            pan_error, tilt_error = self.calculate_pan_tilt_error(self.target_position)
            cv2.putText(frame, f"Target: ({self.target_position[0]}, {self.target_position[1]})", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            status_y += line_height
            
            cv2.putText(frame, f"Error: Pan={pan_error:4.0f} Tilt={tilt_error:4.0f}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            status_y += line_height
            
            # Show servo angles if calibrated
            try:
                target_pan, target_tilt = self.calibrator.pixel_to_servo(
                    self.target_position[0], self.target_position[1]
                )
                cv2.putText(frame, f"Servo Target: Pan={target_pan:.1f}° Tilt={target_tilt:.1f}°", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                status_y += line_height
            except:
                pass
        
        # Servo status
        if self.servo_controller.connected:
            current_pan = self.servo_controller.current_pan
            current_tilt = self.servo_controller.current_tilt
            cv2.putText(frame, f"Servos: Pan={current_pan:.1f}° Tilt={current_tilt:.1f}°", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            cv2.putText(frame, "Servos: DISCONNECTED", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def run(self):
        """Main run loop"""
        print("Starting main tracking loop...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset tracking") 
        print("  't' - Toggle tracking on/off")
        print("  'c' - Center servos")
        print("  's' - Save current frame")
        
        cv2.namedWindow("Pan-Tilt YOLO Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pan-Tilt YOLO Tracker", 1280, 720)
        
        try:
            while self.running:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # Process frame
                vis_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow("Pan-Tilt YOLO Tracker", vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.yolo_tracker.reset_tracking()
                    self.target_position = None
                    print("Tracking reset")
                elif key == ord('t'):
                    self.tracking_enabled = not self.tracking_enabled
                    print(f"Tracking {'enabled' if self.tracking_enabled else 'disabled'}")
                elif key == ord('c'):
                    self.servo_controller.center_servos()
                    print("Servos centered")
                elif key == ord('s'):
                    # Save the frame in the current experiment directory
                    logs_dir = self.tracking_logger.log_dir
                    
                    # Create frames subdirectory in the experiment folder
                    frames_dir = os.path.join(logs_dir, "frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    # Create filename with timestamp
                    timestamp = int(time.time())
                    filename = f"frame_{timestamp}.jpg"
                    filepath = os.path.join(frames_dir, filename)
                    
                    # Add additional debug info to the frame before saving
                    debug_frame = vis_frame.copy()
                    
                    # Add timestamp and current servo positions to the image
                    timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                    cv2.putText(debug_frame, f"Time: {timestamp_str}", (10, debug_frame.shape[0] - 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Add servo positions if available
                    if hasattr(self.servo_controller, 'current_pan') and hasattr(self.servo_controller, 'current_tilt'):
                        pan_position = self.servo_controller.current_pan
                        tilt_position = self.servo_controller.current_tilt
                        cv2.putText(debug_frame, f"Pan: {pan_position:.1f}°, Tilt: {tilt_position:.1f}°", 
                                  (10, debug_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Add coordinates of target if available
                    if self.target_position:
                        x, y = self.target_position
                        cv2.putText(debug_frame, f"Target: ({x}, {y})", 
                                  (10, debug_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Save the enhanced visualization frame
                    cv2.imwrite(filepath, debug_frame)
                    print(f"Frame saved with tracking data as {filepath}")
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Pan-Tilt YOLO Tracking Camera')
    parser.add_argument('--config', '-c', type=str, default='config/config.json',
                       help='Configuration file path')
    parser.add_argument('--camera', type=int, help='Camera index override')
    parser.add_argument('--model', type=str, help='YOLO model path override')
    parser.add_argument('--experiment', '-e', type=str, 
                       help='Name for this experiment run (for logging)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.camera is not None:
        config['camera']['index'] = args.camera
    if args.model:
        config['tracking']['model_path'] = args.model
    if args.experiment:
        config['experiment_name'] = args.experiment
    
    # Create and run tracker
    tracker = PanTiltYOLOTracker(config=config)
    
    if tracker.start():
        try:
            tracker.run()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Failed to start tracking system")


if __name__ == "__main__":
    main()
