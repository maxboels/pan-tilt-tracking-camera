#!/usr/bin/env python3
"""
Synthetic Tracking Benchmark for Pan-Tilt Camera

This script runs a benchmarking evaluation of the tracking system using 
synthetic detection data. It allows reproducible experiments with controlled
movement patterns.
"""

import sys
import os
# Add parent directory to the Python path to make 'src' importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
import argparse
import json
import random
from collections import deque
import threading

# Import from src module
from src.synthetic_detection import SyntheticDetectionGenerator
from src.servo_controller import ArduinoServoController
from src.calibration import CameraServoCalibrator
from src.yolo_tracker import YOLOTracker
from src.tracking_logger import TrackingLogger


class SyntheticTrackingBenchmark:
    """Benchmark for tracking system using synthetic data"""
    
    def __init__(self, config=None):
        """
        Initialize the benchmark system
        
        Args:
            config: Configuration dictionary
        """
        # Default configuration
        default_config = {
            "simulation": {
                "frame_width": 1920,
                "frame_height": 1080,
                "fps": 30,
                "duration": 60,  # seconds
                "movement_pattern": "circular",
                "pattern_params": {
                    "center_x": 960,
                    "center_y": 540,
                    "radius": 400,
                    "speed": 1,
                    "clockwise": True
                }
            },
            "tracking": {
                "model_path": "yolov8n.pt",
                "confidence_threshold": 0.5,
                "dead_zone": 50
            },
            "servo": {
                "inverted_pan": True
            },
            "experiment_name": f"benchmark_{int(time.time())}"
        }
        
        # Use provided config or default
        self.config = default_config
        if config:
            # Merge with defaults
            for key, value in config.items():
                if key in self.config and isinstance(self.config[key], dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        print("Initializing Synthetic Tracking Benchmark...")
        
        # Set up simulation parameters
        sim_config = self.config['simulation']
        self.frame_width = sim_config['frame_width']
        self.frame_height = sim_config['frame_height']
        self.fps = sim_config['fps']
        self.duration = sim_config['duration']
        self.frame_time = 1.0 / self.fps
        
        # Set up frame center
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        
        # Create the synthetic detection generator
        self.generator = SyntheticDetectionGenerator(
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )
        
        # Set movement pattern
        self.generator.set_movement_pattern(
            sim_config['movement_pattern'],
            **sim_config['pattern_params']
        )
        
        # Set up servo controller (in simulation mode)
        self.servo_controller = ArduinoServoController(
            port="SIMULATION",  # Flag for simulation mode
            inverted_pan=self.config['servo'].get('inverted_pan', True)
        )
        
        # Set up calibrator
        self.calibrator = CameraServoCalibrator("config/calibration.json")
        self.calibrator.load_calibration()
        self.calibrator.frame_center = self.frame_center
        
        # Set up YOLO tracker
        tracking_config = self.config['tracking']
        self.yolo_tracker = YOLOTracker(
            model_path=tracking_config.get('model_path', 'yolov8n.pt'),
            confidence_threshold=tracking_config.get('confidence_threshold', 0.5)
        )
        
        # Control variables
        self.running = False
        self.frame_count = 0
        self.current_time = 0
        self.target_position = None
        self.tracking_enabled = True
        self.dead_zone = tracking_config.get('dead_zone', 50)
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        
        # Initialize tracking logger for this experiment
        self.tracking_logger = TrackingLogger(
            log_dir="logs",
            experiment_name=self.config.get('experiment_name', "synthetic_benchmark")
        )
        
        # Tracking data for evaluation
        self.ground_truth_positions = []  # List of actual target positions
        self.tracked_positions = []  # List of tracked positions
        self.servo_positions = []  # List of servo positions
        
        # Control thread for servo simulation
        self.control_thread = None
        
        print("Synthetic Tracking Benchmark initialized")
    
    def start(self):
        """Start the benchmark system"""
        print("Starting Synthetic Tracking Benchmark...")
        
        # Start servo controller in simulation mode
        self.servo_controller.simulation_mode = True
        self.servo_controller.connected = True
        
        # Center servos at startup
        self.servo_controller.center_servos()
        time.sleep(0.1)
        
        # Start control thread
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()
        
        print("System started successfully!")
        return True
    
    def stop(self):
        """Stop the benchmark system"""
        print("Stopping benchmark...")
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2)
        
        cv2.destroyAllWindows()
        print("System stopped")
    
    def control_loop(self):
        """Simulated servo control loop"""
        while self.running:
            if self.target_position and self.tracking_enabled:
                try:
                    # Calculate pixel error from center
                    pixel_error_x = self.target_position[0] - self.frame_center[0]
                    pixel_error_y = self.target_position[1] - self.frame_center[1]
                    
                    # Convert pixel coordinates to servo angles
                    target_pan, target_tilt = self.calibrator.pixel_to_servo(
                        self.target_position[0], self.target_position[1]
                    )
                    
                    # Apply servo limits
                    target_pan = np.clip(target_pan, -90, 90)
                    target_tilt = np.clip(target_tilt, -45, 45)
                    
                    # Get current servo positions
                    current_pan = self.servo_controller.current_pan
                    current_tilt = self.servo_controller.current_tilt
                    
                    # Calculate smooth movement
                    max_step = 2.0  # degrees per control cycle
                    pan_diff = target_pan - current_pan
                    tilt_diff = target_tilt - current_tilt
                    
                    # Limit movement speed
                    if abs(pan_diff) > max_step:
                        pan_diff = max_step if pan_diff > 0 else -max_step
                    if abs(tilt_diff) > max_step:
                        tilt_diff = max_step if tilt_diff > 0 else -max_step
                    
                    new_pan = current_pan + pan_diff
                    new_tilt = current_tilt + tilt_diff
                    
                    # Move servos if movement is significant
                    if abs(pan_diff) > 0.5 or abs(tilt_diff) > 0.5:
                        self.servo_controller.move_servos(new_pan, new_tilt)
                        
                        # Log servo command
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
    
    def process_frame(self, synthetic_detection):
        """
        Process a synthetic detection as if it came from the YOLO tracker
        
        Args:
            synthetic_detection: Synthetic detection object
        
        Returns:
            Visualization frame
        """
        start_time = time.time()
        
        # Create a list of detections with our synthetic one
        detections = []
        if synthetic_detection:
            # Create a detection object that matches the YOLOTracker.Detection class
            from src.yolo_tracker import Detection
            detection = Detection(
                class_id=synthetic_detection['class_id'],
                class_name=synthetic_detection['class_name'],
                confidence=synthetic_detection['confidence'],
                bbox=synthetic_detection['bbox'],
                center=synthetic_detection['center'],
                area=synthetic_detection['area']
            )
            detections.append(detection)
            
            # Store ground truth position for evaluation
            self.ground_truth_positions.append((self.current_time, synthetic_detection['center']))
        
        # Create a blank frame for visualization
        frame = self.generator.create_synthetic_frame(background_color=(50, 50, 50))
        
        # Update tracking
        current_target = self.yolo_tracker.update_tracking(detections)
        
        # Get smoothed target position
        if current_target:
            smoothed_pos = self.yolo_tracker.get_smoothed_target_position()
            if smoothed_pos:
                # Check dead zone
                pan_error = smoothed_pos[0] - self.frame_center[0]
                tilt_error = smoothed_pos[1] - self.frame_center[1]
                
                if abs(pan_error) > self.dead_zone or abs(tilt_error) > self.dead_zone:
                    self.target_position = smoothed_pos
                    # Store tracked position for evaluation
                    self.tracked_positions.append((self.current_time, smoothed_pos))
                else:
                    # Person is in dead zone - stop tracking to avoid jitter
                    self.target_position = None
                    # Store None for evaluation
                    self.tracked_positions.append((self.current_time, None))
        else:
            self.target_position = None
            # Store None for evaluation
            self.tracked_positions.append((self.current_time, None))
        
        # Store servo position for evaluation
        self.servo_positions.append(
            (self.current_time, 
             (self.servo_controller.current_pan, self.servo_controller.current_tilt))
        )
        
        # Draw visualization
        vis_frame = self.yolo_tracker.draw_detections(frame, detections)
        
        # Draw UI elements (similar to main.py)
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
                except Exception as e:
                    print(f"Error calculating servo angles: {e}")
            
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
        """Draw user interface elements on the frame"""
        # Draw frame center crosshair
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
        
        # Simulation time
        cv2.putText(frame, f"Time: {self.current_time:.1f}s", (10, status_y), 
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
            cv2.putText(frame, f"Target: ({self.target_position[0]}, {self.target_position[1]})", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            status_y += line_height
            
            # Calculate errors
            pan_error = self.target_position[0] - self.frame_center[0]
            tilt_error = self.target_position[1] - self.frame_center[1]
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
            except Exception as e:
                print(f"Error drawing UI: {e}")
        
        # Servo status
        current_pan = self.servo_controller.current_pan
        current_tilt = self.servo_controller.current_tilt
        cv2.putText(frame, f"Servos: Pan={current_pan:.1f}° Tilt={current_tilt:.1f}°", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        status_y += line_height
        
        # Movement pattern
        cv2.putText(frame, f"Pattern: {self.config['simulation']['movement_pattern']}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def run(self):
        """Run the benchmark simulation"""
        print("Starting benchmark simulation...")
        
        # Show benchmark information
        print(f"Movement pattern: {self.config['simulation']['movement_pattern']}")
        print(f"Duration: {self.duration} seconds")
        print(f"Frame rate: {self.fps} FPS")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"Experiment: {self.config['experiment_name']}")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset tracking") 
        print("  't' - Toggle tracking on/off")
        print("  'c' - Center servos")
        print("  's' - Save current frame")
        
        cv2.namedWindow("Synthetic Tracking Benchmark", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Synthetic Tracking Benchmark", 1280, 720)
        
        start_time = time.time()
        
        try:
            while self.running:
                # Check if we've reached the duration
                self.current_time = time.time() - start_time
                if self.current_time >= self.duration:
                    print("Benchmark duration completed")
                    break
                
                # Update position for this frame
                pattern_complete = self.generator.update_position()
                if pattern_complete:
                    print("Movement pattern completed")
                    break
                
                # Get synthetic detection
                detection = self.generator.get_detection()
                
                # Process frame
                vis_frame = self.process_frame(detection)
                
                # Display
                cv2.imshow("Synthetic Tracking Benchmark", vis_frame)
                
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
                    # Save the frame
                    frames_dir = os.path.join(self.tracking_logger.log_dir, "frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    filename = f"frame_{int(time.time())}.jpg"
                    filepath = os.path.join(frames_dir, filename)
                    cv2.imwrite(filepath, vis_frame)
                    print(f"Frame saved as {filepath}")
                
                self.frame_count += 1
                
                # Control frame rate
                elapsed = time.time() - start_time - self.current_time
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop()
            
            # Save evaluation data
            self._save_evaluation_data()
            
            # Generate performance report
            self._generate_performance_report()
    
    def _save_evaluation_data(self):
        """Save tracking evaluation data"""
        eval_data = {
            'ground_truth': self.ground_truth_positions,
            'tracked_positions': self.tracked_positions,
            'servo_positions': self.servo_positions,
            'config': self.config
        }
        
        # Save to experiment directory
        eval_path = os.path.join(self.tracking_logger.log_dir, "evaluation_data.json")
        
        with open(eval_path, 'w') as f:
            # Convert numpy values and tuples to lists for JSON serialization
            import json
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, tuple):
                        return list(obj)
                    return json.JSONEncoder.default(self, obj)
            
            json.dump(eval_data, f, cls=NumpyEncoder)
        
        print(f"Evaluation data saved to {eval_path}")
    
    def _generate_performance_report(self):
        """Generate performance report"""
        # Calculate tracking accuracy metrics
        total_frames = len(self.ground_truth_positions)
        if total_frames == 0:
            print("No frames recorded for evaluation")
            return
        
        # Match up ground truth with tracking results
        errors = []
        for (gt_time, gt_pos), (tr_time, tr_pos) in zip(
            self.ground_truth_positions, self.tracked_positions):
            
            if tr_pos is None:
                # Target was not tracked in this frame
                continue
                
            # Calculate error in pixels
            error_x = gt_pos[0] - tr_pos[0]
            error_y = gt_pos[1] - tr_pos[1]
            error_distance = np.sqrt(error_x ** 2 + error_y ** 2)
            
            errors.append(error_distance)
        
        if not errors:
            print("No tracking matches found for evaluation")
            return
            
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        # Calculate tracking rate
        tracking_rate = len(errors) / total_frames
        
        # Save report
        report = {
            'total_frames': total_frames,
            'frames_with_tracking': len(errors),
            'tracking_rate': tracking_rate,
            'average_error_pixels': avg_error,
            'max_error_pixels': max_error,
            'min_error_pixels': min_error,
            'experiment_name': self.config.get('experiment_name'),
            'movement_pattern': self.config['simulation']['movement_pattern'],
            'pattern_params': self.config['simulation']['pattern_params']
        }
        
        # Save to experiment directory
        report_path = os.path.join(self.tracking_logger.log_dir, "performance_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved to {report_path}")
        print(f"Tracking rate: {tracking_rate:.2f}")
        print(f"Average tracking error: {avg_error:.2f} pixels")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Synthetic Tracking Benchmark')
    parser.add_argument('--config', '-c', type=str,
                        help='Configuration file path')
    parser.add_argument('--duration', '-d', type=int,
                        help='Simulation duration in seconds')
    parser.add_argument('--pattern', '-p', type=str,
                        choices=['linear', 'circular', 'random', 'zigzag', 'spiral'],
                        help='Movement pattern')
    parser.add_argument('--experiment', '-e', type=str,
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return
    else:
        config = {}
    
    # Apply command line overrides
    if args.duration:
        if 'simulation' not in config:
            config['simulation'] = {}
        config['simulation']['duration'] = args.duration
    
    if args.pattern:
        if 'simulation' not in config:
            config['simulation'] = {}
        config['simulation']['movement_pattern'] = args.pattern
    
    if args.experiment:
        config['experiment_name'] = args.experiment
    
    # Create and run benchmark
    benchmark = SyntheticTrackingBenchmark(config=config)
    
    if benchmark.start():
        try:
            benchmark.run()
        except Exception as e:
            print(f"Error during benchmark: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Failed to start benchmark")


if __name__ == "__main__":
    main()