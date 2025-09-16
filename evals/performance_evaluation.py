#!/usr/bin/env python3
"""
Performance Evaluation Tool for Pan-Tilt Tracking Camera
Analyzes system performance, bottlenecks, and failure cases
"""

import sys
import os
import time
import cv2
import numpy as np
import threading
import argparse
import json
import matplotlib.pyplot as plt
from collections import deque

# Add parent directory to path to import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.usb_camera import USBCamera
from src.servo_controller import ArduinoServoController
from src.calibration import CameraServoCalibrator
from src.yolo_tracker import YOLOTracker

class PerformanceMonitor:
    """Performance monitoring and analysis for the tracking system"""
    
    def __init__(self, log_file=None):
        """Initialize performance monitor"""
        self.log_file = log_file
        self.metrics = {
            'camera_fps': deque(maxlen=100),
            'yolo_inference_time': deque(maxlen=100),
            'tracking_time': deque(maxlen=100),
            'servo_command_time': deque(maxlen=100),
            'overall_fps': deque(maxlen=100),
            'tracking_success': deque(maxlen=100),  # 1 for success, 0 for failure
            'num_detections': deque(maxlen=100),
        }
        
        # Timestamps for computing durations
        self.timestamps = {}
        
        # Log file setup
        if log_file:
            with open(log_file, 'w') as f:
                f.write("timestamp,camera_fps,yolo_time,tracking_time,servo_time,overall_fps,success,detections\n")
    
    def start_timer(self, name):
        """Start timing a specific operation"""
        self.timestamps[name] = time.time()
    
    def stop_timer(self, name):
        """Stop timing an operation and record the duration"""
        if name in self.timestamps:
            duration = time.time() - self.timestamps[name]
            if name + '_time' in self.metrics:
                self.metrics[name + '_time'].append(duration)
            return duration
        return None
    
    def record_metric(self, name, value):
        """Record a specific performance metric"""
        if name in self.metrics:
            self.metrics[name].append(value)
    
    def log_frame_metrics(self, frame_number):
        """Log metrics for the current frame to file"""
        if not self.log_file:
            return
        
        # Get latest metrics
        try:
            camera_fps = self.metrics['camera_fps'][-1] if self.metrics['camera_fps'] else 0
            yolo_time = self.metrics['yolo_inference_time'][-1] if self.metrics['yolo_inference_time'] else 0
            tracking_time = self.metrics['tracking_time'][-1] if self.metrics['tracking_time'] else 0
            servo_time = self.metrics['servo_command_time'][-1] if self.metrics['servo_command_time'] else 0
            overall_fps = self.metrics['overall_fps'][-1] if self.metrics['overall_fps'] else 0
            success = self.metrics['tracking_success'][-1] if self.metrics['tracking_success'] else 0
            detections = self.metrics['num_detections'][-1] if self.metrics['num_detections'] else 0
            
            with open(self.log_file, 'a') as f:
                f.write(f"{time.time()},{camera_fps:.2f},{yolo_time:.4f},{tracking_time:.4f},"
                        f"{servo_time:.4f},{overall_fps:.2f},{success},{detections}\n")
        except Exception as e:
            print(f"Error logging metrics: {e}")
    
    def get_average_metrics(self):
        """Get average values for all metrics"""
        result = {}
        for key, values in self.metrics.items():
            if values:
                result[key] = sum(values) / len(values)
            else:
                result[key] = 0
        return result
    
    def plot_performance(self, save_path=None):
        """Plot performance metrics"""
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot 1: FPS over time
        if self.metrics['overall_fps']:
            axs[0].plot(list(self.metrics['overall_fps']), label='Overall FPS')
            axs[0].plot(list(self.metrics['camera_fps']), label='Camera FPS')
            axs[0].set_title('FPS Performance')
            axs[0].set_xlabel('Frame Number')
            axs[0].set_ylabel('Frames Per Second')
            axs[0].legend()
            axs[0].grid(True)
        
        # Plot 2: Processing time breakdown
        if self.metrics['yolo_inference_time'] and self.metrics['tracking_time'] and self.metrics['servo_command_time']:
            axs[1].plot(list(self.metrics['yolo_inference_time']), label='YOLO Inference')
            axs[1].plot(list(self.metrics['tracking_time']), label='Tracking Algorithm')
            axs[1].plot(list(self.metrics['servo_command_time']), label='Servo Command')
            axs[1].set_title('Processing Time Breakdown')
            axs[1].set_xlabel('Frame Number')
            axs[1].set_ylabel('Time (seconds)')
            axs[1].legend()
            axs[1].grid(True)
        
        # Plot 3: Tracking success and detections
        if self.metrics['tracking_success'] and self.metrics['num_detections']:
            axs[2].plot(list(self.metrics['tracking_success']), label='Tracking Success')
            axs[2].plot(list(self.metrics['num_detections']), label='Number of Detections')
            axs[2].set_title('Tracking Performance')
            axs[2].set_xlabel('Frame Number')
            axs[2].set_ylabel('Count')
            axs[2].legend()
            axs[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Performance plot saved to {save_path}")
        else:
            plt.show()


class PerformanceEvaluator:
    """Evaluate performance of the pan-tilt tracking system"""
    
    def __init__(self, config=None, resolution=None, log_file=None):
        """Initialize the performance evaluator"""
        # Load configuration
        if config is None:
            self.config = self.load_config()
        else:
            self.config = config
        
        # Override resolution if specified
        if resolution:
            self.config['camera']['resolution'] = resolution
        
        # Initialize components
        camera_config = self.config.get('camera', {})
        self.camera = USBCamera(
            camera_index=camera_config.get('index', 0),
            resolution=tuple(camera_config.get('resolution', [1920, 1080])),
            fps=camera_config.get('fps', 30)
        )
        
        servo_config = self.config.get('servo', {})
        self.servo_controller = ArduinoServoController(
            port=servo_config.get('port', '/dev/ttyUSB0'),
            baudrate=servo_config.get('baudrate', 115200),
            inverted_pan=servo_config.get('inverted_pan', True)
        )
        
        tracking_config = self.config.get('tracking', {})
        self.yolo_tracker = YOLOTracker(
            model_path=tracking_config.get('model_path', 'yolov8n.pt'),
            confidence_threshold=tracking_config.get('confidence_threshold', 0.5)
        )
        
        # Set up calibrator
        self.calibrator = CameraServoCalibrator("../config/calibration.json")
        self.calibrator.load_calibration()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(log_file)
        
        # Control parameters
        self.frame_center = (
            camera_config.get('resolution', [1920, 1080])[0] // 2,
            camera_config.get('resolution', [1920, 1080])[1] // 2
        )
        self.calibrator.frame_center = self.frame_center
        self.dead_zone = tracking_config.get('dead_zone', 50)
        
        # State variables
        self.running = False
        self.target_position = None
        self.total_frames = 0
        
        # Debug information
        self.debug_info = {}
    
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
                "camera": {"resolution": [1920, 1080], "fps": 30, "index": 0},
                "servo": {"port": "/dev/ttyUSB0", "baudrate": 115200, "inverted_pan": True},
                "tracking": {"model_path": "yolov8n.pt", "confidence_threshold": 0.5, "dead_zone": 50}
            }
    
    def start(self):
        """Start the evaluation system"""
        print("Starting performance evaluation...")
        
        # Try multiple camera indices if needed
        camera_indices = [0, 1, 2]  # Try camera indices 0, 1, and 2
        camera_resolutions = [
            (1920, 1080),
            (1280, 720),
            (640, 480)
        ]
        
        camera_opened = False
        
        # First try with configured settings
        if self.camera.open():
            if self.camera.start_capture():
                camera_opened = True
                # Verify camera is working by reading a test frame
                for i in range(10):  # Try multiple times
                    test_frame = self.camera.get_frame()
                    if test_frame is not None:
                        print(f"Camera test successful. Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                        break
                    print(f"Camera not returning frames, attempt {i+1}/10...")
                    time.sleep(0.5)
                else:
                    print("Warning: Camera opened but not returning frames. Trying direct OpenCV capture...")
                    camera_opened = False
                    self.camera.close()
        
        # If configured camera didn't work, try direct OpenCV capture
        if not camera_opened:
            print("Trying direct OpenCV camera capture...")
            for idx in camera_indices:
                for width, height in camera_resolutions:
                    print(f"Trying camera index {idx} with resolution {width}x{height}...")
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"Direct OpenCV capture successful with index {idx}, resolution {width}x{height}")
                            
                            # Update our camera with working settings
                            self.camera.close()
                            self.camera = USBCamera(
                                camera_index=idx,
                                resolution=(width, height),
                                fps=30
                            )
                            
                            if self.camera.open() and self.camera.start_capture():
                                print(f"USBCamera configured with working settings.")
                                camera_opened = True
                                break
                        cap.release()
                if camera_opened:
                    break
        
        if not camera_opened:
            print("Error: Could not open camera with any settings. Please check camera connection.")
            return False
            
        # Verify camera is now returning frames
        test_frame = self.camera.get_frame()
        if test_frame is None:
            print("Error: Camera opened but still not returning frames.")
            self.camera.close()
            return False
            
        # Connect servo controller (optional for evaluation)
        servo_connected = self.servo_controller.connect()
        if not servo_connected:
            print("Warning: Servo controller not connected (evaluation will run without servo control)")
        else:
            self.servo_controller.center_servos()
        
        self.running = True
        return True
    
    def stop(self):
        """Stop the evaluation system"""
        print("Stopping evaluation...")
        self.running = False
        
        if self.servo_controller.connected:
            self.servo_controller.center_servos()
            self.servo_controller.disconnect()
        
        self.camera.close()
        cv2.destroyAllWindows()
    
    def process_frame(self, frame, enable_tracking=True, enable_servos=True):
        """Process a frame with performance monitoring"""
        # Record overall processing start time
        self.performance_monitor.start_timer('overall')
        
        # Camera FPS
        camera_fps = self.camera.get_fps()
        self.performance_monitor.record_metric('camera_fps', camera_fps)
        
        # Start YOLO inference timing
        self.performance_monitor.start_timer('yolo_inference')
        detections = self.yolo_tracker.detect_objects(frame)
        yolo_time = self.performance_monitor.stop_timer('yolo_inference')
        
        # Number of detections
        num_detections = len(detections) if detections else 0
        self.performance_monitor.record_metric('num_detections', num_detections)
        
        # Start tracking algorithm timing
        self.performance_monitor.start_timer('tracking')
        if enable_tracking:
            current_target = self.yolo_tracker.update_tracking(detections)
            smoothed_pos = self.yolo_tracker.get_smoothed_target_position() if current_target else None
            
            if smoothed_pos:
                # Check dead zone
                pan_error = smoothed_pos[0] - self.frame_center[0]
                tilt_error = smoothed_pos[1] - self.frame_center[1]
                
                if abs(pan_error) > self.dead_zone or abs(tilt_error) > self.dead_zone:
                    self.target_position = smoothed_pos
                    tracking_success = 1
                else:
                    self.target_position = None
                    tracking_success = 1  # Still successful, just in dead zone
            else:
                self.target_position = None
                tracking_success = 0  # No target found
        else:
            self.target_position = None
            tracking_success = 0
            
        self.performance_monitor.record_metric('tracking_success', tracking_success)
        tracking_time = self.performance_monitor.stop_timer('tracking')
        
        # Servo movement (if enabled and connected)
        self.performance_monitor.start_timer('servo_command')
        if enable_servos and self.servo_controller.connected and self.target_position:
            try:
                # Convert pixel coordinates to servo angles
                target_pan, target_tilt = self.calibrator.pixel_to_servo(
                    self.target_position[0], self.target_position[1]
                )
                
                # Apply servo limits
                target_pan = np.clip(target_pan, -90, 90)
                target_tilt = np.clip(target_tilt, -45, 45)
                
                # Send servo command
                self.servo_controller.move_servos(target_pan, target_tilt)
                
            except Exception as e:
                print(f"Servo control error: {e}")
        servo_time = self.performance_monitor.stop_timer('servo_command')
        
        # Draw visualization
        vis_frame = frame.copy()
        if detections:
            vis_frame = self.yolo_tracker.draw_detections(vis_frame, detections)
        
        # Draw tracking information
        if self.target_position:
            # Draw target position
            cv2.circle(vis_frame, (int(self.target_position[0]), int(self.target_position[1])), 
                      10, (0, 255, 255), 2)
            
            # Show target info
            cv2.putText(vis_frame, f"Target: ({int(self.target_position[0])}, {int(self.target_position[1])})",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw center crosshair and dead zone
        cv2.circle(vis_frame, self.frame_center, 5, (255, 0, 0), -1)
        cv2.circle(vis_frame, self.frame_center, self.dead_zone, (255, 0, 0), 2)
        
        # Display performance metrics on frame
        self.draw_performance_overlay(vis_frame, {
            'YOLO Inference': f"{yolo_time*1000:.1f} ms",
            'Tracking': f"{tracking_time*1000:.1f} ms",
            'Servo Command': f"{servo_time*1000:.1f} ms",
            'Camera FPS': f"{camera_fps:.1f}",
            'Detections': f"{num_detections}"
        })
        
        # Calculate overall FPS
        overall_time = self.performance_monitor.stop_timer('overall')
        if overall_time > 0:
            overall_fps = 1.0 / overall_time
        else:
            overall_fps = 0
        self.performance_monitor.record_metric('overall_fps', overall_fps)
        
        # Log metrics for this frame
        self.performance_monitor.log_frame_metrics(self.total_frames)
        self.total_frames += 1
        
        return vis_frame
    
    def draw_performance_overlay(self, frame, metrics_dict):
        """Draw performance metrics overlay on the frame"""
        y = 30
        for name, value in metrics_dict.items():
            cv2.putText(frame, f"{name}: {value}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
    
    def run_evaluation(self, duration=60, enable_tracking=True, enable_servos=True, 
                      show_video=True, output_video=None):
        """Run evaluation for specified duration"""
        if not self.start():
            print("Failed to start evaluation due to camera issues.")
            return False
        
        print(f"Running performance evaluation for {duration} seconds...")
        print(f"Tracking enabled: {enable_tracking}")
        print(f"Servo control enabled: {enable_servos}")
        
        # Setup video writer if needed
        video_writer = None
        if output_video:
            frame = self.camera.get_frame()
            if frame is not None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(output_video, fourcc, 30.0, (w, h))
        
        # Setup display window
        if show_video:
            cv2.namedWindow("Performance Evaluation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Performance Evaluation", 1280, 720)
        
        start_time = time.time()
        try:
            while self.running and (time.time() - start_time) < duration:
                # Get frame
                frame = self.camera.get_frame()
                if frame is None:
                    print("Warning: Camera returned None frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                vis_frame = self.process_frame(frame, enable_tracking, enable_servos)
                
                # Write to output video
                if video_writer:
                    video_writer.write(vis_frame)
                
                # Display frame
                if show_video:
                    cv2.imshow("Performance Evaluation", vis_frame)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    # Print progress
                    elapsed = time.time() - start_time
                    if int(elapsed) % 5 == 0 and int(elapsed) > 0:  # Print every 5 seconds
                        print(f"Evaluation progress: {elapsed:.1f}/{duration} seconds - {self.total_frames} frames processed", end='\r')
                        sys.stdout.flush()
        
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            if video_writer:
                video_writer.write(vis_frame)
            
            self.stop()
            
            # Print summary
            self.print_performance_summary()
            
            return True
    
    def print_performance_summary(self):
        """Print performance evaluation summary"""
        avg_metrics = self.performance_monitor.get_average_metrics()
        
        print("\n=== Performance Evaluation Summary ===")
        print(f"Total frames processed: {self.total_frames}")
        print(f"Average camera FPS: {avg_metrics['camera_fps']:.2f}")
        print(f"Average overall FPS: {avg_metrics['overall_fps']:.2f}")
        print("\nAverage processing times:")
        print(f"  YOLO inference: {avg_metrics['yolo_inference_time']*1000:.2f} ms")
        print(f"  Tracking algorithm: {avg_metrics['tracking_time']*1000:.2f} ms")
        print(f"  Servo command: {avg_metrics['servo_command_time']*1000:.2f} ms")
        
        print("\nTracking performance:")
        print(f"  Tracking success rate: {avg_metrics['tracking_success']*100:.2f}%")
        print(f"  Average detections per frame: {avg_metrics['num_detections']:.2f}")
        
        # Calculate processing time distribution
        total_time = (avg_metrics['yolo_inference_time'] + 
                      avg_metrics['tracking_time'] + 
                      avg_metrics['servo_command_time'])
        
        if total_time > 0:
            print("\nProcessing time distribution:")
            print(f"  YOLO inference: {avg_metrics['yolo_inference_time']/total_time*100:.2f}%")
            print(f"  Tracking algorithm: {avg_metrics['tracking_time']/total_time*100:.2f}%")
            print(f"  Servo command: {avg_metrics['servo_command_time']/total_time*100:.2f}%")
        
        # Identify bottlenecks
        print("\nBottleneck analysis:")
        bottleneck = max(
            ('YOLO inference', avg_metrics['yolo_inference_time']),
            ('Tracking algorithm', avg_metrics['tracking_time']),
            ('Servo command', avg_metrics['servo_command_time']),
            key=lambda x: x[1]
        )
        
        print(f"  Primary bottleneck: {bottleneck[0]} ({bottleneck[1]*1000:.2f} ms)")
        
        if avg_metrics['overall_fps'] < 15:
            print("  ⚠️ System running below real-time performance (< 15 FPS)")
            print("  Consider using a smaller YOLO model or lower resolution")
        
        if avg_metrics['tracking_success'] < 0.7:
            print("  ⚠️ Low tracking success rate (< 70%)")
            print("  Check lighting conditions and camera positioning")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Pan-Tilt Tracking Performance Evaluation')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Evaluation duration in seconds')
    parser.add_argument('--resolution', '-r', type=str, default=None,
                        help='Camera resolution (WxH, e.g. 1280x720)')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable tracking (YOLO inference only)')
    parser.add_argument('--no-servos', action='store_true',
                        help='Disable servo control')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video display')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output video file path')
    parser.add_argument('--log', '-l', type=str, default=None,
                        help='Performance log file path')
    parser.add_argument('--plot', '-p', type=str, default=None,
                        help='Save performance plot to file')
    
    args = parser.parse_args()
    
    # Parse resolution if provided
    resolution = None
    if args.resolution:
        try:
            w, h = map(int, args.resolution.split('x'))
            resolution = [w, h]
        except:
            print(f"Invalid resolution format: {args.resolution}. Using default.")
    
    # Create evaluator
    evaluator = PerformanceEvaluator(resolution=resolution, log_file=args.log)
    
    # Run evaluation
    success = evaluator.run_evaluation(
        duration=args.duration,
        enable_tracking=not args.no_tracking,
        enable_servos=not args.no_servos,
        show_video=not args.no_video,
        output_video=args.output
    )
    
    # Generate performance plot if requested
    if success and args.plot:
        evaluator.performance_monitor.plot_performance(args.plot)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
