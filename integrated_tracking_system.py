#!/usr/bin/env python3
"""
Main ZED2 Tracking System
Integrates all components: ZED camera, YOLO detection, point cloud processing, and pan-tilt control
"""

import argparse
import json
import time
import cv2
import numpy as np
from typing import Optional
import threading
import sys

# Import our modules
try:
    from zed_yolo_tracker import ZEDYOLOTracker
    from pantilt_controller import PanTiltController, PanTiltTracker
    from pointcloud_processor import PointCloudProcessor
    print("All modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are in the same directory")
    sys.exit(1)

class MainTrackingSystem:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the complete tracking system"""
        # Load configuration
        self.config = self.load_config(config_path)
        print(f"Configuration loaded from {config_path}")
        
        # Initialize ZED + YOLO tracker
        print("Initializing ZED camera and YOLO detector...")
        self.zed_tracker = ZEDYOLOTracker(
            model_path=self.config['yolo']['model_path'],
            target_classes=self.config['yolo']['target_classes']
        )
        
        # Initialize pan-tilt controller (if enabled)
        self.pantilt_controller = None
        self.pantilt_tracker = None
        if self.config['pantilt']['enabled']:
            try:
                print("Initializing pan-tilt controller...")
                self.pantilt_controller = PanTiltController(
                    pan_channel=self.config['pantilt']['pan_channel'],
                    tilt_channel=self.config['pantilt']['tilt_channel'],
                    smooth_movement=self.config['pantilt']['smooth_movement']
                )
                self.pantilt_tracker = PanTiltTracker(self.pantilt_controller)
                print("Pan-tilt system initialized")
            except Exception as e:
                print(f"Pan-tilt initialization failed: {e}")
                print("Running in camera-only mode")
                self.config['pantilt']['enabled'] = False
        
        # Initialize point cloud processor (if enabled)
        self.pointcloud_processor = None
        if self.config['pointcloud']['enabled']:
            try:
                print("Initializing point cloud processor...")
                self.pointcloud_processor = PointCloudProcessor(
                    max_depth=self.config['pointcloud']['max_depth'],
                    voxel_size=self.config['pointcloud']['voxel_size']
                )
                print("Point cloud processor initialized")
            except Exception as e:
                print(f"Point cloud processor initialization failed: {e}")
                print("Running without point cloud processing")
                self.config['pointcloud']['enabled'] = False
        
        # Control thread for pan-tilt
        self.control_thread = None
        self.running = False
        
        print("Main tracking system initialized successfully")

    def load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()
        except json.JSONDecodeError as e:
            print(f"Config file {config_path} has invalid JSON: {e}")
            return self.get_default_config()

    def get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            "zed": {
                "resolution": "HD720",
                "depth_mode": "PERFORMANCE"
            },
            "yolo": {
                "model_path": "yolov8n.pt",
                "target_classes": ["person", "car", "bicycle"]
            },
            "pantilt": {
                "enabled": False,  # Default to False for laptop
                "pan_channel": 0,
                "tilt_channel": 1,
                "smooth_movement": True,
                "tracking_enabled": True
            },
            "pointcloud": {
                "enabled": True,
                "max_depth": 10.0,
                "voxel_size": 0.05,
                "save_clouds": False
            },
            "display": {
                "show_detections": True,
                "show_pointcloud": False,  # Can be heavy
                "window_size": [1280, 720]
            }
        }

    def save_config(self, config_path: str = "config.json"):
        """Save current configuration"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to {config_path}")

    def start_control_thread(self):
        """Start the pan-tilt control thread"""
        if self.pantilt_tracker and not self.control_thread:
            self.running = True
            self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
            self.control_thread.start()
            print("Pan-tilt control thread started")

    def stop_control_thread(self):
        """Stop the pan-tilt control thread"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1)
            print("Pan-tilt control thread stopped")

    def control_loop(self):
        """Pan-tilt control loop running in separate thread"""
        while self.running:
            try:
                # Get latest command from tracker
                cmd = self.zed_tracker.get_latest_pantilt_command()
                
                if cmd and self.pantilt_tracker:
                    # Execute tracking command
                    self.pantilt_tracker.track_target(cmd.pan_angle, cmd.tilt_angle)
                else:
                    # Check for timeout
                    if self.pantilt_tracker:
                        self.pantilt_tracker.check_timeout()
                
                time.sleep(0.05)  # 20Hz control rate
                
            except Exception as e:
                print(f"Control loop error: {e}")
                time.sleep(0.1)

    def run_tracking_system(self, duration: Optional[float] = None):
        """Run the complete tracking system"""
        print("Starting complete tracking system...")
        print("Controls:")
        print("  'q' or ESC - Quit")
        print("  'c' - Center pan-tilt (if enabled)")
        print("  's' - Save current configuration")
        if self.config['pantilt']['enabled']:
            print(f"  Pan-tilt enabled on channels {self.config['pantilt']['pan_channel']}, {self.config['pantilt']['tilt_channel']}")
        else:
            print("  Pan-tilt disabled (camera tracking only)")
        
        # Start control thread if pan-tilt enabled
        if self.config['pantilt']['enabled']:
            self.start_control_thread()
        
        # Setup display
        if self.config['display']['show_detections']:
            window_name = "ZED2 Tracking System"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, *self.config['display']['window_size'])
        
        start_time = time.time()
        frame_count = 0
        
        try:
            # Run the ZED detection loop with our enhancements
            self.run_enhanced_detection_loop(duration, start_time)
            
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        
        finally:
            # Cleanup
            print("Cleaning up...")
            self.stop_control_thread()
            
            if self.pantilt_controller:
                self.pantilt_controller.center()
                time.sleep(1)
                self.pantilt_controller.shutdown()
            
            # ZED cleanup happens in the tracker
            if self.config['display']['show_detections']:
                cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            print(f"Session complete - processed {frame_count} frames in {elapsed:.1f}s")
            if elapsed > 0:
                print(f"Average FPS: {frame_count/elapsed:.1f}")

    def run_enhanced_detection_loop(self, duration: Optional[float], start_time: float):
        """Enhanced detection loop with all integrations"""
        frame_count = 0
        fps = 0
        fps_start = time.time()
        
        # Access ZED camera directly
        zed = self.zed_tracker.zed
        runtime_params = self.zed_tracker.runtime_params
        
        print("Starting detection loop...")
        print("Waiting for first frame...")
        
        while True:
            # Check duration limit
            if duration and (time.time() - start_time) > duration:
                print("Duration limit reached, stopping...")
                break
            
            # Grab frame
            grab_result = zed.grab(runtime_params)
            if grab_result != 0:  # sl.ERROR_CODE.SUCCESS == 0
                if frame_count == 0:
                    print(f"Waiting for first frame... (grab result: {grab_result})")
                time.sleep(0.001)
                continue
            
            if frame_count == 0:
                print("First frame captured successfully!")
            
            # Get images and point cloud
            zed.retrieve_image(self.zed_tracker.left_image, 0)  # sl.VIEW.LEFT == 0
            zed.retrieve_measure(self.zed_tracker.depth_map, 1)  # sl.MEASURE.DEPTH == 1
            if self.pointcloud_processor:
                zed.retrieve_measure(self.zed_tracker.point_cloud, 4)  # sl.MEASURE.XYZ == 4
            
            # Convert to numpy
            image_bgra = self.zed_tracker.left_image.get_data()
            image_bgr = image_bgra[..., :3]
            
            if frame_count == 0:
                print(f"Image shape: {image_bgr.shape}")
                print("Running first YOLO detection...")
            
            # Run YOLO detection
            detections = self.zed_tracker.detect_objects(image_bgr)
            
            if frame_count == 0:
                print(f"First detection complete, found {len(detections)} objects")
            
            # Enhanced processing with point cloud
            if self.pointcloud_processor and detections:
                point_cloud_data = self.zed_tracker.point_cloud.get_data()
                detections = self.process_frame_with_pointcloud(detections, point_cloud_data)
            
            # Update tracking target
            self.zed_tracker.current_detections = detections
            self.zed_tracker.tracking_target = self.zed_tracker.select_tracking_target(detections)
            
            # Generate pan-tilt command
            if self.zed_tracker.tracking_target and self.config['pantilt']['enabled']:
                pantilt_cmd = self.zed_tracker.calculate_pantilt_command(self.zed_tracker.tracking_target)
                if pantilt_cmd and not self.zed_tracker.pantilt_queue.full():
                    self.zed_tracker.pantilt_queue.put(pantilt_cmd)
            
            # Display
            if self.config['display']['show_detections']:
                vis = self.zed_tracker.draw_detections(image_bgr, detections)
                
                # Add system status
                frame_count += 1
                if time.time() - fps_start >= 1.0:
                    fps = frame_count / (time.time() - fps_start)
                    fps_start = time.time()
                    frame_count = 0
                
                status_lines = [
                    f"FPS: {fps:.1f}",
                    f"Detections: {len(detections)}",
                    f"Tracking: {'Yes' if self.zed_tracker.tracking_target else 'No'}"
                ]
                
                if self.pantilt_controller:
                    pan, tilt = self.pantilt_controller.get_position()
                    status_lines.append(f"Pan/Tilt: {pan:.1f}°/{tilt:.1f}°")
                
                # Draw status
                for i, line in enumerate(status_lines):
                    cv2.putText(vis, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                
                # Draw crosshair at center
                h, w = vis.shape[:2]
                center_x, center_y = w // 2, h // 2
                cv2.drawMarker(vis, (center_x, center_y), (0, 255, 255), 
                              cv2.MARKER_CROSS, 20, 2)
                
                cv2.imshow("ZED2 Tracking System", vis)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('c') and self.pantilt_controller:  # center
                    self.pantilt_controller.center()
                    print("Pan-tilt centered")
                elif key == ord('s'):  # save config
                    self.save_config()

    def process_frame_with_pointcloud(self, detections, point_cloud_data):
        """Enhanced processing with point cloud data"""
        if not self.pointcloud_processor or not detections:
            return detections
        
        enhanced_detections = []
        
        for detection in detections:
            # Extract point cloud for this detection
            bbox = detection.bbox
            img_width = self.zed_tracker.left_image.get_width()
            img_height = self.zed_tracker.left_image.get_height()
            
            try:
                object_points, region = self.pointcloud_processor.extract_object_pointcloud(
                    point_cloud_data, bbox, img_width, img_height
                )
                
                if region:
                    # Update detection with enhanced 3D information
                    detection.center_3d = region.center
                    detection.distance = region.center[2]  # Z coordinate is distance
                    detection.confidence *= region.confidence  # Boost confidence with 3D data
            except Exception as e:
                print(f"Point cloud processing error: {e}")
            
            enhanced_detections.append(detection)
        
        return enhanced_detections


def main():
    parser = argparse.ArgumentParser(description="ZED2 Complete Tracking System")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--duration", type=float, help="Run duration in seconds")
    parser.add_argument("--generate-config", action="store_true", help="Generate default config file")
    parser.add_argument("--no-pantilt", action="store_true", help="Disable pan-tilt control")
    parser.add_argument("--no-pointcloud", action="store_true", help="Disable point cloud processing")
    
    args = parser.parse_args()
    
    if args.generate_config:
        system = MainTrackingSystem()
        system.save_config(args.config)
        print(f"Default configuration saved to {args.config}")
        return
    
    # Load and modify config based on arguments
    system = MainTrackingSystem(args.config)
    
    if args.no_pantilt:
        system.config['pantilt']['enabled'] = False
        print("Pan-tilt control disabled by command line argument")
    
    if args.no_pointcloud:
        system.config['pointcloud']['enabled'] = False
        print("Point cloud processing disabled by command line argument")
    
    # Run the system
    print("Starting ZED2 tracking system...")
    system.run_tracking_system(args.duration)


if __name__ == "__main__":
    main()