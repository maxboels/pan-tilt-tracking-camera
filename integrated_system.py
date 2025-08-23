#!/usr/bin/env python3
"""
Integrated ZED2 + YOLO + Point Cloud + Pan-Tilt Tracking System
Main orchestrator for the complete tracking pipeline
"""

import argparse
import json
import time
from typing import Optional
import threading

from zed_yolo_tracker import ZEDYOLOTracker, Detection
from pantilt_controller import PanTiltController, PanTiltTracker
from pointcloud_processor import PointCloudProcessor

class IntegratedTrackingSystem:
    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.zed_tracker = ZEDYOLOTracker(
            model_path=self.config['yolo']['model_path'],
            zed_resolution=getattr(sl.RESOLUTION, self.config['zed']['resolution']),
            target_classes=self.config['yolo']['target_classes']
        )
        
        if self.config['pantilt']['enabled']:
            self.pantilt_controller = PanTiltController(
                pan_channel=self.config['pantilt']['pan_channel'],
                tilt_channel=self.config['pantilt']['tilt_channel'],
                smooth_movement=self.config['pantilt']['smooth_movement']
            )
            self.pantilt_tracker = PanTiltTracker(self.pantilt_controller)
        else:
            self.pantilt_controller = None
            self.pantilt_tracker = None
        
        if self.config['pointcloud']['enabled']:
            self.pointcloud_processor = PointCloudProcessor(
                max_depth=self.config['pointcloud']['max_depth'],
                voxel_size=self.config['pointcloud']['voxel_size']
            )
        else:
            self.pointcloud_processor = None
        
        # Control thread
        self.control_thread = None
        self.running = False
        
        print("Integrated tracking system initialized")

    def load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
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
                "enabled": True,
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
                "show_pointcloud": True,
                "window_size": [1280, 720]
            }
        }

    def save_config(self, config_path: str = "config.json"):
        """Save current configuration"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def start_control_thread(self):
        """Start the pan-tilt control thread"""
        if self.pantilt_tracker and not self.control_thread:
            self.running = True
            self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
            self.control_thread.start()

    def stop_control_thread(self):
        """Stop the pan-tilt control thread"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1)

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

    def process_frame_with_pointcloud(self, detections: list, point_cloud_data):
        """Enhanced processing with point cloud data"""
        if not self.pointcloud_processor or not detections:
            return detections
        
        enhanced_detections = []
        
        for detection in detections:
            # Extract point cloud for this detection
            bbox = detection.bbox
            object_points, region = self.pointcloud_processor.extract_object_pointcloud(
                point_cloud_data, bbox, 
                self.zed_tracker.left_image.get_width(),
                self.zed_tracker.left_image.get_height()
            )
            
            if region:
                # Estimate object pose
                pose = self.pointcloud_processor.estimate_object_pose(object_points)
                
                # Update detection with enhanced 3D information
                detection.center_3d = region.center
                detection.distance = region.center[2]  # Z coordinate is distance
                detection.confidence *= region.confidence  # Boost confidence with 3D data
                
                # Add pose information (could extend Detection class)
                detection.pose_info = pose
                detection.point_count = region.point_count
            
            enhanced_detections.append(detection)
        
        return enhanced_detections

    def run_tracking_system(self, duration: Optional[float] = None):
        """Run the complete tracking system"""
        print("Starting integrated tracking system...")
        print("Press 'q' to quit, 'c' to center pan-tilt, 's' to save config")
        
        # Start control thread
        if self.config['pantilt']['enabled']:
            self.start_control_thread()
        
        start_time = time.time()
        frame_count = 0
        
        try:
            # Override the ZED tracker's detection loop to add our enhancements
            zed = self.zed_tracker.zed
            runtime_params = self.zed_tracker.runtime_params
            
            import cv2
            import pyzed.sl as sl
            
            cv2.namedWindow("Integrated Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Integrated Tracking", *self.config['display']['window_size'])
            
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Grab frame
                if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.001)
                    continue
                
                # Get images and point cloud
                zed.retrieve_image(self.zed_tracker.left_image, sl.VIEW.LEFT)
                zed.retrieve_measure(self.zed_tracker.depth_map, sl.MEASURE.DEPTH)
                if self.pointcloud_processor:
                    zed.retrieve_measure(self.zed_tracker.point_cloud, sl.MEASURE.XYZ)
                
                # Convert to numpy
                image_bgra = self.zed_tracker.left_image.get_data()
                image_bgr = image_bgra[..., :3]
                
                # Run YOLO detection
                detections = self.zed_tracker.detect_objects(image_bgr)
                
                # Enhanced processing with point cloud
                if self.pointcloud_processor:
                    point_cloud_data = self.zed_tracker.point_cloud.get_data()
                    detections = self.process_frame_with_pointcloud(detections, point_cloud_data)
                
                # Update tracking target
                self.zed_tracker.current_detections = detections
                self.zed_tracker.tracking_target = self.zed_tracker.select_tracking_target(detections)
                
                # Generate pan-tilt command
                if self.zed_tracker.tracking_target:
                    pantilt_cmd = self.zed_tracker.calculate_pantilt_command(self.zed_tracker.tracking_target)
                    if pantilt_cmd and not self.zed_tracker.pantilt_queue.full():
                        self.zed_tracker.pantilt_queue.put(pantilt_cmd)
                
                # Visualization
                vis = self.zed_tracker.draw_detections(image_bgr, detections)
                
                # Add system status
                status_lines = [
                    f"Frame: {frame_count}",
                    f"Detections: {len(detections)}",
                    f"Tracking: {'Yes' if self.zed_tracker.tracking_target else 'No'}"
                ]
                
                if self.pantilt_controller:
                    pan, tilt = self.pantilt_controller.get_position()
                    status_lines.append(f"Pan/Tilt: {pan:.1f}°/{tilt:.1f}°")
                
                for i, line in enumerate(status_lines):
                    cv2.putText(vis, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                
                # Show point cloud overlay if enabled
                if self.pointcloud_processor and self.config['display']['show_pointcloud']:
                    if 'point_cloud_data' in locals():
                        # Create point cloud visualization
                        pc_flat = point_cloud_data.reshape(-1, 3)
                        pc_vis = self.pointcloud_processor.visualize_point_cloud_2d(
                            pc_flat, vis.shape[:2])
                        
                        # Blend with main image
                        alpha = 0.3
                        vis = cv2.addWeighted(vis, 1-alpha, pc_vis, alpha, 0)
                
                cv2.imshow("Integrated Tracking", vis)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('c') and self.pantilt_controller:  # center
                    self.pantilt_controller.center()
                    print("Pan-tilt centered")
                elif key == ord('s'):  # save config
                    self.save_config()
                    print("Configuration saved")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nStopping system...")
        
        finally:
            # Cleanup
            self.stop_control_thread()
            
            if self.pantilt_controller:
                self.pantilt_controller.shutdown()
            
            zed.close()
            cv2.destroyAllWindows()
            
            # Print statistics
            elapsed = time.time() - start_time
            print(f"Processed {frame_count} frames in {elapsed:.1f}s")
            print(f"Average FPS: {frame_count/elapsed:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Integrated ZED2 Tracking System")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--duration", type=float, help="Run duration in seconds")
    parser.add_argument("--generate-config", action="store_true", help="Generate default config file")
    
    args = parser.parse_args()
    
    if args.generate_config:
        system = IntegratedTrackingSystem()
        system.save_config(args.config)
        print(f"Default configuration saved to {args.config}")
        return
    
    # Run the system
    system = IntegratedTrackingSystem(args.config)
    system.run_tracking_system(args.duration)


if __name__ == "__main__":
    main()