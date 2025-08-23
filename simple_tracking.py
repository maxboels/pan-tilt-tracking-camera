#!/usr/bin/env python3
"""
Simplified ZED2 Tracking System
Single-threaded version to avoid threading conflicts that cause freezing
"""

import numpy as np
import cv2
import pyzed.sl as sl
import time
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json

@dataclass
class Detection:
    """Simple detection data structure"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_2d: Tuple[int, int]
    center_3d: Optional[Tuple[float, float, float]] = None
    distance: Optional[float] = None

class SimpleTrackingSystem:
    def __init__(self, config_path: str = "config.json"):
        """Initialize simplified tracking system"""
        print("Initializing simple tracking system...")
        
        # Load config
        self.config = self.load_config(config_path)
        
        # Initialize ZED camera
        print("Setting up ZED camera...")
        self.zed = sl.Camera()
        init_params = sl.InitParameters(
            camera_resolution=sl.RESOLUTION.HD720,
            depth_mode=sl.DEPTH_MODE.PERFORMANCE,
            coordinate_units=sl.UNIT.METER,
            depth_minimum_distance=0.3,
            depth_maximum_distance=10.0
        )
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED: {err}")
        
        # ZED matrices
        self.left_image = sl.Mat()
        self.depth_map = sl.Mat()
        self.point_cloud = sl.Mat()
        self.runtime_params = sl.RuntimeParameters(confidence_threshold=70)
        
        # Initialize YOLO
        print("Loading YOLO model...")
        self.yolo = YOLO(self.config['yolo']['model_path'])
        self.target_classes = self.config['yolo']['target_classes']
        
        # Tracking state
        self.tracking_target = None
        self.last_detection_time = 0
        self.tracking_timeout = 3.0
        
        # Camera FOV for angle calculations
        self.horizontal_fov = 110.0
        self.vertical_fov = 70.0
        
        print("Simple tracking system ready!")

    def load_config(self, config_path: str) -> dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            return {
                "yolo": {
                    "model_path": "yolov8n.pt",
                    "target_classes": ["person", "car", "bicycle"]
                }
            }

    def get_3d_coordinates(self, x: int, y: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get 3D coordinates for pixel location"""
        point_cloud_value = self.point_cloud.get_value(x, y)
        
        if np.isfinite(point_cloud_value).all():
            return float(point_cloud_value[0]), float(point_cloud_value[1]), float(point_cloud_value[2])
        return None, None, None

    def detect_objects(self, image: np.ndarray) -> List[Detection]:
        """Run YOLO detection"""
        results = self.yolo(image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.yolo.names[class_id]
                
                # Filter by target classes
                if self.target_classes and class_name not in self.target_classes:
                    continue
                
                # Calculate center
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Get 3D coordinates
                x_3d, y_3d, z_3d = self.get_3d_coordinates(center_x, center_y)
                distance = z_3d if z_3d is not None else None
                
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center_2d=(center_x, center_y),
                    center_3d=(x_3d, y_3d, z_3d) if all(coord is not None for coord in [x_3d, y_3d, z_3d]) else None,
                    distance=distance
                )
                
                detections.append(detection)
        
        return detections

    def select_tracking_target(self, detections: List[Detection]) -> Optional[Detection]:
        """Select best target to track"""
        valid_detections = [d for d in detections if d.distance is not None]
        
        if not valid_detections:
            return None
        
        # Prefer people, then closest
        people = [d for d in valid_detections if d.class_name == 'person']
        candidates = people if people else valid_detections
        
        # Select closest
        return min(candidates, key=lambda d: d.distance)

    def pixel_to_angles(self, x: int, y: int, img_width: int, img_height: int) -> Tuple[float, float]:
        """Convert pixel coordinates to pan/tilt angles"""
        # Normalize to -1 to +1
        norm_x = (x - img_width / 2) / (img_width / 2)
        norm_y = (y - img_height / 2) / (img_height / 2)
        
        # Convert to angles
        pan_angle = norm_x * (self.horizontal_fov / 2)
        tilt_angle = -norm_y * (self.vertical_fov / 2)
        
        # Clamp to reasonable servo limits
        pan_angle = np.clip(pan_angle, -90, 90)
        tilt_angle = np.clip(tilt_angle, -30, 30)
        
        return pan_angle, tilt_angle

    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection overlays"""
        vis = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            center_x, center_y = detection.center_2d
            
            # Color coding
            if detection == self.tracking_target:
                color = (0, 255, 255)  # Yellow for tracking target
                thickness = 3
            elif detection.class_name == 'person':
                color = (0, 255, 0)  # Green for people
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for other objects
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            cv2.circle(vis, (center_x, center_y), 5, color, -1)
            
            # Label with distance
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if detection.distance:
                label += f" ({detection.distance:.2f}m)"
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
            cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return vis

    def run_tracking(self, duration: Optional[float] = None):
        """Main tracking loop - single threaded"""
        print("Starting tracking loop...")
        print("Controls: 'q' to quit, 'c' to center (if servos connected)")
        
        # Setup display
        cv2.namedWindow("Simple ZED Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Simple ZED Tracking", 1280, 720)
        
        start_time = time.time()
        frame_count = 0
        fps_counter = 0
        fps_start = time.time()
        current_fps = 0
        
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Grab frame
                if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.001)
                    continue
                
                # Get images
                self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZ)
                
                # Convert to numpy
                image_bgra = self.left_image.get_data()
                image_bgr = image_bgra[..., :3]
                img_height, img_width = image_bgr.shape[:2]
                
                # Run detection
                detections = self.detect_objects(image_bgr)
                
                # Update tracking target
                current_time = time.time()
                
                if detections:
                    self.tracking_target = self.select_tracking_target(detections)
                    self.last_detection_time = current_time
                    
                    if self.tracking_target:
                        # Calculate tracking angles (for display)
                        center_x, center_y = self.tracking_target.center_2d
                        pan_angle, tilt_angle = self.pixel_to_angles(center_x, center_y, img_width, img_height)
                        
                        # Here you would send pan_angle, tilt_angle to servo controller
                        # For now, just display them
                        
                else:
                    # Check timeout
                    if current_time - self.last_detection_time > self.tracking_timeout:
                        if self.tracking_target:
                            print("Tracking timeout - lost target")
                            self.tracking_target = None
                
                # Visualization
                vis = self.draw_detections(image_bgr, detections)
                
                # FPS calculation
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()
                
                # Status overlay
                status_lines = [
                    f"FPS: {current_fps:.1f}",
                    f"Detections: {len(detections)}",
                    f"Tracking: {'Yes' if self.tracking_target else 'No'}"
                ]
                
                if self.tracking_target:
                    center_x, center_y = self.tracking_target.center_2d
                    pan_angle, tilt_angle = self.pixel_to_angles(center_x, center_y, img_width, img_height)
                    status_lines.append(f"Target angles: {pan_angle:.1f}°, {tilt_angle:.1f}°")
                
                for i, line in enumerate(status_lines):
                    cv2.putText(vis, line, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Crosshair at center
                center_x, center_y = img_width // 2, img_height // 2
                cv2.drawMarker(vis, (center_x, center_y), (0, 255, 255), 
                              cv2.MARKER_CROSS, 20, 2)
                
                cv2.imshow("Simple ZED Tracking", vis)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('c'):
                    print("Center command (no servos connected)")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            # Cleanup
            self.zed.close()
            cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            print(f"Processed {frame_count} frames in {elapsed:.1f}s")
            if elapsed > 0:
                print(f"Average FPS: {frame_count/elapsed:.1f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple ZED2 Tracking System")
    parser.add_argument("--config", default="config.json", help="Config file")
    parser.add_argument("--duration", type=float, help="Run duration in seconds")
    
    args = parser.parse_args()
    
    try:
        tracker = SimpleTrackingSystem(args.config)
        tracker.run_tracking(args.duration)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()