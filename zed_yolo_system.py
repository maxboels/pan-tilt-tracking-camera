#!/usr/bin/env python3
"""
ZED2 + YOLO Object Detection and Tracking System
Integrates stereo vision, object detection, point clouds, and pan-tilt control
"""

import numpy as np
import cv2
import pyzed.sl as sl
import time
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
import queue

@dataclass
class Detection:
    """Object detection with 3D coordinates"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_2d: Tuple[int, int]
    center_3d: Optional[Tuple[float, float, float]] = None  # x, y, z in meters
    distance: Optional[float] = None

@dataclass  
class PanTiltCommand:
    """Pan-tilt servo command"""
    pan_angle: float  # degrees (-90 to +90)
    tilt_angle: float  # degrees (-30 to +30)
    timestamp: float

class ZEDYOLOTracker:
    def __init__(self, 
                 model_path: str = 'yolov8n.pt',
                 zed_resolution: sl.RESOLUTION = sl.RESOLUTION.HD720,
                 depth_mode: sl.DEPTH_MODE = sl.DEPTH_MODE.PERFORMANCE,
                 target_classes: List[str] = None):
        
        # Initialize ZED camera
        self.zed = sl.Camera()
        init_params = sl.InitParameters(
            camera_resolution=zed_resolution,
            depth_mode=depth_mode,
            coordinate_units=sl.UNIT.METER,
            depth_minimum_distance=0.3,
            depth_maximum_distance=10.0
        )
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")
        
        # Get camera info
        self.camera_info = self.zed.get_camera_information()
        self.calibration = self.camera_info.camera_configuration.calibration_parameters.left_cam
        
        # Initialize YOLO
        self.yolo = YOLO(model_path)
        self.target_classes = target_classes or ['person']  # Default to tracking people
        
        # ZED matrices
        self.left_image = sl.Mat()
        self.depth_map = sl.Mat()
        self.point_cloud = sl.Mat()
        
        # Runtime parameters
        self.runtime_params = sl.RuntimeParameters(confidence_threshold=70)
        
        # Tracking state
        self.current_detections: List[Detection] = []
        self.tracking_target: Optional[Detection] = None
        
        # Pan-tilt control queue
        self.pantilt_queue = queue.Queue(maxsize=10)
        
        print(f"ZED camera initialized: {self.camera_info.camera_model}")
        print(f"YOLO model loaded: {model_path}")
        print(f"Target classes: {self.target_classes}")

    def get_3d_coordinates(self, x: int, y: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get 3D world coordinates for a 2D pixel location"""
        point_cloud_value = self.point_cloud.get_value(x, y)
        
        if np.isfinite(point_cloud_value).all():
            return float(point_cloud_value[0]), float(point_cloud_value[1]), float(point_cloud_value[2])
        return None, None, None

    def pixel_to_camera_angles(self, x: int, y: int, image_width: int, image_height: int) -> Tuple[float, float]:
        """Convert pixel coordinates to camera pan/tilt angles"""
        # Camera field of view (approximate for ZED2)
        horizontal_fov = 110.0  # degrees
        vertical_fov = 70.0     # degrees
        
        # Convert pixel to normalized coordinates (-1 to +1)
        norm_x = (x - image_width / 2) / (image_width / 2)
        norm_y = (y - image_height / 2) / (image_height / 2)
        
        # Convert to angles
        pan_angle = norm_x * (horizontal_fov / 2)
        tilt_angle = -norm_y * (vertical_fov / 2)  # Negative because image Y is inverted
        
        return pan_angle, tilt_angle

    def detect_objects(self, image: np.ndarray) -> List[Detection]:
        """Run YOLO detection on image"""
        results = self.yolo(image, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Extract detection data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.yolo.names[class_id]
                
                # Filter by target classes
                if self.target_classes and class_name not in self.target_classes:
                    continue
                
                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Get 3D coordinates
                x_3d, y_3d, z_3d = self.get_3d_coordinates(center_x, center_y)
                distance = None
                if z_3d is not None:
                    distance = float(z_3d)
                
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
        """Select the best target to track (closest person with good 3D data)"""
        valid_detections = [d for d in detections if d.center_3d is not None and d.distance is not None]
        
        if not valid_detections:
            return None
        
        # Priority: person class, then closest
        people = [d for d in valid_detections if d.class_name == 'person']
        targets = people if people else valid_detections
        
        # Select closest target
        return min(targets, key=lambda d: d.distance)

    def calculate_pantilt_command(self, target: Detection) -> Optional[PanTiltCommand]:
        """Calculate pan-tilt angles to center on target"""
        if not target.center_2d:
            return None
        
        center_x, center_y = target.center_2d
        image_width = self.left_image.get_width()
        image_height = self.left_image.get_height()
        
        pan_angle, tilt_angle = self.pixel_to_camera_angles(center_x, center_y, image_width, image_height)
        
        # Clamp angles to servo limits
        pan_angle = np.clip(pan_angle, -90, 90)
        tilt_angle = np.clip(tilt_angle, -30, 30)
        
        return PanTiltCommand(
            pan_angle=pan_angle,
            tilt_angle=tilt_angle,
            timestamp=time.time()
        )

    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and 3D info on image"""
        vis = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Choose color based on class
            if detection.class_name == 'person':
                color = (0, 255, 0)  # Green for people
            else:
                color = (255, 0, 0)  # Blue for other objects
            
            # Highlight tracking target
            if self.tracking_target and detection == self.tracking_target:
                color = (0, 255, 255)  # Yellow for tracking target
                thickness = 3
            else:
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with confidence and distance
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if detection.distance is not None:
                label += f" ({detection.distance:.2f}m)"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center point and 3D coordinates
            center_x, center_y = detection.center_2d
            cv2.circle(vis, (center_x, center_y), 5, color, -1)
            
            if detection.center_3d:
                x_3d, y_3d, z_3d = detection.center_3d
                coord_text = f"({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})"
                cv2.putText(vis, coord_text, (center_x + 10, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis

    def run_detection_loop(self, display: bool = True, save_pantilt_commands: bool = False):
        """Main detection and tracking loop"""
        if display:
            cv2.namedWindow("ZED YOLO Tracker", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ZED YOLO Tracker", 1280, 720)
        
        pantilt_commands = []
        fps_counter = 0
        fps_start = time.time()
        
        try:
            while True:
                # Grab frame
                if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.001)
                    continue
                
                # Retrieve images and point cloud
                self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZ)
                
                # Convert to numpy
                image_bgra = self.left_image.get_data()
                image_bgr = image_bgra[..., :3]
                
                # Run object detection
                self.current_detections = self.detect_objects(image_bgr)
                
                # Select tracking target
                self.tracking_target = self.select_tracking_target(self.current_detections)
                
                # Calculate pan-tilt command
                if self.tracking_target:
                    pantilt_cmd = self.calculate_pantilt_command(self.tracking_target)
                    if pantilt_cmd and not self.pantilt_queue.full():
                        self.pantilt_queue.put(pantilt_cmd)
                        if save_pantilt_commands:
                            pantilt_commands.append(pantilt_cmd)
                
                # Draw visualization
                if display:
                    vis = self.draw_detections(image_bgr, self.current_detections)
                    
                    # Add FPS and status info
                    fps_counter += 1
                    if time.time() - fps_start >= 1.0:
                        fps = fps_counter / (time.time() - fps_start)
                        fps_counter = 0
                        fps_start = time.time()
                    else:
                        fps = 0
                    
                    status_text = f"FPS: {fps:.1f} | Detections: {len(self.current_detections)}"
                    if self.tracking_target:
                        status_text += f" | Tracking: {self.tracking_target.class_name}"
                        if self.tracking_target.distance:
                            status_text += f" at {self.tracking_target.distance:.2f}m"
                    
                    cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("ZED YOLO Tracker", vis)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        break
                
        except KeyboardInterrupt:
            print("\nStopping tracker...")
        
        finally:
            if display:
                cv2.destroyAllWindows()
            self.zed.close()
            
            if save_pantilt_commands and pantilt_commands:
                print(f"Saved {len(pantilt_commands)} pan-tilt commands")
                return pantilt_commands
    
    def get_latest_pantilt_command(self) -> Optional[PanTiltCommand]:
        """Get the latest pan-tilt command from queue (non-blocking)"""
        try:
            return self.pantilt_queue.get_nowait()
        except queue.Empty:
            return None


def main():
    # Initialize tracker
    tracker = ZEDYOLOTracker(
        model_path='yolov8n.pt',  # Will download automatically if not present
        target_classes=['person', 'bicycle', 'car']  # Add classes you want to track
    )
    
    # Run the tracking loop
    tracker.run_detection_loop(display=True)


if __name__ == "__main__":
    main()