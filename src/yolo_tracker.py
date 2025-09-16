#!/usr/bin/env python3
"""
YOLO-based Object Tracker for Pan-Tilt Camera
Handles person detection and tracking using YOLO models
"""

import numpy as np
import cv2
from ultralytics import YOLO  # Import YOLO directly
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os


@dataclass
class Detection:
    """Detection data structure"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int


class DetectedObject:
    """Simple class to hold detection information"""
    def __init__(self, class_name, confidence, bbox, center):
        """Initialize detection object"""
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.center = center  # (x, y)
        self.track_id = None
        self.track_history = []
        # Calculate area from bbox
        x1, y1, x2, y2 = bbox
        self.area = (x2 - x1) * (y2 - y1)


class YOLOTracker:
    """YOLOv8-based object detector and tracker"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5, track_head: bool = False):
        """Initialize YOLOv8 detector"""
        print(f"Loading YOLO model: {model_path}")
        try:
            # Use YOLO directly instead of torch.hub
            self.model = YOLO(model_path)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            # Fallback to local file if model_path exists
            if os.path.exists(model_path):
                print(f"Trying to load model directly from file: {model_path}")
                try:
                    self.model = YOLO(model_path)
                    print("YOLO model loaded from local file successfully")
                except Exception as e2:
                    print(f"Failed to load model from file: {e2}")
                    raise RuntimeError(f"Could not load YOLO model: {e2}")
            else:
                print(f"Model file not found at: {model_path}")
                raise RuntimeError(f"YOLO model file not found: {model_path}")
        
        self.confidence_threshold = confidence_threshold
        self.track_head = track_head
        
        # Tracking parameters
        self.current_target = None
        self.target_history = []
        self.max_history = 10  # Maximum history for smoothing
        
        # Object ID to track (if multiple people are detected)
        self.tracked_id = None
        
        print("YOLO tracker initialized")

    def detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detect objects in frame using YOLO"""
        results = self.model(frame, verbose=False)
        detections = []
        
        try:
            # Process results from YOLOv8
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence
                        confidence = float(box.conf[0].item())
                        
                        # Get class
                        class_id = int(box.cls[0].item())
                        class_name = result.names[class_id]
                        
                        if class_name == "person" and confidence >= self.confidence_threshold:
                            bbox = (int(x1), int(y1), int(x2), int(y2))
                            
                            # Always use center of bounding box for tracking
                            # (removed head position estimation)
                            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            
                            detection = DetectedObject(class_name, confidence, bbox, center)
                            detections.append(detection)
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        continue
        except Exception as e:
            print(f"Error processing YOLO results: {e}")
        
        return detections
    
    def select_best_target(self, detections):
        """Select the best detection to track"""
        if not detections:
            return None
        
        # Prefer larger detections (closer people)
        # and those with higher confidence
        scored_detections = []
        for detection in detections:
            # Score based on area and confidence
            # Handle both Detection and DetectedObject classes
            if hasattr(detection, 'area'):
                area = detection.area
            else:
                # Calculate area from bounding box
                x1, y1, x2, y2 = detection.bbox
                area = (x2 - x1) * (y2 - y1)
            
            area_score = min(area / 50000, 1.0)  # Normalize area
            confidence_score = detection.confidence
            total_score = (area_score * 0.6) + (confidence_score * 0.4)
            scored_detections.append((total_score, detection))
        
        # Return highest scoring detection
        if scored_detections:
            return max(scored_detections, key=lambda x: x[0])[1]
        return None
    
    def update_tracking(self, detections):
        """Update tracking target"""
        current_target = self.select_best_target(detections)
        
        if current_target:
            self.current_target = current_target
            # Add to history for smoothing
            self.target_history.append(current_target.center)
            if len(self.target_history) > self.max_history:
                self.target_history.pop(0)
        else:
            # Gradually fade out tracking if no detections
            if len(self.target_history) > 0:
                self.target_history.pop(0)
            if not self.target_history:
                self.current_target = None
        
        return self.current_target
    
    def get_smoothed_target_position(self) -> Optional[Tuple[int, int]]:
        """Get smoothed target position from history"""
        if not self.target_history:
            return None
        
        # Use weighted average of recent positions
        weights = np.linspace(0.5, 1.0, len(self.target_history))
        weights = weights / weights.sum()
        
        x_coords = [pos[0] for pos in self.target_history]
        y_coords = [pos[1] for pos in self.target_history]
        
        smoothed_x = int(np.average(x_coords, weights=weights))
        smoothed_y = int(np.average(y_coords, weights=weights))
        
        return (smoothed_x, smoothed_y)
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """Draw detection overlays on frame"""
        result_frame = frame.copy()
        
        for det in detections:
            # Always draw the person bounding box
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(result_frame, det.center, 5, (255, 0, 0), -1)
        
        # Draw current tracking target with gentle highlight
        if self.current_target:
            smoothed_pos = self.get_smoothed_target_position()
            if smoothed_pos:
                # Fuschia/pink color (255, 0, 255) instead of red (0, 0, 255)
                cv2.circle(result_frame, smoothed_pos, 10, (255, 0, 255), 2)
                
                # Draw tracking indicator with gentler fuschia/pink color
                cv2.line(result_frame, 
                       (smoothed_pos[0] - 15, smoothed_pos[1]), 
                       (smoothed_pos[0] + 15, smoothed_pos[1]), 
                       (255, 0, 255), 2)
                cv2.line(result_frame, 
                       (smoothed_pos[0], smoothed_pos[1] - 15), 
                       (smoothed_pos[0], smoothed_pos[1] + 15), 
                       (255, 0, 255), 2)
                
                # Label as person tracking with gentler color
                cv2.putText(result_frame, "PERSON TRACKING", 
                          (smoothed_pos[0] + 20, smoothed_pos[1] + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return result_frame
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.current_target = None
        self.target_history.clear()
        print("Tracking reset")


if __name__ == "__main__":
    # Test the tracker with webcam
    tracker = YOLOTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("YOLO Tracker Test - Press 'q' to quit, 'r' to reset tracking")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track
        detections = tracker.detect_objects(frame)
        tracker.update_tracking(detections)
        
        # Draw visualization
        vis_frame = tracker.draw_detections(frame, detections)
        
        # Add frame info
        cv2.putText(vis_frame, f"Detections: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if tracker.current_target:
            target_pos = tracker.get_smoothed_target_position()
            cv2.putText(vis_frame, f"Tracking: {target_pos}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('YOLO Tracker Test', vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset_tracking()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test the tracker with webcam
    tracker = YOLOTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("YOLO Tracker Test - Press 'q' to quit, 'r' to reset tracking")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track
        detections = tracker.detect_objects(frame)
        tracker.update_tracking(detections)
        
        # Draw visualization
        vis_frame = tracker.draw_detections(frame, detections)
        
        # Add frame info
        cv2.putText(vis_frame, f"Detections: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if tracker.current_target:
            target_pos = tracker.get_smoothed_target_position()
            cv2.putText(vis_frame, f"Tracking: {target_pos}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('YOLO Tracker Test', vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset_tracking()
    
    cap.release()
    cv2.destroyAllWindows()
