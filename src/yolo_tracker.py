#!/usr/bin/env python3
"""
YOLO-based Object Tracker for Pan-Tilt Camera
Handles person detection and tracking using YOLO models
"""

import numpy as np
import cv2
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Detection:
    """Detection data structure"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int


class YOLOTracker:
    """YOLO-based object tracker optimized for person tracking"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """Initialize YOLO tracker"""
        print(f"Loading YOLO model: {model_path}")
        self.yolo = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.target_classes = ["person"]  # Focus on people
        
        # Tracking state
        self.tracking_target = None
        self.tracking_history = []
        self.max_history = 10
        
        print("YOLO tracker initialized")
    
    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame using YOLO"""
        results = self.yolo(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Extract detection data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.yolo.names[class_id]
                
                # Filter by confidence and target classes
                if confidence < self.confidence_threshold:
                    continue
                if class_name not in self.target_classes:
                    continue
                
                # Calculate center and area
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                area = int((x2 - x1) * (y2 - y1))
                
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=(center_x, center_y),
                    area=area
                )
                
                detections.append(detection)
        
        return detections
    
    def select_best_target(self, detections: List[Detection]) -> Optional[Detection]:
        """Select the best detection to track"""
        if not detections:
            return None
        
        # Prefer larger detections (closer people)
        # and those with higher confidence
        scored_detections = []
        for detection in detections:
            # Score based on area and confidence
            area_score = min(detection.area / 50000, 1.0)  # Normalize area
            confidence_score = detection.confidence
            total_score = (area_score * 0.6) + (confidence_score * 0.4)
            scored_detections.append((total_score, detection))
        
        # Return highest scoring detection
        return max(scored_detections, key=lambda x: x[0])[1]
    
    def update_tracking(self, detections: List[Detection]) -> Optional[Detection]:
        """Update tracking target"""
        current_target = self.select_best_target(detections)
        
        if current_target:
            self.tracking_target = current_target
            # Add to history for smoothing
            self.tracking_history.append(current_target.center)
            if len(self.tracking_history) > self.max_history:
                self.tracking_history.pop(0)
        else:
            # Gradually fade out tracking if no detections
            if len(self.tracking_history) > 0:
                self.tracking_history.pop(0)
            if not self.tracking_history:
                self.tracking_target = None
        
        return self.tracking_target
    
    def get_smoothed_target_position(self) -> Optional[Tuple[int, int]]:
        """Get smoothed target position from history"""
        if not self.tracking_history:
            return None
        
        # Use weighted average of recent positions
        weights = np.linspace(0.5, 1.0, len(self.tracking_history))
        weights = weights / weights.sum()
        
        x_coords = [pos[0] for pos in self.tracking_history]
        y_coords = [pos[1] for pos in self.tracking_history]
        
        smoothed_x = int(np.average(x_coords, weights=weights))
        smoothed_y = int(np.average(y_coords, weights=weights))
        
        return (smoothed_x, smoothed_y)
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection overlays on frame"""
        vis_frame = frame.copy()
        
        # Draw all detections
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            center_x, center_y = detection.center
            
            # Color coding
            if detection == self.tracking_target:
                color = (0, 255, 255)  # Yellow for tracking target
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for other people
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            cv2.circle(vis_frame, (center_x, center_y), 8, color, -1)
            cv2.circle(vis_frame, (center_x, center_y), 8, (255, 255, 255), 2)
            
            # Label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(vis_frame, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw smoothed tracking point if available
        smoothed_pos = self.get_smoothed_target_position()
        if smoothed_pos:
            cv2.circle(vis_frame, smoothed_pos, 12, (255, 0, 255), 3)  # Magenta
            cv2.circle(vis_frame, smoothed_pos, 12, (255, 255, 255), 1)
        
        # Draw tracking history trail
        for i, pos in enumerate(self.tracking_history):
            alpha = (i + 1) / len(self.tracking_history)
            radius = int(3 + alpha * 3)
            color_intensity = int(100 + alpha * 155)
            cv2.circle(vis_frame, pos, radius, (color_intensity, 0, color_intensity), -1)
        
        return vis_frame
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.tracking_target = None
        self.tracking_history.clear()
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
        
        if tracker.tracking_target:
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
