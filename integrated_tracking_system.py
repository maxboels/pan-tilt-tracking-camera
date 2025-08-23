#!/usr/bin/env python3
"""
Complete ZED2 + Arduino Servo Tracking System
Integrates YOLO object detection with pan-tilt servo control
"""

import numpy as np
import cv2
import pyzed.sl as sl
import time
import threading
import queue
import serial
from ultralytics import YOLO
from typing import Optional, Tuple, List
import argparse

class ArduinoServoController:
    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connected = False
        self.command_queue = queue.Queue(maxsize=5)
        self.response_queue = queue.Queue()
        
        # Current position tracking
        self.current_pan = 0.0
        self.current_tilt = 0.0
        
        self.connect()
        
        # Start communication thread
        self.running = True
        self.comm_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.comm_thread.start()

    def connect(self):
        """Connect to Arduino"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino reset time
            
            # Clear buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # Test connection
            self.serial_conn.write(b'CENTER\n')
            self.serial_conn.flush()
            time.sleep(0.5)
            
            response = ""
            if self.serial_conn.in_waiting > 0:
                response = self.serial_conn.read(self.serial_conn.in_waiting).decode()
            
            if "OK" in response:
                self.connected = True
                print(f"Arduino servo controller connected on {self.port}")
            else:
                print(f"Arduino connection issue: {response}")
                
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            self.connected = False

    def _communication_loop(self):
        """Handle serial communication in separate thread"""
        while self.running and self.connected:
            try:
                # Send queued commands
                if not self.command_queue.empty():
                    command = self.command_queue.get_nowait()
                    
                    self.serial_conn.reset_input_buffer()
                    self.serial_conn.write((command + '\n').encode())
                    self.serial_conn.flush()
                    
                    # Read response
                    response = ""
                    start_time = time.time()
                    while time.time() - start_time < 1:
                        if self.serial_conn.in_waiting > 0:
                            data = self.serial_conn.read(self.serial_conn.in_waiting).decode()
                            response += data
                            if '\n' in data:
                                break
                        time.sleep(0.01)
                    
                    self.response_queue.put((command, response.strip()))
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                print(f"Communication error: {e}")
                time.sleep(0.1)

    def move_servo(self, channel: int, angle: float, blocking: bool = False):
        """Move servo to angle"""
        angle = max(-90, min(90, angle))  # Clamp to valid range
        command = f"SERVO,{channel},{angle}"
        
        if not self.command_queue.full():
            self.command_queue.put(command)
            
            # Update position tracking
            if channel == 0:
                self.current_pan = angle
            elif channel == 1:
                self.current_tilt = angle
                
            if blocking:
                time.sleep(0.1)  # Allow time for movement

    def move_to(self, pan_angle: float, tilt_angle: float):
        """Move to pan/tilt position"""
        self.move_servo(0, pan_angle)
        self.move_servo(1, tilt_angle)

    def center(self):
        """Center both servos"""
        if not self.command_queue.full():
            self.command_queue.put("CENTER")
        self.current_pan = 0.0
        self.current_tilt = 0.0

    def get_position(self) -> Tuple[float, float]:
        """Get current position"""
        return self.current_pan, self.current_tilt

    def close(self):
        """Clean shutdown"""
        self.running = False
        if hasattr(self, 'comm_thread'):
            self.comm_thread.join(timeout=1)
        
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(b'OFF\n')
                self.serial_conn.flush()
                time.sleep(0.5)
                self.serial_conn.close()
            except:
                pass

class ZEDTrackingSystem:
    def __init__(self, 
                 arduino_port: str = '/dev/ttyACM0',
                 yolo_model: str = 'yolov8n.pt',
                 target_classes: List[str] = None):
        
        # Initialize ZED camera
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
        
        # Get camera parameters
        self.camera_info = self.zed.get_camera_information()
        
        # Initialize YOLO
        self.yolo = YOLO(yolo_model)
        self.target_classes = target_classes or ['person']
        
        # Initialize servo controller
        self.servo_controller = ArduinoServoController(arduino_port)
        
        # ZED data containers
        self.left_image = sl.Mat()
        self.depth_map = sl.Mat()
        self.runtime_params = sl.RuntimeParameters(confidence_threshold=70)
        
        # Tracking state
        self.tracking_target = None
        self.last_detection_time = 0
        self.tracking_timeout = 3.0  # seconds
        
        # Camera field of view (ZED2 HD720)
        self.horizontal_fov = 110.0  # degrees
        self.vertical_fov = 70.0     # degrees
        
        print("ZED tracking system initialized")

    def pixel_to_angles(self, x: int, y: int, img_width: int, img_height: int) -> Tuple[float, float]:
        """Convert pixel coordinates to pan/tilt angles"""
        # Normalize to -1 to +1
        norm_x = (x - img_width / 2) / (img_width / 2)
        norm_y = (y - img_height / 2) / (img_height / 2)
        
        # Convert to angles
        pan_angle = norm_x * (self.horizontal_fov / 2)
        tilt_angle = -norm_y * (self.vertical_fov / 2)  # Negative for correct direction
        
        # Clamp to servo limits
        pan_angle = np.clip(pan_angle, -90, 90)
        tilt_angle = np.clip(tilt_angle, -30, 30)  # Tilt has smaller range
        
        return pan_angle, tilt_angle

    def detect_objects(self, image: np.ndarray) -> List[dict]:
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
                
                detection = {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (center_x, center_y),
                    'confidence': float(confidence),
                    'class_name': class_name,
                    'area': (x2 - x1) * (y2 - y1)
                }
                
                detections.append(detection)
        
        return detections

    def get_depth_at_point(self, x: int, y: int) -> Optional[float]:
        """Get depth at specific pixel"""
        try:
            depth_value = self.depth_map.get_value(x, y)
            if np.isfinite(depth_value) and depth_value > 0:
                return float(depth_value)
        except:
            pass
        return None

    def select_best_target(self, detections: List[dict]) -> Optional[dict]:
        """Select the best tracking target"""
        if not detections:
            return None
        
        # Prioritize people, then by confidence and size
        people = [d for d in detections if d['class_name'] == 'person']
        candidates = people if people else detections
        
        if not candidates:
            return None
        
        # Score by confidence and size (larger objects preferred)
        def score_detection(det):
            conf_score = det['confidence']
            size_score = min(det['area'] / 50000, 1.0)  # Normalize area
            return conf_score * 0.7 + size_score * 0.3
        
        return max(candidates, key=score_detection)

    def calculate_tracking_command(self, target: dict, img_width: int, img_height: int) -> Tuple[float, float]:
        """Calculate pan/tilt angles for target"""
        center_x, center_y = target['center']
        
        # Get depth for distance info
        depth = self.get_depth_at_point(center_x, center_y)
        
        # Convert to angles
        pan_angle, tilt_angle = self.pixel_to_angles(center_x, center_y, img_width, img_height)
        
        return pan_angle, tilt_angle

    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw detection overlays"""
        vis = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            
            # Color coding
            if detection == self.tracking_target:
                color = (0, 255, 255)  # Yellow for tracking target
                thickness = 3
            elif detection['class_name'] == 'person':
                color = (0, 255, 0)  # Green for people
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for other objects
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            cv2.circle(vis, (center_x, center_y), 5, color, -1)
            
            # Label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            
            # Add depth if available
            depth = self.get_depth_at_point(center_x, center_y)
            if depth:
                label += f" ({depth:.1f}m)"
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
            cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return vis

    def run_tracking(self, display: bool = True, duration: Optional[float] = None):
        """Main tracking loop"""
        if display:
            cv2.namedWindow("ZED Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ZED Tracking", 1280, 720)
        
        # Center servos at start
        self.servo_controller.center()
        time.sleep(1)
        
        start_time = time.time()
        frame_count = 0
        fps = 0
        
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
                
                # Convert to numpy
                image_bgra = self.left_image.get_data()
                image_bgr = image_bgra[..., :3]
                img_height, img_width = image_bgr.shape[:2]
                
                # Detect objects
                detections = self.detect_objects(image_bgr)
                
                # Select tracking target
                current_time = time.time()
                
                if detections:
                    # Update target
                    self.tracking_target = self.select_best_target(detections)
                    self.last_detection_time = current_time
                    
                    if self.tracking_target:
                        # Calculate tracking command
                        pan_angle, tilt_angle = self.calculate_tracking_command(
                            self.tracking_target, img_width, img_height)
                        
                        # Send to servos with smoothing
                        current_pan, current_tilt = self.servo_controller.get_position()
                        
                        # Smooth movement (limit change per frame)
                        max_change = 3.0  # degrees per frame
                        pan_diff = pan_angle - current_pan
                        tilt_diff = tilt_angle - current_tilt
                        
                        if abs(pan_diff) > max_change:
                            pan_angle = current_pan + max_change * np.sign(pan_diff)
                        if abs(tilt_diff) > max_change:
                            tilt_angle = current_tilt + max_change * np.sign(tilt_diff)
                        
                        self.servo_controller.move_to(pan_angle, tilt_angle)
                
                else:
                    # No detections - check timeout
                    if current_time - self.last_detection_time > self.tracking_timeout:
                        if self.tracking_target:
                            print("Tracking timeout - returning to center")
                            self.servo_controller.center()
                            self.tracking_target = None
                
                # Display
                if display:
                    vis = self.draw_detections(image_bgr, detections)
                    
                    # Add status info
                    frame_count += 1
                    if frame_count % 30 == 0:  # Update FPS every 30 frames
                        fps = 30 / (time.time() - (start_time + (frame_count-30)/fps if fps > 0 else start_time))
                    
                    pan, tilt = self.servo_controller.get_position()
                    
                    status_lines = [
                        f"FPS: {fps:.1f}",
                        f"Detections: {len(detections)}",
                        f"Tracking: {'Yes' if self.tracking_target else 'No'}",
                        f"Pan/Tilt: {pan:.1f}°/{tilt:.1f}°"
                    ]
                    
                    for i, line in enumerate(status_lines):
                        cv2.putText(vis, line, (10, 30 + i*25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add crosshair at center
                    center_x, center_y = img_width // 2, img_height // 2
                    cv2.drawMarker(vis, (center_x, center_y), (0, 255, 255), 
                                  cv2.MARKER_CROSS, 20, 2)
                    
                    cv2.imshow("ZED Tracking", vis)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        break
                    elif key == ord('c'):  # center
                        self.servo_controller.center()
                        print("Servos centered manually")
                
        except KeyboardInterrupt:
            print("\nStopping tracking...")
            
        finally:
            # Cleanup
            self.servo_controller.center()
            time.sleep(1)
            self.servo_controller.close()
            self.zed.close()
            if display:
                cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            print(f"Processed {frame_count} frames in {elapsed:.1f}s")
            if elapsed > 0:
                print(f"Average FPS: {frame_count/elapsed:.1f}")

def main():
    parser = argparse.ArgumentParser(description="ZED2 + Arduino Servo Tracking System")
    parser.add_argument("--arduino-port", default="/dev/ttyACM0", help="Arduino serial port")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--target-classes", nargs='+', default=['person'], 
                       help="Object classes to track")
    parser.add_argument("--no-display", action="store_true", help="Run without display")
    parser.add_argument("--duration", type=float, help="Run duration in seconds")
    
    args = parser.parse_args()
    
    print("=== ZED2 + Arduino Servo Tracking System ===")
    print(f"Arduino port: {args.arduino_port}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Target classes: {args.target_classes}")
    print("Controls: 'q' to quit, 'c' to center servos")
    print()
    
    try:
        tracker = ZEDTrackingSystem(
            arduino_port=args.arduino_port,
            yolo_model=args.yolo_model,
            target_classes=args.target_classes
        )
        
        tracker.run_tracking(
            display=not args.no_display,
            duration=args.duration
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()