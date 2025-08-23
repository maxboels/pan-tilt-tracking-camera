#!/usr/bin/env python3
"""
Complete ZED2 Tracking System with Servo Control
Based on the working tracking system, now adds servo integration
"""

import numpy as np
import cv2
import pyzed.sl as sl
import time
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import serial
import threading
import queue
import torch

# Add this check right after your imports
print("--- Hardware Check ---")
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU. This will be very slow.")
print("--------------------")

@dataclass
class Detection:
    """Detection data structure"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_2d: Tuple[int, int]
    distance: Optional[float] = None

class ServoController:
    """Simple servo controller for Arduino bridge"""
    def __init__(self, port: str = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connected = False
        
        # Current positions
        self.current_pan = 0.0
        self.current_tilt = 0.0
        
        # Movement limits and smoothing
        self.max_change_per_frame = 2.0  # degrees
        
        if port is None:
            port = self.find_arduino_port()
            
        if port:
            self.port = port
            self.connect()
        else:
            print("No Arduino found on available ports")

    def find_arduino_port(self):
        """Auto-detect Arduino port"""
        import serial.tools.list_ports
        
        # Common Arduino port patterns
        arduino_patterns = ['arduino', 'usb', 'acm', 'ch340', 'cp210', 'ftdi']
        
        ports = serial.tools.list_ports.comports()
        for port in ports:
            description = port.description.lower()
            if any(pattern in description for pattern in arduino_patterns):
                print(f"Found potential Arduino: {port.device} - {port.description}")
                return port.device
        
        # Try common Arduino ports as fallback
        common_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
        for port in common_ports:
            try:
                test_serial = serial.Serial(port, 115200, timeout=1)
                test_serial.close()
                print(f"Found working port: {port}")
                return port
            except:
                continue
        
        return None

    def connect(self):
        """Connect to Arduino servo bridge"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino reset time
            
            # Test with center command
            self.serial_conn.write(b'CENTER\n')
            self.serial_conn.flush()
            time.sleep(0.5)
            
            response = ""
            if self.serial_conn.in_waiting > 0:
                response = self.serial_conn.read(self.serial_conn.in_waiting).decode()
            
            # THE FIX IS HERE:
            # We now check for "Ready" which is in your Arduino's response message.
            if "Ready" in response:
                self.connected = True
                print(f"Servo controller connected on {self.port}")
            else:
                print(f"Servo response did not contain 'Ready': '{response.strip()}'")
                
        except Exception as e:
            print(f"Servo connection failed: {e}")
            self.connected = False

    def send_command(self, command: str) -> bool:
        """Send command to servo controller"""
        if not self.connected or not self.serial_conn:
            return False
        
        try:
            self.serial_conn.write((command + '\n').encode())
            self.serial_conn.flush()
            time.sleep(0.05)  # Short delay for command processing
            return True
        except Exception as e:
            print(f"Servo command failed: {e}")
            return False

    def move_to(self, pan_angle: float, tilt_angle: float, smooth: bool = True) -> bool:
        """Move servos to specified angles with optional smoothing"""
        # Clamp angles to safe ranges
        pan_angle = np.clip(pan_angle, -90, 90)
        tilt_angle = np.clip(tilt_angle, -30, 30)
        
        if smooth:
            # Limit movement speed for smooth tracking
            pan_diff = pan_angle - self.current_pan
            tilt_diff = tilt_angle - self.current_tilt
            
            if abs(pan_diff) > self.max_change_per_frame:
                pan_angle = self.current_pan + self.max_change_per_frame * np.sign(pan_diff)
            if abs(tilt_diff) > self.max_change_per_frame:
                tilt_angle = self.current_tilt + self.max_change_per_frame * np.sign(tilt_diff)
        
        # Send commands
        success = True
        if abs(pan_angle - self.current_pan) > 0.5:  # Only move if significant change
            success &= self.send_command(f"SERVO,0,{pan_angle:.1f}")
            self.current_pan = pan_angle
        
        if abs(tilt_angle - self.current_tilt) > 0.5:
            success &= self.send_command(f"SERVO,1,{tilt_angle:.1f}")
            self.current_tilt = tilt_angle
        
        return success

    def center(self) -> bool:
        """Center both servos"""
        success = self.send_command("CENTER")
        if success:
            self.current_pan = 0.0
            self.current_tilt = 0.0
        return success

    def get_position(self) -> Tuple[float, float]:
        """Get current servo positions"""
        return self.current_pan, self.current_tilt

    def close(self):
        """Close connection"""
        if self.connected and self.serial_conn:
            self.send_command("CENTER")
            time.sleep(1)
            self.send_command("OFF")
            self.serial_conn.close()
            self.connected = False

class CompleteTrackingSystem:
    def __init__(self, config_path: str = "config.json", enable_servos: bool = True):
        """Initialize complete tracking system"""
        print("Initializing complete ZED2 tracking system...")
        
        # Load config
        self.config = self.load_config(config_path)
        self.enable_servos = enable_servos
        
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
        self.runtime_params = sl.RuntimeParameters(confidence_threshold=70)
        
        # Initialize YOLO
        print("Loading YOLO model...")
        self.yolo = YOLO(self.config['yolo']['model_path'])
        self.target_classes = self.config['yolo']['target_classes']
        
        # Initialize servo controller
        self.servo_controller = None
        if self.enable_servos:
            try:
                print("Initializing servo controller...")
                self.servo_controller = ServoController()
                if self.servo_controller.connected:
                    print("Servo control enabled")
                else:
                    print("Servo control disabled - connection failed")
                    self.servo_controller = None
            except Exception as e:
                print(f"Servo initialization failed: {e}")
                self.servo_controller = None
        
        # Tracking state
        self.tracking_target = None
        self.last_detection_time = 0
        self.tracking_timeout = 3.0
        
        # Camera FOV
        self.horizontal_fov = 110.0
        self.vertical_fov = 70.0
        
        print("Complete tracking system ready!")
        if self.servo_controller:
            print("Servo control: ENABLED")
        else:
            print("Servo control: DISABLED (camera tracking only)")

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

    def get_depth_at_point(self, x: int, y: int) -> Optional[float]:
        """Get depth at specific pixel"""
        try:
            depth_value = self.depth_map.get_value(x, y)
            if np.isfinite(depth_value) and depth_value > 0.1:
                return float(depth_value)
        except:
            pass
        return None

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
                
                if self.target_classes and class_name not in self.target_classes:
                    continue
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                distance = self.get_depth_at_point(center_x, center_y)
                
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center_2d=(center_x, center_y),
                    distance=distance
                )
                
                detections.append(detection)
        
        return detections

    def select_tracking_target(self, detections: List[Detection]) -> Optional[Detection]:
        """Select best target to track"""
        if not detections:
            return None
        
        # Prefer people, then by confidence/distance
        people = [d for d in detections if d.class_name == 'person']
        candidates = people if people else detections
        
        # Score by confidence and distance (closer = better)
        def score_target(det):
            conf_score = det.confidence
            dist_score = 1.0 / (det.distance + 0.1) if det.distance else 0.5
            return conf_score * 0.6 + dist_score * 0.4
        
        return max(candidates, key=score_target)

    def pixel_to_angles(self, x: int, y: int, img_width: int, img_height: int) -> Tuple[float, float]:
        """Convert pixel coordinates to servo angles"""
        norm_x = (x - img_width / 2) / (img_width / 2)
        norm_y = (y - img_height / 2) / (img_height / 2)
        
        pan_angle = norm_x * (self.horizontal_fov / 2)
        tilt_angle = -norm_y * (self.vertical_fov / 2)
        
        # Scale down for smoother servo movement
        pan_angle *= 0.8  # Reduce sensitivity
        tilt_angle *= 0.8
        
        pan_angle = np.clip(pan_angle, -90, 90)
        tilt_angle = np.clip(tilt_angle, -30, 30)
        
        return pan_angle, tilt_angle

    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection overlays"""
        vis = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            center_x, center_y = detection.center_2d
            
            if detection == self.tracking_target:
                color = (0, 255, 255)  # Yellow
                thickness = 3
            elif detection.class_name == 'person':
                color = (0, 255, 0)  # Green
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue
                thickness = 2
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(vis, (center_x, center_y), 5, color, -1)
            
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            if detection.distance:
                label += f" ({detection.distance:.2f}m)"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
            cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return vis


    # fast tracking
    def run_tracking(self, duration: Optional[float] = None):
        """Main tracking loop with performance optimizations."""
        print("Starting complete tracking system...")
        print("Controls:")
        print("  'q' - Quit, 'c' - Center servos, 'r' - Reset tracking, 't' - Toggle servo tracking")
        
        cv2.namedWindow("Complete ZED2 Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Complete ZED2 Tracking", 1280, 720)
        
        start_time = time.time()
        frame_count = 0
        fps_counter = 0
        fps_start = time.time()
        current_fps = 0
        servo_tracking_enabled = True
        
        # --- OPTIMIZATION: Define a smaller size for YOLO processing ---
        yolo_processing_size = (640, 360) 

        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.001)
                    continue
                
                self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                
                image_bgr = self.left_image.get_data()[..., :3]
                img_height, img_width = image_bgr.shape[:2]
                
                # --- OPTIMIZATION: Resize image for YOLO ---
                # 1. Create a small image for fast processing
                input_small = cv2.resize(image_bgr, yolo_processing_size)
                
                # 2. Run detection on the small image
                detections = self.detect_objects(input_small)
                
                # 3. Scale detection coordinates back to the original image size
                scale_x = img_width / yolo_processing_size[0]
                scale_y = img_height / yolo_processing_size[1]
                
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    det.bbox = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
                    cx, cy = det.center_2d
                    det.center_2d = (int(cx * scale_x), int(cy * scale_y))
                    # The depth is now looked up on the original full-res depth map
                    det.distance = self.get_depth_at_point(det.center_2d[0], det.center_2d[1])
                
                # --- End of Optimization ---

                # Tracking logic remains the same
                current_time = time.time()
                
                if detections:
                    self.tracking_target = self.select_tracking_target(detections)
                    self.last_detection_time = current_time
                    
                    if (self.tracking_target and self.servo_controller and 
                        self.servo_controller.connected and servo_tracking_enabled):
                        
                        center_x, center_y = self.tracking_target.center_2d
                        pan_angle, tilt_angle = self.pixel_to_angles(center_x, center_y, img_width, img_height)
                        
                        self.servo_controller.move_to(pan_angle, tilt_angle, smooth=True)
                else:
                    if time.time() - self.last_detection_time > self.tracking_timeout:
                        if self.tracking_target:
                            print("Lost target")
                            self.tracking_target = None
                
                # Visualization
                vis = self.draw_detections(image_bgr, detections)
                
                # FPS and Status display... (rest of the function is the same)
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()
                
                status_lines = [
                    f"FPS: {current_fps:.1f}",
                    f"Detections: {len(detections)}",
                    f"Tracking: {'Yes' if self.tracking_target else 'No'}"
                ]
                if self.servo_controller and self.servo_controller.connected:
                    pan, tilt = self.servo_controller.get_position()
                    status_lines.append(f"Servos: Pan {pan:.1f}°, Tilt {tilt:.1f}°")
                    status_lines.append(f"Servo Tracking: {'ON' if servo_tracking_enabled else 'OFF'}")
                else:
                    status_lines.append("Servos: DISCONNECTED")
                
                if self.tracking_target:
                    status_lines.append(f"Target: {self.tracking_target.class_name}")
                    if self.tracking_target.distance:
                        status_lines.append(f"Distance: {self.tracking_target.distance:.2f}m")
                
                for i, line in enumerate(status_lines):
                    cv2.putText(vis, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    cv2.putText(vis, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                h, w = vis.shape[:2]
                center_x, center_y = w // 2, h // 2
                cv2.drawMarker(vis, (center_x, center_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                
                if self.tracking_target:
                    target_x, target_y = self.tracking_target.center_2d
                    cv2.arrowedLine(vis, (center_x, center_y), (target_x, target_y), (0, 255, 255), 3)
                
                cv2.imshow("Complete ZED2 Tracking", vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: break
                elif key == ord('c') and self.servo_controller: self.servo_controller.center()
                elif key == ord('r'): self.tracking_target = None
                elif key == ord('t'): servo_tracking_enabled = not servo_tracking_enabled
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            if self.servo_controller:
                self.servo_controller.center()
                time.sleep(1)
                self.servo_controller.close()
            
            self.zed.close()
            cv2.destroyAllWindows()
            elapsed = time.time() - start_time
            print(f"Session complete: {frame_count} frames in {elapsed:.1f}s")
            if elapsed > 0: print(f"Average FPS: {frame_count/elapsed:.1f}")

    def run_tracking_slow(self, duration: Optional[float] = None):
        """Main tracking loop"""
        print("Starting complete tracking system...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Center servos")
        print("  'r' - Reset tracking")
        print("  't' - Toggle servo tracking")
        
        cv2.namedWindow("Complete ZED2 Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Complete ZED2 Tracking", 1280, 720)
        
        start_time = time.time()
        frame_count = 0
        fps_counter = 0
        fps_start = time.time()
        current_fps = 0
        servo_tracking_enabled = True
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.001)
                    continue
                
                self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
                
                image_bgra = self.left_image.get_data()
                image_bgr = image_bgra[..., :3]
                img_height, img_width = image_bgr.shape[:2]
                
                # Detection and tracking
                detections = self.detect_objects(image_bgr)
                current_time = time.time()
                
                if detections:
                    self.tracking_target = self.select_tracking_target(detections)
                    self.last_detection_time = current_time
                    
                    # Servo control
                    if (self.tracking_target and self.servo_controller and 
                        self.servo_controller.connected and servo_tracking_enabled):
                        
                        center_x, center_y = self.tracking_target.center_2d
                        pan_angle, tilt_angle = self.pixel_to_angles(center_x, center_y, img_width, img_height)
                        
                        # Move servos with smoothing
                        self.servo_controller.move_to(pan_angle, tilt_angle, smooth=True)
                else:
                    # Check timeout
                    if current_time - self.last_detection_time > self.tracking_timeout:
                        if self.tracking_target:
                            print("Lost target")
                            self.tracking_target = None
                
                # Visualization
                vis = self.draw_detections(image_bgr, detections)
                
                # FPS
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()
                
                # Status
                status_lines = [
                    f"FPS: {current_fps:.1f}",
                    f"Detections: {len(detections)}",
                    f"Tracking: {'Yes' if self.tracking_target else 'No'}"
                ]
                
                if self.servo_controller and self.servo_controller.connected:
                    pan, tilt = self.servo_controller.get_position()
                    status_lines.append(f"Servos: Pan {pan:.1f}°, Tilt {tilt:.1f}°")
                    status_lines.append(f"Servo Tracking: {'ON' if servo_tracking_enabled else 'OFF'}")
                else:
                    status_lines.append("Servos: DISCONNECTED")
                
                if self.tracking_target:
                    status_lines.append(f"Target: {self.tracking_target.class_name}")
                    if self.tracking_target.distance:
                        status_lines.append(f"Distance: {self.tracking_target.distance:.2f}m")
                
                # Draw status with outline
                for i, line in enumerate(status_lines):
                    cv2.putText(vis, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 0), 3)
                    cv2.putText(vis, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                
                # Crosshair and tracking arrow
                h, w = vis.shape[:2]
                center_x, center_y = w // 2, h // 2
                cv2.drawMarker(vis, (center_x, center_y), (0, 255, 255), 
                              cv2.MARKER_CROSS, 20, 2)
                
                if self.tracking_target:
                    target_x, target_y = self.tracking_target.center_2d
                    cv2.arrowedLine(vis, (center_x, center_y), (target_x, target_y), 
                                   (0, 255, 255), 3)
                
                cv2.imshow("Complete ZED2 Tracking", vis)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('c') and self.servo_controller:
                    print("Centering servos...")
                    self.servo_controller.center()
                elif key == ord('r'):
                    self.tracking_target = None
                    print("Tracking reset")
                elif key == ord('t'):
                    servo_tracking_enabled = not servo_tracking_enabled
                    print(f"Servo tracking: {'ON' if servo_tracking_enabled else 'OFF'}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        
        finally:
            # Cleanup
            if self.servo_controller:
                print("Centering and shutting down servos...")
                self.servo_controller.center()
                time.sleep(1)
                self.servo_controller.close()
            
            self.zed.close()
            cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            print(f"Session complete: {frame_count} frames in {elapsed:.1f}s")
            if elapsed > 0:
                print(f"Average FPS: {frame_count/elapsed:.1f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete ZED2 Tracking with Servos")
    parser.add_argument("--config", default="config.json", help="Config file")
    parser.add_argument("--duration", type=float, help="Run duration in seconds")
    parser.add_argument("--no-servos", action="store_true", help="Disable servo control")
    
    args = parser.parse_args()
    
    try:
        tracker = CompleteTrackingSystem(args.config, enable_servos=not args.no_servos)
        tracker.run_tracking(args.duration)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()