#!/usr/bin/env python3
"""
Diagnostic version to isolate the moving object freeze issue
"""

import numpy as np
import cv2
import pyzed.sl as sl
import time
from ultralytics import YOLO

class DiagnosticTracker:
    def __init__(self):
        print("Setting up diagnostic tracker...")
        
        # Initialize ZED
        self.zed = sl.Camera()
        init_params = sl.InitParameters(
            camera_resolution=sl.RESOLUTION.HD720,
            depth_mode=sl.DEPTH_MODE.PERFORMANCE,
            coordinate_units=sl.UNIT.METER
        )
        
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to open ZED")
        
        self.left_image = sl.Mat()
        self.depth_map = sl.Mat()
        self.point_cloud = sl.Mat()
        self.runtime_params = sl.RuntimeParameters(confidence_threshold=70)
        
        # Initialize YOLO
        self.yolo = YOLO('yolov8n.pt')
        
        print("Diagnostic tracker ready")

    def test_mode_1_yolo_only(self):
        """Test 1: YOLO detection only, no 3D data"""
        print("\n=== TEST 1: YOLO Detection Only ===")
        cv2.namedWindow("Test 1: YOLO Only", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        detection_count = 0
        
        try:
            while frame_count < 100:  # Limit frames for testing
                if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
                    continue
                
                self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
                image_bgra = self.left_image.get_data()
                image_bgr = image_bgra[..., :3]
                
                # Time the YOLO inference
                start_time = time.time()
                results = self.yolo(image_bgr, verbose=False)
                inference_time = time.time() - start_time
                
                # Count detections
                total_detections = 0
                for result in results:
                    if result.boxes is not None:
                        total_detections += len(result.boxes)
                
                if total_detections > 0:
                    detection_count += 1
                    print(f"Frame {frame_count}: {total_detections} objects, inference: {inference_time*1000:.1f}ms")
                
                # Simple visualization
                vis = image_bgr.copy()
                cv2.putText(vis, f"Frame: {frame_count}, Detections: {total_detections}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis, f"Inference: {inference_time*1000:.1f}ms", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Test 1: YOLO Only", vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                frame_count += 1
                
        except Exception as e:
            print(f"Test 1 failed: {e}")
        
        cv2.destroyAllWindows()
        print(f"Test 1 complete: {detection_count} frames with detections out of {frame_count}")

    def test_mode_2_with_3d_lookup(self):
        """Test 2: YOLO + 3D coordinate lookup"""
        print("\n=== TEST 2: YOLO + 3D Lookup ===")
        cv2.namedWindow("Test 2: With 3D", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        
        try:
            while frame_count < 100:
                if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
                    continue
                
                # Get all data
                self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZ)
                
                image_bgra = self.left_image.get_data()
                image_bgr = image_bgra[..., :3]
                
                # YOLO detection
                start_time = time.time()
                results = self.yolo(image_bgr, verbose=False)
                yolo_time = time.time() - start_time
                
                # Process detections with 3D lookup
                start_3d = time.time()
                detection_info = []
                
                for result in results:
                    if result.boxes is None:
                        continue
                    
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # This is where it might freeze - 3D coordinate lookup
                        try:
                            point_cloud_value = self.point_cloud.get_value(center_x, center_y)
                            if np.isfinite(point_cloud_value).all():
                                distance = float(point_cloud_value[2])
                                detection_info.append((center_x, center_y, distance))
                            else:
                                detection_info.append((center_x, center_y, None))
                        except Exception as e:
                            print(f"3D lookup failed: {e}")
                            detection_info.append((center_x, center_y, None))
                
                lookup_time = time.time() - start_3d
                
                if detection_info:
                    print(f"Frame {frame_count}: {len(detection_info)} objects, "
                          f"YOLO: {yolo_time*1000:.1f}ms, 3D: {lookup_time*1000:.1f}ms")
                
                # Visualization
                vis = image_bgr.copy()
                cv2.putText(vis, f"Frame: {frame_count}, Objects: {len(detection_info)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis, f"YOLO: {yolo_time*1000:.1f}ms, 3D: {lookup_time*1000:.1f}ms", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw detection centers
                for center_x, center_y, distance in detection_info:
                    color = (0, 255, 0) if distance else (0, 0, 255)
                    cv2.circle(vis, (center_x, center_y), 5, color, -1)
                    if distance:
                        cv2.putText(vis, f"{distance:.2f}m", (center_x+10, center_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                cv2.imshow("Test 2: With 3D", vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                frame_count += 1
                
        except Exception as e:
            print(f"Test 2 failed: {e}")
            import traceback
            traceback.print_exc()
        
        cv2.destroyAllWindows()
        print(f"Test 2 complete: {frame_count} frames processed")

    def test_mode_3_minimal_detection(self):
        """Test 3: Minimal detection processing"""
        print("\n=== TEST 3: Minimal Detection Processing ===")
        cv2.namedWindow("Test 3: Minimal", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        
        try:
            while frame_count < 100:
                if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
                    continue
                
                self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
                image_bgra = self.left_image.get_data()
                image_bgr = image_bgra[..., :3]
                
                # YOLO with minimal processing
                results = self.yolo(image_bgr, verbose=False)
                
                # Just count, don't process
                detection_count = 0
                if results[0].boxes is not None:
                    detection_count = len(results[0].boxes)
                
                # Minimal visualization
                vis = image_bgr.copy()
                cv2.putText(vis, f"Frame: {frame_count}, Count: {detection_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if detection_count > 0:
                    print(f"Frame {frame_count}: {detection_count} detections (minimal processing)")
                
                cv2.imshow("Test 3: Minimal", vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                frame_count += 1
                
        except Exception as e:
            print(f"Test 3 failed: {e}")
        
        cv2.destroyAllWindows()
        print(f"Test 3 complete: {frame_count} frames processed")

    def run_diagnostic_sequence(self):
        """Run all diagnostic tests in sequence"""
        print("Starting diagnostic sequence...")
        print("Move in front of the camera to trigger detections")
        
        tests = [
            ("YOLO Detection Only", self.test_mode_1_yolo_only),
            ("Minimal Detection", self.test_mode_3_minimal_detection), 
            ("YOLO + 3D Lookup", self.test_mode_2_with_3d_lookup)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"Starting: {test_name}")
            print("Press 'q' to move to next test")
            input("Press Enter to start this test...")
            
            try:
                test_func()
                print(f"{test_name}: COMPLETED")
            except Exception as e:
                print(f"{test_name}: FAILED - {e}")
            
            time.sleep(1)
        
        print("\nDiagnostic sequence complete!")

    def cleanup(self):
        """Clean up resources"""
        self.zed.close()
        cv2.destroyAllWindows()


def main():
    tracker = DiagnosticTracker()
    
    try:
        tracker.run_diagnostic_sequence()
    except KeyboardInterrupt:
        print("\nDiagnostic interrupted")
    finally:
        tracker.cleanup()


if __name__ == "__main__":
    main()