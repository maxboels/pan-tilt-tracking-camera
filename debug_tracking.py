#!/usr/bin/env python3
"""
Debug version of tracking system to isolate the freezing issue
"""

import numpy as np
import cv2
import pyzed.sl as sl
import time
import sys

def test_basic_zed():
    """Test basic ZED functionality"""
    print("Testing basic ZED camera...")
    
    zed = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.HD720,
        depth_mode=sl.DEPTH_MODE.PERFORMANCE,
        coordinate_units=sl.UNIT.METER
    )
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED: {err}")
        return False
    
    print("ZED camera opened successfully")
    
    # Test grabbing a few frames
    runtime = sl.RuntimeParameters()
    left = sl.Mat()
    
    for i in range(5):
        print(f"Grabbing frame {i+1}...")
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left, sl.VIEW.LEFT)
            image = left.get_data()
            print(f"Frame {i+1}: {image.shape}")
        else:
            print(f"Failed to grab frame {i+1}")
            break
        time.sleep(0.1)
    
    zed.close()
    print("ZED test complete")
    return True

def test_yolo_only():
    """Test YOLO without ZED camera"""
    print("Testing YOLO detection...")
    
    try:
        from ultralytics import YOLO
        print("YOLO imported successfully")
        
        model = YOLO('yolov8n.pt')
        print("YOLO model loaded")
        
        # Test with a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        print("Running YOLO on dummy image...")
        
        results = model(dummy_image, verbose=False)
        print(f"YOLO inference complete, found {len(results)} result objects")
        
        return True
    except Exception as e:
        print(f"YOLO test failed: {e}")
        return False

def test_opencv_display():
    """Test OpenCV display functionality"""
    print("Testing OpenCV display...")
    
    try:
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        print("Creating OpenCV window...")
        cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
        cv2.imshow("Test Window", test_image)
        
        print("Displaying test image for 3 seconds...")
        for i in range(30):  # 3 seconds at ~10fps
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("OpenCV display test complete")
        return True
    except Exception as e:
        print(f"OpenCV display test failed: {e}")
        return False

def test_combined_minimal():
    """Test ZED + YOLO + Display in minimal form"""
    print("Testing combined minimal system...")
    
    # Initialize ZED
    zed = sl.Camera()
    init_params = sl.InitParameters(
        camera_resolution=sl.RESOLUTION.HD720,
        depth_mode=sl.DEPTH_MODE.PERFORMANCE
    )
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED")
        return False
    
    # Initialize YOLO
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("YOLO loaded for combined test")
    except Exception as e:
        print(f"YOLO failed in combined test: {e}")
        zed.close()
        return False
    
    # Setup display
    cv2.namedWindow("Combined Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Combined Test", 640, 480)
    
    runtime = sl.RuntimeParameters()
    left = sl.Mat()
    
    print("Running combined test for 10 frames...")
    
    try:
        for frame_num in range(10):
            print(f"Processing frame {frame_num + 1}/10...")
            
            # Grab frame
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                print(f"Failed to grab frame {frame_num + 1}")
                continue
            
            # Get image
            zed.retrieve_image(left, sl.VIEW.LEFT)
            image_bgra = left.get_data()
            image_bgr = image_bgra[..., :3]
            
            print(f"Frame {frame_num + 1}: Running YOLO...")
            # Run YOLO (this is where it might freeze)
            results = model(image_bgr, verbose=False)
            print(f"Frame {frame_num + 1}: YOLO complete")
            
            # Simple visualization
            vis = image_bgr.copy()
            cv2.putText(vis, f"Frame {frame_num + 1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            print(f"Frame {frame_num + 1}: Displaying...")
            cv2.imshow("Combined Test", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Test stopped by user")
                break
            
            print(f"Frame {frame_num + 1}: Complete")
    
    except Exception as e:
        print(f"Combined test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        zed.close()
        cv2.destroyAllWindows()
    
    print("Combined test finished")
    return True

def main():
    print("=== ZED2 Tracking Debug Tool ===")
    print("This will test each component individually to find the freezing issue")
    print()
    
    tests = [
        ("Basic ZED Camera", test_basic_zed),
        ("YOLO Only", test_yolo_only), 
        ("OpenCV Display", test_opencv_display),
        ("Combined Minimal", test_combined_minimal)
    ]
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            print(f"{test_name}: {'PASSED' if success else 'FAILED'}")
        except KeyboardInterrupt:
            print(f"{test_name}: INTERRUPTED")
            break
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    main()