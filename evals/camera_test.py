#!/usr/bin/env python3
"""
Camera Test Utility
Tests camera connection and frame capture with multiple settings
"""

import cv2
import time
import argparse
import sys
import os

# Add parent directory to path to import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.usb_camera import USBCamera

def test_direct_opencv(indices=[0, 1, 2], resolutions=[(1920, 1080), (1280, 720), (640, 480)]):
    """Test direct OpenCV camera capture with various settings"""
    print("\n=== Testing Direct OpenCV Camera Capture ===")
    
    for idx in indices:
        for width, height in resolutions:
            print(f"Trying camera index {idx} with resolution {width}x{height}...")
            
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"  Failed to open camera with index {idx}")
                continue
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Try to read a frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"  ✅ SUCCESS: Captured frame with size {actual_width}x{actual_height}")
                print(f"  Frame shape: {frame.shape}")
                
                # Display the frame
                cv2.imshow(f"Camera {idx} - {actual_width}x{actual_height}", frame)
                cv2.waitKey(1000)  # Show for 1 second
                
                # Try to capture multiple frames to test stability
                print(f"  Testing continuous capture for 3 seconds...")
                start_time = time.time()
                frame_count = 0
                
                while time.time() - start_time < 3:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_count += 1
                        cv2.imshow(f"Camera {idx} - {actual_width}x{actual_height}", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("  ❌ Frame capture failed during continuous test")
                        break
                
                fps = frame_count / 3
                print(f"  Captured {frame_count} frames in 3 seconds ({fps:.1f} FPS)")
                
                cv2.destroyAllWindows()
            else:
                print(f"  ❌ Failed to capture frame")
            
            cap.release()

def test_usbcamera(indices=[0, 1, 2], resolutions=[(1920, 1080), (1280, 720), (640, 480)]):
    """Test USBCamera class with various settings"""
    print("\n=== Testing USBCamera Class ===")
    
    for idx in indices:
        for width, height in resolutions:
            print(f"Trying USBCamera with index {idx}, resolution {width}x{height}...")
            
            camera = USBCamera(
                camera_index=idx,
                resolution=(width, height),
                fps=30
            )
            
            if camera.open():
                print(f"  Camera opened successfully")
                
                if camera.start_capture():
                    print(f"  Capture started successfully")
                    
                    # Try to read a frame
                    frame = camera.get_frame()
                    
                    if frame is not None:
                        print(f"  ✅ SUCCESS: Captured frame with size {frame.shape[1]}x{frame.shape[0]}")
                        
                        # Display the frame
                        cv2.imshow(f"USBCamera {idx} - {width}x{height}", frame)
                        cv2.waitKey(1000)  # Show for 1 second
                        
                        # Try to capture multiple frames
                        print(f"  Testing continuous capture for 3 seconds...")
                        start_time = time.time()
                        frame_count = 0
                        
                        while time.time() - start_time < 3:
                            frame = camera.get_frame()
                            if frame is not None:
                                frame_count += 1
                                cv2.imshow(f"USBCamera {idx} - {width}x{height}", frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            else:
                                print("  ❌ Frame capture failed during continuous test")
                                break
                        
                        fps = frame_count / 3
                        print(f"  Captured {frame_count} frames in 3 seconds ({fps:.1f} FPS)")
                        
                        cv2.destroyAllWindows()
                    else:
                        print(f"  ❌ Failed to capture frame")
                else:
                    print(f"  ❌ Failed to start capture")
                
                camera.close()
            else:
                print(f"  ❌ Failed to open camera")

def main():
    parser = argparse.ArgumentParser(description='Camera Test Utility')
    parser.add_argument('--index', '-i', type=int, default=None,
                        help='Test specific camera index')
    parser.add_argument('--resolution', '-r', type=str, default=None,
                        help='Test specific resolution (WxH)')
    parser.add_argument('--opencv-only', action='store_true',
                        help='Test only with OpenCV, not USBCamera class')
    parser.add_argument('--class-only', action='store_true',
                        help='Test only with USBCamera class, not direct OpenCV')
    
    args = parser.parse_args()
    
    # Prepare test parameters
    indices = [args.index] if args.index is not None else [0, 1, 2]
    
    resolutions = []
    if args.resolution:
        try:
            w, h = map(int, args.resolution.split('x'))
            resolutions = [(w, h)]
        except:
            print(f"Invalid resolution format: {args.resolution}")
            resolutions = [(1920, 1080), (1280, 720), (640, 480)]
    else:
        resolutions = [(1920, 1080), (1280, 720), (640, 480)]
    
    print("Camera Test Utility")
    print("This utility will test camera connections with various settings")
    print(f"Testing camera indices: {indices}")
    print(f"Testing resolutions: {resolutions}")
    
    # Run tests
    if not args.class_only:
        test_direct_opencv(indices, resolutions)
    
    if not args.opencv_only:
        test_usbcamera(indices, resolutions)
    
    print("\nCamera testing complete")

if __name__ == "__main__":
    main()
