#!/usr/bin/env python3
"""
USB Camera Interface for Pan-Tilt Tracking Camera
Handles camera access and image capture
"""

import cv2
import time
import threading
import queue

class USBCamera:
    """USB Camera interface with thread-safe frame capture"""
    
    def __init__(self, camera_index=0, resolution=(1920, 1080), fps=30):
        """
        Initialize USB camera interface
        
        Args:
            camera_index: Camera device index
            resolution: Desired camera resolution
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        
        # OpenCV capture object
        self.cap = None
        
        # Frame capture thread
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=2)  # Only keep most recent frames
        self.running = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.current_fps = 0
        
        # Frame dimensions
        self.frame_width = 0
        self.frame_height = 0
    
    def open(self):
        """Open camera and initialize capture"""
        # Close existing capture if any
        if self.cap is not None:
            self.close()
        
        try:
            # Try to open the camera using direct OpenCV approach
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Set FPS if supported
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual camera settings (may differ from requested)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Verify camera is working by reading a test frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print("Camera opened but failed to read test frame")
                self.cap.release()
                self.cap = None
                return False
            
            print(f"Camera opened successfully:")
            print(f"  Index: {self.camera_index}")
            print(f"  Resolution: {self.frame_width}x{self.frame_height}")
            print(f"  FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            print(f"Error opening camera: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def start_capture(self):
        """Start continuous frame capture thread"""
        if self.cap is None or not self.cap.isOpened():
            print("Camera not opened")
            return False
        
        if self.running:
            print("Capture already running")
            return True
        
        # Clear any old frames
        while not self.frame_queue.empty():
            self.frame_queue.get()
        
        # Reset frame counter and FPS
        self.frame_count = 0
        self.start_time = time.time()
        
        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        return True
    
    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                print("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Update frame counter and FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:  # Update FPS every second
                self.current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
            
            # Add to queue, replacing oldest frame if full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Skip frame if queue is still full
    
    def get_frame(self):
        """Get the most recent camera frame (thread-safe)"""
        if not self.running or self.cap is None:
            return None
        
        # First try to get from queue (threaded capture)
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            # Fall back to direct capture if queue is empty
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    return frame
        
        return None
    
    def get_fps(self):
        """Get current FPS estimate"""
        return self.current_fps
    
    def close(self):
        """Close camera and release resources"""
        # Stop capture thread
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
        
        # Release OpenCV capture
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("Camera closed")

# Test function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='USB Camera Test')
    parser.add_argument('--index', '-i', type=int, default=0, help='Camera index')
    parser.add_argument('--width', '-W', type=int, default=640, help='Capture width')
    parser.add_argument('--height', '-H', type=int, default=480, help='Capture height')
    parser.add_argument('--fps', '-f', type=int, default=30, help='Target FPS')
    
    args = parser.parse_args()
    
    camera = USBCamera(
        camera_index=args.index,
        resolution=(args.width, args.height),
        fps=args.fps
    )
    
    if not camera.open():
        print("Failed to open camera")
        exit(1)
    
    if not camera.start_capture():
        print("Failed to start capture")
        camera.close()
        exit(1)
    
    print("Camera test running. Press 'q' to quit, 's' to save frame")
    
    try:
        last_fps_print = time.time()
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("No frame received")
                time.sleep(0.1)
                continue
            
            # Show FPS on frame
            fps = camera.get_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("USB Camera Test", frame)
            
            # Print FPS every second
            now = time.time()
            if now - last_fps_print > 1.0:
                print(f"FPS: {fps:.1f}")
                last_fps_print = now
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame to {filename}")
    
    finally:
        camera.close()
        cv2.destroyAllWindows()