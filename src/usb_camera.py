#!/usr/bin/env python3
"""
USB Camera module for Pan-Tilt Tracking Camera
Handles USB camera capture and configuration
"""

import cv2
import threading
import time
from collections import deque

class USBCamera:
    def __init__(self, camera_index=0, resolution=(1920, 1080), fps=30):
        """
        Initialize USB camera
        
        Args:
            camera_index: Camera device index (usually 0 for first camera)
            resolution: Tuple of (width, height)
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        
        # Threading for frame capture
        self.capture_thread = None
        self.frame_queue = deque(maxlen=2)  # Keep only latest frames
        self.capture_running = False
        self.frame_lock = threading.Lock()
        
        # Camera status
        self.is_opened = False
        
    def open(self):
        """Open camera connection"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera opened successfully:")
            print(f"  Index: {self.camera_index}")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  FPS: {actual_fps}")
            
            self.is_opened = True
            return True
            
        except Exception as e:
            print(f"Error opening camera: {e}")
            return False
    
    def start_capture(self):
        """Start threaded frame capture"""
        if not self.is_opened or not self.cap:
            return False
        
        self.capture_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        return True
    
    def _capture_loop(self):
        """Internal capture loop running in separate thread"""
        while self.capture_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if ret:
                with self.frame_lock:
                    self.frame_queue.append(frame.copy())
            else:
                print("Failed to read frame from camera")
                break
            
            time.sleep(1.0 / self.fps)  # Control capture rate
    
    def get_frame(self):
        """Get the latest frame from the camera"""
        with self.frame_lock:
            if self.frame_queue:
                return self.frame_queue[-1]  # Return latest frame
            return None
    
    def close(self):
        """Close camera connection"""
        self.capture_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_opened = False
        print("Camera closed")
    
    def get_properties(self):
        """Get current camera properties"""
        if not self.cap:
            return None
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
        }
    
    def set_property(self, property_id, value):
        """Set camera property"""
        if self.cap:
            return self.cap.set(property_id, value)
        return False

# Test function
if __name__ == "__main__":
    camera = USBCamera()
    
    if camera.open():
        if camera.start_capture():
            print("Camera capture started. Press 'q' to quit.")
            
            while True:
                frame = camera.get_frame()
                if frame is not None:
                    cv2.imshow('USB Camera Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            cv2.destroyAllWindows()
        
        camera.close()
    else:
        print("Failed to open camera")