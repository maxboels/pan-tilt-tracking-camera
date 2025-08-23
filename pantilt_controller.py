#!/usr/bin/env python3
"""
Pan-Tilt Servo Controller using PCA9685
Designed for Jetson Nano deployment
"""

import time
import math
from typing import Tuple, Optional
import threading
import queue

try:
    from adafruit_pca9685 import PCA9685
    import board
    import busio
    PCA9685_AVAILABLE = True
except ImportError:
    PCA9685_AVAILABLE = False
    print("PCA9685 not available - running in simulation mode")

class PanTiltController:
    def __init__(self, 
                 pan_channel: int = 0,
                 tilt_channel: int = 1,
                 pan_min_pulse: int = 500,
                 pan_max_pulse: int = 2500,
                 tilt_min_pulse: int = 500,
                 tilt_max_pulse: int = 2500,
                 pan_range: Tuple[float, float] = (-90, 90),
                 tilt_range: Tuple[float, float] = (-30, 30),
                 smooth_movement: bool = True,
                 movement_speed: float = 2.0):  # degrees per update
        
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel
        self.pan_min_pulse = pan_min_pulse
        self.pan_max_pulse = pan_max_pulse
        self.tilt_min_pulse = tilt_min_pulse
        self.tilt_max_pulse = tilt_max_pulse
        self.pan_range = pan_range
        self.tilt_range = tilt_range
        self.smooth_movement = smooth_movement
        self.movement_speed = movement_speed
        
        # Current position
        self.current_pan = 0.0
        self.current_tilt = 0.0
        
        # Initialize PCA9685
        self.pca = None
        if PCA9685_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.pca = PCA9685(i2c)
                self.pca.frequency = 50  # 50Hz for servos
                print("PCA9685 initialized successfully")
            except Exception as e:
                print(f"Failed to initialize PCA9685: {e}")
                self.pca = None
        
        # Movement thread
        self.target_pan = 0.0
        self.target_tilt = 0.0
        self.movement_queue = queue.Queue()
        self.running = True
        
        if self.smooth_movement:
            self.movement_thread = threading.Thread(target=self._movement_loop, daemon=True)
            self.movement_thread.start()
        
        # Move to center position
        self.move_to(0, 0, immediate=True)
        print(f"Pan-tilt controller initialized (channels: pan={pan_channel}, tilt={tilt_channel})")

    def _angle_to_pulse(self, angle: float, min_pulse: int, max_pulse: int, angle_range: Tuple[float, float]) -> int:
        """Convert angle to servo pulse width"""
        min_angle, max_angle = angle_range
        angle = max(min_angle, min(max_angle, angle))  # Clamp angle
        
        # Map angle to pulse width
        angle_ratio = (angle - min_angle) / (max_angle - min_angle)
        pulse_width = int(min_pulse + angle_ratio * (max_pulse - min_pulse))
        
        return pulse_width

    def _set_servo_angle(self, channel: int, angle: float, min_pulse: int, max_pulse: int, angle_range: Tuple[float, float]):
        """Set servo to specific angle"""
        pulse_width = self._angle_to_pulse(angle, min_pulse, max_pulse, angle_range)
        
        if self.pca is not None:
            # Convert pulse width (microseconds) to duty cycle for PCA9685
            # PCA9685 has 4096 steps per period (20ms)
            duty_cycle = int((pulse_width / 20000.0) * 4096)
            self.pca.channels[channel].duty_cycle = duty_cycle
        else:
            # Simulation mode
            print(f"SERVO SIM - Channel {channel}: {angle:.1f}° (pulse: {pulse_width}μs)")

    def _movement_loop(self):
        """Smooth movement thread"""
        while self.running:
            try:
                # Check for new target
                try:
                    new_target = self.movement_queue.get_nowait()
                    self.target_pan, self.target_tilt = new_target
                except queue.Empty:
                    pass
                
                # Calculate movement step
                pan_diff = self.target_pan - self.current_pan
                tilt_diff = self.target_tilt - self.current_tilt
                
                # Move towards target
                if abs(pan_diff) > 0.1:  # Small deadband
                    step = min(self.movement_speed, abs(pan_diff))
                    self.current_pan += step if pan_diff > 0 else -step
                    self._set_servo_angle(self.pan_channel, self.current_pan, 
                                         self.pan_min_pulse, self.pan_max_pulse, self.pan_range)
                
                if abs(tilt_diff) > 0.1:
                    step = min(self.movement_speed, abs(tilt_diff))
                    self.current_tilt += step if tilt_diff > 0 else -step
                    self._set_servo_angle(self.tilt_channel, self.current_tilt,
                                         self.tilt_min_pulse, self.tilt_max_pulse, self.tilt_range)
                
                time.sleep(0.02)  # 50Hz update rate
                
            except Exception as e:
                print(f"Movement loop error: {e}")
                time.sleep(0.1)

    def move_to(self, pan_angle: float, tilt_angle: float, immediate: bool = False):
        """Move to specified pan/tilt angles"""
        # Clamp angles to valid ranges
        pan_angle = max(self.pan_range[0], min(self.pan_range[1], pan_angle))
        tilt_angle = max(self.tilt_range[0], min(self.tilt_range[1], tilt_angle))
        
        if immediate or not self.smooth_movement:
            # Immediate movement
            self.current_pan = pan_angle
            self.current_tilt = tilt_angle
            self.target_pan = pan_angle
            self.target_tilt = tilt_angle
            
            self._set_servo_angle(self.pan_channel, pan_angle, 
                                 self.pan_min_pulse, self.pan_max_pulse, self.pan_range)
            self._set_servo_angle(self.tilt_channel, tilt_angle,
                                 self.tilt_min_pulse, self.tilt_max_pulse, self.tilt_range)
        else:
            # Smooth movement via queue
            if not self.movement_queue.full():
                try:
                    self.movement_queue.put_nowait((pan_angle, tilt_angle))
                except queue.Full:
                    pass  # Skip if queue is full

    def get_position(self) -> Tuple[float, float]:
        """Get current pan/tilt position"""
        return self.current_pan, self.current_tilt

    def center(self):
        """Move to center position"""
        self.move_to(0, 0)

    def scan_pattern(self, scan_range: float = 45.0, steps: int = 5, tilt_angle: float = 0):
        """Execute a scanning pattern"""
        positions = []
        for i in range(steps):
            pan = -scan_range + (2 * scan_range * i / (steps - 1))
            positions.append((pan, tilt_angle))
        
        return positions

    def shutdown(self):
        """Clean shutdown"""
        self.running = False
        if hasattr(self, 'movement_thread'):
            self.movement_thread.join(timeout=1)
        
        # Center servos
        self.move_to(0, 0, immediate=True)
        time.sleep(0.5)
        
        # Disable servos
        if self.pca is not None:
            self.pca.channels[self.pan_channel].duty_cycle = 0
            self.pca.channels[self.tilt_channel].duty_cycle = 0
            self.pca.deinit()
        
        print("Pan-tilt controller shutdown")


class PanTiltTracker:
    """High-level tracker that combines detection data with servo control"""
    
    def __init__(self, controller: PanTiltController):
        self.controller = controller
        self.tracking_active = False
        self.last_detection_time = 0
        self.detection_timeout = 2.0  # seconds
        
        # Tracking parameters
        self.max_movement_per_frame = 5.0  # degrees
        self.smoothing_factor = 0.7
        
    def track_target(self, pan_angle: float, tilt_angle: float):
        """Track a target at the specified angles"""
        current_time = time.time()
        
        # Limit movement speed
        current_pan, current_tilt = self.controller.get_position()
        
        pan_diff = pan_angle - current_pan
        tilt_diff = tilt_angle - current_tilt
        
        # Limit movement per frame
        if abs(pan_diff) > self.max_movement_per_frame:
            pan_diff = self.max_movement_per_frame * (1 if pan_diff > 0 else -1)
        if abs(tilt_diff) > self.max_movement_per_frame:
            tilt_diff = self.max_movement_per_frame * (1 if tilt_diff > 0 else -1)
        
        # Apply smoothing
        new_pan = current_pan + pan_diff * self.smoothing_factor
        new_tilt = current_tilt + tilt_diff * self.smoothing_factor
        
        self.controller.move_to(new_pan, new_tilt)
        self.last_detection_time = current_time
        self.tracking_active = True
    
    def check_timeout(self):
        """Check if tracking should timeout due to no recent detections"""
        if self.tracking_active:
            if time.time() - self.last_detection_time > self.detection_timeout:
                print("Tracking timeout - returning to center")
                self.controller.center()
                self.tracking_active = False


def test_pantilt():
    """Test the pan-tilt controller"""
    controller = PanTiltController(smooth_movement=True, movement_speed=3.0)
    
    try:
        print("Testing pan-tilt controller...")
        
        # Test positions
        test_positions = [
            (0, 0),      # Center
            (45, 15),    # Right up
            (-45, 15),   # Left up  
            (-45, -15),  # Left down
            (45, -15),   # Right down
            (0, 0),      # Back to center
        ]
        
        for pan, tilt in test_positions:
            print(f"Moving to pan={pan}°, tilt={tilt}°")
            controller.move_to(pan, tilt)
            time.sleep(2)
            
            actual_pan, actual_tilt = controller.get_position()
            print(f"Current position: pan={actual_pan:.1f}°, tilt={actual_tilt:.1f}°")
        
        # Test scanning pattern
        print("Testing scan pattern...")
        scan_positions = controller.scan_pattern(scan_range=60, steps=7, tilt_angle=10)
        for pan, tilt in scan_positions:
            controller.move_to(pan, tilt)
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Test interrupted")
    
    finally:
        controller.shutdown()


if __name__ == "__main__":
    test_pantilt()