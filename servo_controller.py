#!/usr/bin/env python3
"""
Arduino Servo Controller for Pan-Tilt Tracking Camera
Communicates with Arduino running servo bridge firmware
"""

import serial
import time
import threading
from collections import deque

class ArduinoServoController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, pan_channel=0, tilt_channel=1):
        """
        Initialize Arduino servo controller
        
        Args:
            port: Serial port for Arduino connection
            baudrate: Serial communication speed
            pan_channel: PCA9685 channel for pan servo
            tilt_channel: PCA9685 channel for tilt servo
        """
        self.port = port
        self.baudrate = baudrate
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel
        self.serial_connection = None
        
        # Servo position limits (in degrees)
        self.pan_min = -90
        self.pan_max = 90
        self.tilt_min = -45
        self.tilt_max = 45
        
        # Current servo positions
        self.current_pan = 0
        self.current_tilt = 0
        
        # Movement parameters
        self.max_speed = 5.0  # degrees per update
        self.error_to_angle_ratio = 0.1  # Convert pixel error to degrees
        
        # Command queue for thread-safe operation
        self.command_queue = deque()
        self.queue_lock = threading.Lock()
        
        self.connected = False
        
    def connect(self):
        """Connect to Arduino"""
        # Try multiple common ports
        possible_ports = [self.port, '/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyUSB1']
        
        for port in possible_ports:
            try:
                print(f"Trying to connect to {port}...")
                self.serial_connection = serial.Serial(
                    port, 
                    self.baudrate, 
                    timeout=1.0,
                    write_timeout=1.0
                )
                time.sleep(2)  # Wait for Arduino to reset
                
                # Test connection
                self.serial_connection.write(b"CENTER\n")
                response = self.serial_connection.readline().decode().strip()
                
                if "OK" in response or "Ready" in response:
                    self.connected = True
                    self.port = port  # Update port to the one that worked
                    print(f"Arduino servo controller connected on {port}")
                    print(f"Arduino response: {response}")
                    return True
                else:
                    print(f"No valid response from {port}. Response: {response}")
                    self.serial_connection.close()
                    
            except Exception as e:
                print(f"Failed to connect to {port}: {e}")
                if self.serial_connection:
                    try:
                        self.serial_connection.close()
                    except:
                        pass
                continue
        
        print("Failed to connect to Arduino on any port")
        return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.serial_connection and self.serial_connection.is_open:
            # Center servos before disconnecting
            self.send_command("CENTER")
            time.sleep(0.5)
            self.serial_connection.close()
            self.connected = False
            print("Arduino servo controller disconnected")
    
    def send_command(self, command):
        """Send command to Arduino"""
        if not self.connected or not self.serial_connection:
            return False
        
        try:
            self.serial_connection.write(f"{command}\n".encode())
            response = self.serial_connection.readline().decode().strip()
            return "OK" in response
        except Exception as e:
            print(f"Error sending command '{command}': {e}")
            return False
    
    def move_servo(self, channel, angle):
        """Move servo to specific angle"""
        # Clamp angle to valid range
        if channel == self.pan_channel:
            angle = max(self.pan_min, min(self.pan_max, angle))
        elif channel == self.tilt_channel:
            angle = max(self.tilt_min, min(self.tilt_max, angle))
        
        command = f"SERVO,{channel},{angle:.1f}"
        success = self.send_command(command)
        
        if success:
            if channel == self.pan_channel:
                self.current_pan = angle
            elif channel == self.tilt_channel:
                self.current_tilt = angle
        
        return success
    
    def calculate_pan_command(self, pan_error):
        """Convert pan error to servo angle adjustment"""
        # Convert pixel error to angle adjustment
        angle_adjustment = pan_error * self.error_to_angle_ratio
        
        # Limit movement speed
        angle_adjustment = max(-self.max_speed, min(self.max_speed, angle_adjustment))
        
        # Calculate new position
        new_angle = self.current_pan + angle_adjustment
        new_angle = max(self.pan_min, min(self.pan_max, new_angle))
        
        return new_angle
    
    def calculate_tilt_command(self, tilt_error):
        """Convert tilt error to servo angle adjustment"""
        # Convert pixel error to angle adjustment (negative because camera Y is inverted)
        angle_adjustment = -tilt_error * self.error_to_angle_ratio
        
        # Limit movement speed
        angle_adjustment = max(-self.max_speed, min(self.max_speed, angle_adjustment))
        
        # Calculate new position
        new_angle = self.current_tilt + angle_adjustment
        new_angle = max(self.tilt_min, min(self.tilt_max, new_angle))
        
        return new_angle
    
    def move_servos(self, pan_angle, tilt_angle):
        """Move both servos to specified angles"""
        pan_success = self.move_servo(self.pan_channel, pan_angle)
        tilt_success = self.move_servo(self.tilt_channel, tilt_angle)
        
        if pan_success and tilt_success:
            print(f"Servos moved - Pan: {pan_angle:.1f}°, Tilt: {tilt_angle:.1f}°")
        
        return pan_success and tilt_success
    
    def center_servos(self):
        """Center both servos"""
        return self.send_command("CENTER")
    
    def disable_servos(self):
        """Disable servo outputs"""
        return self.send_command("OFF")
    
    def get_status(self):
        """Get current servo status"""
        return {
            'connected': self.connected,
            'pan_angle': self.current_pan,
            'tilt_angle': self.current_tilt,
            'port': self.port
        }

# Test function
if __name__ == "__main__":
    controller = ArduinoServoController()
    
    if controller.connect():
        print("Testing servo movements...")
        
        # Center servos
        controller.center_servos()
        time.sleep(1)
        
        # Test pan movement
        controller.move_servo(0, 30)  # Pan right
        time.sleep(1)
        controller.move_servo(0, -30)  # Pan left
        time.sleep(1)
        
        # Test tilt movement
        controller.move_servo(1, 20)  # Tilt up
        time.sleep(1)
        controller.move_servo(1, -20)  # Tilt down
        time.sleep(1)
        
        # Return to center
        controller.center_servos()
        
        controller.disconnect()
    else:
        print("Failed to connect to Arduino")