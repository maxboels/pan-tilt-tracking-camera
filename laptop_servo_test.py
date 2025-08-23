#!/usr/bin/env python3
"""
Laptop Servo Testing via Arduino Bridge
Allows testing pan-tilt servos from laptop through USB-connected Arduino
"""

import serial
import time
import threading
from typing import Optional, Tuple

class ArduinoServoController:
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        """
        Initialize Arduino servo controller
        
        Args:
            port: Serial port (Linux: /dev/ttyUSB0, Windows: COM3, Mac: /dev/cu.usbserial-*)
            baudrate: Serial communication speed
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connected = False
        
        # Current positions
        self.current_pan = 0.0
        self.current_tilt = 0.0
        
        self.connect()

    def connect(self):
        """Connect to Arduino"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            
            # Test connection
            response = self.send_command("CENTER")
            if "OK" in response:
                self.connected = True
                print(f"Connected to Arduino servo bridge on {self.port}")
            else:
                print(f"Arduino not responding properly: {response}")
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            print("Available ports:")
            self.list_serial_ports()

    def list_serial_ports(self):
        """List available serial ports"""
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            for port in ports:
                print(f"  {port.device} - {port.description}")
        except ImportError:
            print("  Install pyserial to list ports: pip install pyserial")

    def send_command(self, command: str) -> str:
        """Send command to Arduino and get response"""
        if not self.connected or not self.serial_conn:
            return "ERROR: Not connected"
        
        try:
            self.serial_conn.write((command + '\n').encode())
            time.sleep(0.1)
            
            response = ""
            while self.serial_conn.in_waiting > 0:
                response += self.serial_conn.read().decode()
            
            return response.strip()
        except Exception as e:
            return f"ERROR: {e}"

    def move_servo(self, channel: int, angle: float) -> bool:
        """Move servo to specified angle"""
        angle = max(-90, min(90, angle))  # Clamp to valid range
        
        command = f"SERVO,{channel},{angle}"
        response = self.send_command(command)
        
        if "OK" in response:
            if channel == 0:
                self.current_pan = angle
            elif channel == 1:
                self.current_tilt = angle
            return True
        else:
            print(f"Servo command failed: {response}")
            return False

    def move_to(self, pan_angle: float, tilt_angle: float):
        """Move to pan/tilt position"""
        success_pan = self.move_servo(0, pan_angle)
        success_tilt = self.move_servo(1, tilt_angle)
        return success_pan and success_tilt

    def center(self):
        """Center both servos"""
        response = self.send_command("CENTER")
        if "OK" in response:
            self.current_pan = 0.0
            self.current_tilt = 0.0
            return True
        return False

    def disable_servos(self):
        """Turn off servo power"""
        response = self.send_command("OFF")
        return "OK" in response

    def get_position(self) -> Tuple[float, float]:
        """Get current pan/tilt position"""
        return self.current_pan, self.current_tilt

    def close(self):
        """Close connection"""
        if self.serial_conn:
            self.disable_servos()
            self.serial_conn.close()
            self.connected = False


class LaptopPanTiltTest:
    """Test interface for pan-tilt system on laptop"""
    
    def __init__(self, arduino_port: str = '/dev/ttyUSB0'):
        self.controller = ArduinoServoController(arduino_port)
        
    def run_interactive_test(self):
        """Interactive servo testing"""
        if not self.controller.connected:
            print("Arduino not connected. Cannot run test.")
            return
        
        print("\n=== Pan-Tilt Servo Test ===")
        print("Commands:")
        print("  move <pan> <tilt>  - Move to position (degrees)")
        print("  center             - Center servos")
        print("  scan              - Run scan pattern")
        print("  off               - Turn off servos")
        print("  quit              - Exit")
        print()
        
        try:
            while True:
                command = input("servo> ").strip().lower().split()
                
                if not command:
                    continue
                
                if command[0] == 'quit' or command[0] == 'q':
                    break
                elif command[0] == 'move' and len(command) == 3:
                    try:
                        pan = float(command[1])
                        tilt = float(command[2])
                        print(f"Moving to pan={pan}°, tilt={tilt}°...")
                        success = self.controller.move_to(pan, tilt)
                        if success:
                            print("Move completed")
                        else:
                            print("Move failed")
                    except ValueError:
                        print("Invalid angles. Use: move <pan> <tilt>")
                elif command[0] == 'center' or command[0] == 'c':
                    print("Centering servos...")
                    self.controller.center()
                elif command[0] == 'scan':
                    print("Running scan pattern...")
                    self.run_scan_test()
                elif command[0] == 'off':
                    print("Turning off servos...")
                    self.controller.disable_servos()
                else:
                    print("Unknown command")
                
                # Show current position
                pan, tilt = self.controller.get_position()
                print(f"Current position: pan={pan:.1f}°, tilt={tilt:.1f}°")
        
        except KeyboardInterrupt:
            print("\nExiting...")
        
        finally:
            self.controller.close()

    def run_scan_test(self):
        """Test scanning pattern"""
        positions = [
            (0, 0),      # Center
            (45, 15),    # Right up
            (-45, 15),   # Left up
            (-45, -15),  # Left down
            (45, -15),   # Right down
            (0, 0),      # Back to center
        ]
        
        for pan, tilt in positions:
            print(f"  Moving to {pan}°, {tilt}°...")
            self.controller.move_to(pan, tilt)
            time.sleep(1.5)

    def test_tracking_simulation(self, tracking_data):
        """Test tracking with simulated detection data"""
        print("Testing tracking simulation...")
        
        for frame_data in tracking_data:
            pan_angle = frame_data.get('pan_angle', 0)
            tilt_angle = frame_data.get('tilt_angle', 0)
            
            print(f"Tracking target at pan={pan_angle:.1f}°, tilt={tilt_angle:.1f}°")
            self.controller.move_to(pan_angle, tilt_angle)
            time.sleep(0.1)  # Simulate frame rate


def detect_arduino_port():
    """Try to detect Arduino port automatically"""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        
        # Look for Arduino-like devices
        arduino_keywords = ['arduino', 'usb', 'serial', 'ch340', 'cp210x', 'ftdi']
        
        for port in ports:
            description = port.description.lower()
            if any(keyword in description for keyword in arduino_keywords):
                print(f"Found potential Arduino at: {port.device}")
                return port.device
        
        # If no Arduino found, return first available port
        if ports:
            return ports[0].device
    except ImportError:
        pass
    
    # Default guesses
    import os
    if os.name == 'nt':  # Windows
        return 'COM3'
    else:  # Linux/Mac
        return '/dev/ttyUSB0'


def main():
    print("=== Laptop Servo Testing ===")
    
    # Try to detect Arduino port
    port = detect_arduino_port()
    print(f"Using port: {port}")
    
    # Run test
    tester = LaptopPanTiltTest(port)
    tester.run_interactive_test()


if __name__ == "__main__":
    main()