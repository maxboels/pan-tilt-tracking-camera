#!/usr/bin/env python3
"""
Arduino Servo Communication Debug Tool
Diagnoses serial communication issues and tests servo control
"""

import serial
import time
import sys
from typing import Optional

def find_arduino_port():
    """Try to find Arduino port automatically"""
    import serial.tools.list_ports
    
    ports = serial.tools.list_ports.comports()
    arduino_ports = []
    
    for port in ports:
        # Look for Arduino-like devices
        desc = port.description.lower()
        if any(keyword in desc for keyword in ['arduino', 'ch340', 'cp210', 'usb', 'acm']):
            arduino_ports.append(port.device)
            print(f"Found potential Arduino: {port.device} - {port.description}")
    
    return arduino_ports

def test_serial_connection(port: str, baudrate: int = 115200) -> Optional[serial.Serial]:
    """Test and establish serial connection"""
    print(f"\nTesting connection to {port} at {baudrate} baud...")
    
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        print(f"Serial port opened: {ser.is_open}")
        
        # Wait for Arduino to reset and initialize
        print("Waiting for Arduino reset...")
        time.sleep(3)
        
        # Clear any existing data
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Send test command
        print("Sending CENTER command...")
        ser.write(b'CENTER\n')
        ser.flush()
        
        # Wait for response
        response = ""
        start_time = time.time()
        while time.time() - start_time < 3:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                response += data
                if 'OK' in response or 'ERROR' in response:
                    break
            time.sleep(0.1)
        
        print(f"Arduino response: '{response.strip()}'")
        
        if 'OK' in response:
            print("✓ Arduino communication working!")
            return ser
        else:
            print("✗ Arduino not responding correctly")
            ser.close()
            return None
            
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None

def interactive_servo_test(ser: serial.Serial):
    """Interactive servo testing"""
    print("\n=== Interactive Servo Test ===")
    print("Commands:")
    print("  pan <angle>     - Move pan servo (-90 to 90)")
    print("  tilt <angle>    - Move tilt servo (-90 to 90)")  
    print("  center          - Center both servos")
    print("  off             - Turn off servos")
    print("  raw <command>   - Send raw command to Arduino")
    print("  quit            - Exit")
    print()
    
    while True:
        try:
            cmd = input("servo> ").strip()
            if not cmd:
                continue
                
            if cmd.lower() in ['quit', 'q', 'exit']:
                break
                
            # Parse commands
            parts = cmd.split()
            arduino_cmd = ""
            
            if parts[0].lower() == 'pan' and len(parts) == 2:
                try:
                    angle = float(parts[1])
                    arduino_cmd = f"SERVO,0,{angle}"
                except ValueError:
                    print("Invalid angle")
                    continue
                    
            elif parts[0].lower() == 'tilt' and len(parts) == 2:
                try:
                    angle = float(parts[1])
                    arduino_cmd = f"SERVO,1,{angle}"
                except ValueError:
                    print("Invalid angle")
                    continue
                    
            elif parts[0].lower() == 'center':
                arduino_cmd = "CENTER"
                
            elif parts[0].lower() == 'off':
                arduino_cmd = "OFF"
                
            elif parts[0].lower() == 'raw' and len(parts) > 1:
                arduino_cmd = ' '.join(parts[1:])
                
            else:
                print("Unknown command")
                continue
            
            # Send command to Arduino
            print(f"Sending: {arduino_cmd}")
            ser.reset_input_buffer()  # Clear old data
            ser.write((arduino_cmd + '\n').encode())
            ser.flush()
            
            # Get response
            response = ""
            start_time = time.time()
            while time.time() - start_time < 2:
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    response += data
                    if '\n' in data:  # Complete line received
                        break
                time.sleep(0.05)
            
            print(f"Arduino: {response.strip()}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def timing_test(ser: serial.Serial):
    """Test command timing and response consistency"""
    print("\n=== Timing Test ===")
    print("Testing command response times...")
    
    test_commands = [
        "CENTER",
        "SERVO,0,45",
        "SERVO,1,-30", 
        "SERVO,0,0",
        "SERVO,1,0",
        "OFF"
    ]
    
    timings = []
    
    for cmd in test_commands:
        print(f"Testing: {cmd}")
        
        # Clear buffers
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Send command and measure time
        start_time = time.time()
        ser.write((cmd + '\n').encode())
        ser.flush()
        
        # Wait for response
        response = ""
        while time.time() - start_time < 3:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                response += data
                if '\n' in data:
                    break
            time.sleep(0.01)
        
        elapsed = time.time() - start_time
        timings.append(elapsed)
        
        success = "✓" if "OK" in response else "✗"
        print(f"  {success} Response in {elapsed*1000:.1f}ms: {response.strip()}")
        
        time.sleep(0.5)  # Pause between commands
    
    print(f"\nTiming summary:")
    print(f"  Average response time: {sum(timings)/len(timings)*1000:.1f}ms")
    print(f"  Min/Max: {min(timings)*1000:.1f}ms / {max(timings)*1000:.1f}ms")

def stress_test(ser: serial.Serial):
    """Stress test with rapid commands"""
    print("\n=== Stress Test ===")
    print("Sending rapid servo commands...")
    
    angles = [-45, -30, -15, 0, 15, 30, 45, 30, 15, 0, -15, -30]
    success_count = 0
    
    for i, angle in enumerate(angles):
        cmd = f"SERVO,0,{angle}"
        print(f"Command {i+1}/{len(angles)}: {cmd}")
        
        ser.reset_input_buffer()
        ser.write((cmd + '\n').encode())
        ser.flush()
        
        # Quick response check
        response = ""
        start_time = time.time()
        while time.time() - start_time < 1:
            if ser.in_waiting > 0:
                response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                if '\n' in response:
                    break
            time.sleep(0.01)
        
        if "OK" in response:
            success_count += 1
            print(f"  ✓ Success")
        else:
            print(f"  ✗ Failed: {response.strip()}")
        
        time.sleep(0.2)  # Small delay between commands
    
    print(f"\nStress test results: {success_count}/{len(angles)} commands successful")

def main():
    print("=== Arduino Servo Debug Tool ===")
    
    # Find Arduino ports
    ports = find_arduino_port()
    
    if not ports:
        print("No Arduino-like devices found!")
        print("Available ports:")
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            print(f"  {port.device} - {port.description}")
        sys.exit(1)
    
    # Try to connect
    ser = None
    for port in ports:
        ser = test_serial_connection(port)
        if ser:
            break
    
    if not ser:
        print("\nFailed to connect to any Arduino!")
        print("Check:")
        print("1. Arduino is connected and powered")
        print("2. Correct drivers installed") 
        print("3. Arduino sketch uploaded")
        print("4. No other programs using the port")
        sys.exit(1)
    
    try:
        # Run tests
        print("\nSelect test mode:")
        print("1. Interactive servo control")
        print("2. Timing test")
        print("3. Stress test") 
        print("4. All tests")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            interactive_servo_test(ser)
        elif choice == '2':
            timing_test(ser)
        elif choice == '3':
            stress_test(ser)
        elif choice == '4':
            timing_test(ser)
            stress_test(ser)
            print("\nStarting interactive mode...")
            interactive_servo_test(ser)
        else:
            print("Invalid choice")
            
    finally:
        if ser:
            # Center servos before closing
            try:
                ser.write(b'CENTER\n')
                ser.flush()
                time.sleep(1)
                ser.write(b'OFF\n')
                ser.flush()
                time.sleep(0.5)
            except:
                pass
            ser.close()
            print("Serial connection closed")

if __name__ == "__main__":
    main()