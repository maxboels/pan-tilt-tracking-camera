#!/usr/bin/env python3
"""
Find and test Arduino ports
"""

import serial
import serial.tools.list_ports
import time

def find_all_ports():
    """List all available serial ports"""
    print("=== All Available Serial Ports ===")
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("No serial ports found")
        return []
    
    for port in ports:
        print(f"Port: {port.device}")
        print(f"  Description: {port.description}")
        print(f"  Hardware ID: {port.hwid}")
        print()
    
    return [port.device for port in ports]

def test_arduino_communication(port_device):
    """Test if port responds to Arduino servo commands"""
    print(f"Testing Arduino communication on {port_device}...")
    
    try:
        ser = serial.Serial(port_device, 115200, timeout=2)
        time.sleep(3)  # Arduino reset time
        
        # Clear buffers
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Send test command
        print("  Sending CENTER command...")
        ser.write(b'CENTER\n')
        ser.flush()
        
        # Wait for response
        response = ""
        start_time = time.time()
        while time.time() - start_time < 3:
            # THE FIX IS ON THE NEXT LINE:
            # Removed the () after ser.in_waiting
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                response += data
                if 'OK' in response or 'ERROR' in response:
                    break
            time.sleep(0.1)
        
        ser.close()
        
        print(f"  Response: '{response.strip()}'")
        
        if 'OK' in response:
            print(f"  ✓ Arduino servo bridge found on {port_device}")
            return True
        else:
            print(f"  ✗ No Arduino servo response on {port_device}")
            return False
            
    except Exception as e:
        print(f"  ✗ Failed to connect: {e}")
        return False

def main():
    print("Arduino Port Finder")
    print("This will help locate your Arduino servo controller")
    print()
    
    # List all ports
    available_ports = find_all_ports()
    
    if not available_ports:
        print("No serial ports found. Check:")
        print("1. Arduino is connected via USB")
        print("2. Arduino drivers are installed")
        print("3. Arduino sketch is uploaded")
        return
    
    # Test each port for Arduino servo bridge
    print("=== Testing Ports for Arduino Servo Bridge ===")
    working_ports = []
    
    for port in available_ports:
        if test_arduino_communication(port):
            working_ports.append(port)
        print()
    
    print("=== Results ===")
    if working_ports:
        print("Arduino servo bridge found on:")
        for port in working_ports:
            print(f"  {port}")
        print()
        print("Use this port in your tracking system:")
        print(f"  python3 complete_tracking_with_servos.py --port {working_ports[0]}")
    else:
        print("No Arduino servo bridge found.")
        print()
        print("Troubleshooting:")
        print("1. Check Arduino USB connection")
        print("2. Verify Arduino sketch is uploaded")
        print("3. Check that Arduino sketch matches the servo bridge code")
        print("4. Try different USB ports/cables")

if __name__ == "__main__":
    main()