#!/usr/bin/env python3
"""
Test Servo Inversion Script
This script helps determine the correct inverted_pan setting for your servo
"""

import sys
import os
import time
import json

# Add parent directory to path to import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.servo_controller import ArduinoServoController

def load_config():
    """Load current configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config, config_path
    except Exception as e:
        print(f"Error loading config: {e}")
        return {"servo": {"inverted_pan": False}}, config_path

def update_config(config_path, inverted_pan):
    """Update config file with new inverted_pan setting"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update the inverted_pan setting
        if "servo" not in config:
            config["servo"] = {}
        config["servo"]["inverted_pan"] = inverted_pan
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Config updated: inverted_pan = {inverted_pan}")
        return True
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def test_servo_direction(inverted):
    """Test servo with specific inversion setting"""
    print(f"\n=== Testing with inverted_pan={inverted} ===")
    controller = ArduinoServoController(inverted_pan=inverted)
    
    if not controller.connect():
        print("❌ Could not connect to servo controller")
        return False
    
    try:
        # Center servos first
        print("Centering servos...")
        controller.center_servos()
        time.sleep(2)
        
        # Test rightward movement
        print("Moving camera RIGHT (test movement)...")
        controller.move_servo(0, 30)  # Pan servo channel, +30 degrees
        time.sleep(2)
        
        # Ask user for confirmation
        response = input("Did the camera PHYSICALLY turn RIGHT? (y/n): ").lower()
        turns_right = response in ['y', 'yes']
        
        # Back to center
        print("Returning to center...")
        controller.center_servos()
        time.sleep(1)
        
        # Test leftward movement
        print("Moving camera LEFT (test movement)...")
        controller.move_servo(0, -30)  # Pan servo channel, -30 degrees
        time.sleep(2)
        
        # Ask user for confirmation
        response = input("Did the camera PHYSICALLY turn LEFT? (y/n): ").lower()
        turns_left = response in ['y', 'yes']
        
        # Back to center
        print("Returning to center...")
        controller.center_servos()
        
        # Determine if the setting was correct
        is_correct = turns_right and turns_left
        print(f"\nResults with inverted_pan={inverted}:")
        print(f"  Right command → Camera turned right: {'✅' if turns_right else '❌'}")
        print(f"  Left command → Camera turned left: {'✅' if turns_left else '❌'}")
        print(f"  Overall: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        
        return is_correct
        
    finally:
        controller.disconnect()

def main():
    """Main test function"""
    print("=== Pan Servo Inversion Test ===")
    print("This test will help determine if your pan servo should be inverted or not.")
    print("The servo will move in different directions to test both settings.")
    
    # Load current config
    config, config_path = load_config()
    current_setting = config.get("servo", {}).get("inverted_pan", False)
    print(f"Current config setting: inverted_pan = {current_setting}")
    
    # Test with current setting first
    print("\nStep 1: Testing with CURRENT setting")
    current_is_correct = test_servo_direction(current_setting)
    
    if current_is_correct:
        print("\n✅ RESULT: Your current setting is CORRECT!")
        print(f"Keep inverted_pan = {current_setting} in your config.")
        return
    
    # If current setting was wrong, test with opposite setting
    print("\nStep 2: Testing with OPPOSITE setting")
    opposite_setting = not current_setting
    opposite_is_correct = test_servo_direction(opposite_setting)
    
    if opposite_is_correct:
        print(f"\n✅ RECOMMENDATION: Change to inverted_pan = {opposite_setting}")
        response = input("Would you like to update the config file now? (y/n): ").lower()
        if response in ['y', 'yes']:
            update_config(config_path, opposite_setting)
    else:
        print("\n❓ INCONCLUSIVE: Neither setting worked correctly.")
        print("This could indicate an issue with the servo wiring or Arduino firmware.")
        print("Check the servo connections and try again.")

if __name__ == "__main__":
    main()
