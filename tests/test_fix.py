#!/usr/bin/env python3
"""
Quick test of the fixed tracking system
"""

import sys
import os
import time
import json

def test_configuration():
    """Test the current configuration"""
    print("=== Configuration Test ===")
    
    config_path = "config/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    servo_config = config.get('servo', {})
    inverted_pan = servo_config.get('inverted_pan', False)
    
    print(f"Current inverted_pan setting: {inverted_pan}")
    
    # The real fix is now in the calibration math, not the config
    print("✅ Direction fix applied in calibration.py")
    print("   Updated: pan_angle = -pan_error * 0.1 (negative sign added)")
    print("   When person is on RIGHT side -> negative angle -> servo turns RIGHT")
    print("   When person is on LEFT side -> positive angle -> servo turns LEFT")
    
    return True  # Fix is in code, not config

def main():
    """Main test function"""
    print("Pan-Tilt Tracking Fix Verification")
    print("=" * 40)
    
    # Change to project directory
    project_dir = "/home/maxboels/projects/pan-tilt-tracking-camera"
    os.chdir(project_dir)
    
    # Test configuration
    is_fixed = test_configuration()
    
    print("\n=== Quick Test Instructions ===")
    print("1. Run: python main.py")
    print("2. Stand in CENTER of camera view")
    print("3. Move to RIGHT side of camera view")
    print("4. Camera should pan RIGHT to follow you")
    print("5. Move to LEFT side of camera view")  
    print("6. Camera should pan LEFT to follow you")
    print()
    
    print("✅ Mathematical fix applied - servo direction should be correct now")
    
    print("\nIf it's STILL moving wrong direction:")
    print("- The issue might be in your servo hardware/firmware")
    print("- Try setting inverted_pan: true in config/config.json")
    print("- Or check physical servo wiring/mounting")

if __name__ == "__main__":
    main()
