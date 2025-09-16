#!/usr/bin/env python3
"""
Fix Pan Inversion Script
Updates calibration.py to remove the negative sign in pan calculation
while keeping inverted_pan=True in config to reflect the physical setup
"""

import os
import sys
import re

def find_calibration_file():
    """Find the calibration.py file"""
    calibration_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'calibration.py')
    if os.path.exists(calibration_path):
        return calibration_path
    return None

def update_calibration_code(file_path):
    """Update the calibration.py file to fix pan inversion"""
    print(f"Updating calibration file: {file_path}")
    
    try:
        # Read current file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if the negative sign exists in the pan_angle calculation
        pan_angle_pattern = r'pan_angle\s*=\s*-\s*pan_error\s*\*'
        if not re.search(pan_angle_pattern, content):
            print("⚠️ Could not find the expected pan_angle calculation pattern.")
            print("The file may have already been modified or uses different code.")
            return False
        
        # Replace the negative sign with positive in pan_angle calculation
        modified_content = re.sub(
            pan_angle_pattern,
            'pan_angle = pan_error *',  # Removed negative sign
            content
        )
        
        # Save the modified file
        with open(file_path, 'w') as f:
            f.write(modified_content)
        
        print("✅ Calibration code updated: Removed negative sign from pan_angle calculation")
        print("\nExplanation:")
        print("The tracking test showed that pan directions were inverted.")
        print("This happened because there were two inversions occurring:")
        print("1. In calibration.py: pan_angle = -pan_error * 0.1  (negative sign)")
        print("2. In servo_controller.py: With inverted_pan=True, pan_direction=-1")
        print("   This creates calibrated_angle = angle * -1")
        print("\nBy removing the negative sign from the calibration code, we keep")
        print("inverted_pan=True (accurately reflecting your physical setup) while")
        print("fixing the double inversion problem.")
        
        return True
    
    except Exception as e:
        print(f"❌ Error updating calibration file: {e}")
        return False

def main():
    """Main function"""
    print("=== Fix Pan Inversion (While Keeping inverted_pan=True) ===")
    print("This script will fix the inverted pan servo direction by updating")
    print("the calibration code rather than changing the config.")
    print("\nBenefits of this approach:")
    print("- Config file accurately reflects your physical setup (inverted pan servo)")
    print("- Prevents double inversion in the code")
    print("- Makes the tracking math more intuitive")
    
    response = input("Update calibration.py to fix pan inversion? (y/n): ").lower()
    
    if response in ['y', 'yes']:
        calibration_file = find_calibration_file()
        if calibration_file:
            update_calibration_code(calibration_file)
            print("\nTo verify the fix:")
            print("1. Run the tracking simulation test again:")
            print("   python evals/test_tracking_simulation.py")
            print("2. Check that pan movements are now correct")
        else:
            print("❌ Could not find calibration.py file")
    else:
        print("Operation cancelled")
        print("\nAlternative: You can still set inverted_pan=False in config.json")
        print("to fix the issue if you prefer that approach.")

if __name__ == "__main__":
    main()
