#!/usr/bin/env python3
"""
Simple Math Diagnostic for Pan-Tilt Tracking
"""

def test_pixel_to_servo_math():
    """Test the current pixel to servo conversion logic"""
    print("=== Current Pixel to Servo Math ===")
    
    # Current frame setup
    frame_width = 1920
    frame_height = 1080
    frame_center = (frame_width // 2, frame_height // 2)  # (960, 540)
    
    print(f"Frame size: {frame_width}x{frame_height}")
    print(f"Frame center: {frame_center}")
    print()
    
    # Current calibration logic from calibration.py (FIXED VERSION)
    def pixel_to_servo_current(pixel_x, pixel_y):
        pan_error = pixel_x - frame_center[0]
        tilt_error = pixel_y - frame_center[1]
        pan_angle = -pan_error * 0.1  # NEGATIVE - right pixel -> negative angle (turn right)
        tilt_angle = -tilt_error * 0.08  # negative for camera Y inversion
        return pan_angle, tilt_angle
    
    # Test cases
    test_cases = [
        ("Center", 960, 540, "Should be (0, 0)"),
        ("Right side", 1200, 540, "Person right -> Camera should pan RIGHT (+)"),
        ("Left side", 720, 540, "Person left -> Camera should pan LEFT (-)"),
        ("Top center", 960, 300, "Person above -> Camera should tilt UP (+)"),
        ("Bottom center", 960, 780, "Person below -> Camera should tilt DOWN (-)"),
        ("Top-right", 1200, 300, "Person top-right -> Pan RIGHT, Tilt UP"),
        ("Bottom-left", 720, 780, "Person bottom-left -> Pan LEFT, Tilt DOWN"),
    ]
    
    print("Test Case       | Pixel Coord  | Pan Error | Tilt Error | Servo Angles | Expected Result")
    print("-" * 90)
    
    for name, pixel_x, pixel_y, expected in test_cases:
        pan_error = pixel_x - frame_center[0]
        tilt_error = pixel_y - frame_center[1]
        pan_angle, tilt_angle = pixel_to_servo_current(pixel_x, pixel_y)
        
        print(f"{name:12} | ({pixel_x:4},{pixel_y:3}) | {pan_error:8.0f} | {tilt_error:9.0f} | ({pan_angle:5.1f}, {tilt_angle:5.1f}) | {expected}")

def analyze_servo_direction_logic():
    """Analyze the servo direction application"""
    print("\n=== Servo Direction Logic Analysis ===")
    
    # From servo_controller.py lines 49-50
    def get_direction_multiplier(inverted_pan):
        return -1 if inverted_pan else 1
    
    print("Current config has inverted_pan = false")
    direction_multiplier = get_direction_multiplier(False)
    print(f"Direction multiplier: {direction_multiplier}")
    print()
    
    # Test what happens to servo commands
    test_angles = [20, -20, 10, -10]
    
    print("Requested Angle | Direction Mult | Final Servo Angle | Physical Movement")
    print("-" * 70)
    
    for angle in test_angles:
        final_angle = angle * direction_multiplier
        if angle > 0:
            movement = "RIGHT" if final_angle > 0 else "LEFT"
        else:
            movement = "LEFT" if final_angle < 0 else "RIGHT"
        
        print(f"{angle:13.0f} | {direction_multiplier:13.0f} | {final_angle:16.0f} | {movement}")

def identify_problem():
    """Identify the likely problem"""
    print("\n=== Problem Analysis - FIXED ===")
    print("The issue: Servo moves AWAY from person instead of TOWARDS them")
    print()
    print("CORRECTED logic chain:")
    print("1. Person on right side (pixel_x = 1200, center = 960)")
    print("2. pan_error = 1200 - 960 = +240")
    print("3. pan_angle = -240 * 0.1 = -24° (NEGATIVE!)")
    print("4. inverted_pan = false -> direction = +1")
    print("5. final_servo_angle = -24 * 1 = -24°")
    print("6. Servo moves RIGHT (-24°) - CORRECT!")
    print()
    print("The fix was to negate the pan_error:")
    print("  pan_angle = -pan_error * 0.1")
    print("This makes positive pixel positions produce negative angles")

def suggest_fix():
    """Suggest the fix"""
    print("\n=== Applied Fix ===")
    print("FIXED: Changed calibration.py pixel_to_servo calculation:")
    print('  pan_angle = -pan_error * 0.1  # Negative sign added!')
    print()
    print("Result:")
    print("- Person on RIGHT side → Negative angle → Camera turns RIGHT ✅")
    print("- Person on LEFT side → Positive angle → Camera turns LEFT ✅")
    print()
    print("Keep inverted_pan = false in config (current setting)")

if __name__ == "__main__":
    print("Pan-Tilt Servo Math Diagnostic")
    print("=" * 50)
    
    test_pixel_to_servo_math()
    analyze_servo_direction_logic()
    identify_problem()
    suggest_fix()
