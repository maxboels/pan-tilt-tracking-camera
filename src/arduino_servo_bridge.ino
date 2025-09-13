/*
 Arduino Servo Bridge for existing PCA9685 setup
 Controls the same PCA9685 module that's normally controlled by Jetson
 
 This version is optimized for faster serial communication by using minimal responses.
*/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Initialize the PCA9685 driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// --- Servo Parameters ---
// Adjust these values to match your specific servos' pulse width range.
// These values correspond to the servo's travel, typically -90 to 90 degrees.
#define SERVO_MIN 150  // Minimum pulse length count (out of 4096)
#define SERVO_MAX 600  // Maximum pulse length count (out of 4096)

void setup() {
  // Start serial communication at a high baud rate for fast data transfer
  Serial.begin(115200);
  Serial.println("Arduino PCA9685 Servo Bridge Ready");

  // Initialize the PWM driver
  pwm.begin();
  // Set the PWM frequency to 50Hz, which is standard for analog servos
  pwm.setPWMFreq(50);
  
  delay(10); // Short delay to allow everything to initialize
}

void loop() {
  // Check if there is data available to read from the serial port
  if (Serial.available() > 0) {
    // Read the incoming command until a newline character is received
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove any leading/trailing whitespace

    // --- Command Parser ---

    // Check for the primary servo control command: "SERVO,channel,angle"
    if (command.startsWith("SERVO,")) {
      // Find the positions of the commas to parse the channel and angle
      int firstComma = command.indexOf(',');
      int secondComma = command.indexOf(',', firstComma + 1);
      
      // Ensure the command format is valid (contains two commas)
      if (firstComma > 0 && secondComma > firstComma) {
        // Extract the channel number (as an integer)
        int channel = command.substring(firstComma + 1, secondComma).toInt();
        // Extract the angle (as a float)
        float angle = command.substring(secondComma + 1).toFloat();
        
        // Map the angle (-90 to 90 degrees) to the servo's pulse width range
        int pulseWidth = map(angle, -90, 90, SERVO_MIN, SERVO_MAX);
        // Constrain the pulse width to the defined min/max to prevent servo damage
        pulseWidth = constrain(pulseWidth, SERVO_MIN, SERVO_MAX);
        
        // Set the PWM for the target channel
        pwm.setPWM(channel, 0, pulseWidth);
        
        // OPTIMIZATION: Send a minimal "OK" response for speed.
        // This reduces serial latency, which is critical for real-time control.
        Serial.println("OK");
      } else {
        // If the command format is wrong, send a minimal error response.
        Serial.println("ERR");
      }
    } 
    // Check for the "CENTER" command to center both servos
    else if (command == "CENTER") {
      int centerPulse = (SERVO_MIN + SERVO_MAX) / 2;
      pwm.setPWM(0, 0, centerPulse); // Center channel 0
      pwm.setPWM(1, 0, centerPulse); // Center channel 1
      Serial.println("OK: Servos centered");
    } 
    // Check for the "OFF" command to disable servo outputs
    else if (command == "OFF") {
      // Setting the pulse width to 0 effectively disables the PWM signal
      pwm.setPWM(0, 0, 0);
      pwm.setPWM(1, 0, 0);
      Serial.println("OK: Servos disabled");
    }
    
    // NOTE: Detailed error messages and other commands like STATUS and TRACK
    // have been removed to reduce code size and serial traffic for performance.
  }
}